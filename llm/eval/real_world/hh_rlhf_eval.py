"""Real-world observability diagnostics for HH-RLHF.

For real preference data, ``chosen_is_good`` is always True (human-labelled),
so the standard resume-style linear probe doesn't apply.  We use two adapted
diagnostics instead:

1. **Margin probe** — OLS: rm_margin ~ demo_B + response_length + prompt_length.
   A negative, significant ``demo_B`` coefficient means the RM gives lower
   confidence scores to Group-B pairs even after controlling for length — the
   coverage-shortcut signature.

2. **Per-group accuracy** — fraction of pairs where score(chosen) > score(rejected),
   split by demographic group.  A gap here is direct evidence of the bias.

These are reported alongside the existing content-controlled linear probe from
``observability_audit.py`` (which works whenever ``quality_score`` is available).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats

from llm.models.reward_model import RewardModel
from llm.utils.logging import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Scoring helper
# --------------------------------------------------------------------------- #

@torch.no_grad()
def score_pairs(
    rm: RewardModel,
    df: pd.DataFrame,
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score all (prompt+chosen) and (prompt+rejected) pairs.

    Returns:
        scores_chosen, scores_rejected, margins  — each shape (N,)
    """
    device = next(rm.model.parameters()).device

    def _batch_score(texts: list[str]) -> np.ndarray:
        out: list[float] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = rm.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            s = rm.score(enc["input_ids"], enc["attention_mask"])
            out.extend(s.cpu().float().tolist())
        return np.array(out, dtype=np.float64)

    chosen_texts = [p + c for p, c in zip(df["prompt"], df["chosen"])]
    rejected_texts = [p + r for p, r in zip(df["prompt"], df["rejected"])]
    sc = _batch_score(chosen_texts)
    sr = _batch_score(rejected_texts)
    return sc, sr, sc - sr


# --------------------------------------------------------------------------- #
# Margin probe (adapted for real data where chosen_is_good ≡ True)
# --------------------------------------------------------------------------- #

def run_margin_probe(
    df: pd.DataFrame,
    margins: np.ndarray,
) -> dict[str, Any]:
    """OLS regression: rm_margin ~ demo_B + response_length + prompt_length.

    A negative, significant ``demo_B`` coefficient indicates the RM assigns
    lower confidence to Group-B pairs independent of response length — the
    coverage-shortcut signature in real data.
    """
    n = len(df)
    demo_b = (df["demographic_signal"] == "B").astype(float).values
    resp_len = np.array([len(str(c).split()) for c in df["chosen"]], dtype=np.float64)
    prompt_len = np.array([len(str(p).split()) for p in df["prompt"]], dtype=np.float64)

    y = margins.astype(np.float64)
    X = np.column_stack([np.ones(n), demo_b, resp_len, prompt_len])
    names = ["intercept", "demo_B", "response_length", "prompt_length"]
    p = X.shape[1]

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    rss = float(np.sum((y - y_hat) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else 0.0

    sigma2 = rss / max(n - p, 1)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t_vals = beta / np.where(se > 0, se, 1.0)
    p_vals = 2.0 * (1.0 - stats.t.cdf(np.abs(t_vals), df=max(n - p, 1)))

    shortcut_detected = bool(
        beta[1] < 0 and p_vals[1] < 0.05
    )

    log.info(
        "Margin probe",
        demo_B_coeff=f"{beta[1]:.4f}",
        demo_B_pval=f"{p_vals[1]:.4f}",
        r_squared=f"{r2:.4f}",
        shortcut_detected=shortcut_detected,
    )
    return {
        "coefficients": dict(zip(names, beta.tolist())),
        "t_stats": dict(zip(names[1:], t_vals[1:].tolist())),
        "p_values": dict(zip(names[1:], p_vals[1:].tolist())),
        "r_squared": r2,
        "n_obs": n,
        "shortcut_detected": shortcut_detected,
    }


# --------------------------------------------------------------------------- #
# Per-group accuracy
# --------------------------------------------------------------------------- #

def run_group_accuracy(
    df: pd.DataFrame,
    margins: np.ndarray,
) -> dict[str, Any]:
    """Compute per-group RM accuracy (chosen scored higher than rejected)."""
    correct = (margins > 0).astype(int)
    results: dict[str, Any] = {}

    for grp in ["A", "B"]:
        mask = df["demographic_signal"] == grp
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n > 0 else float("nan")
        results[f"accuracy_{grp}"] = acc
        results[f"n_{grp}"] = n

    gap = results.get("accuracy_A", float("nan")) - results.get("accuracy_B", float("nan"))
    results["accuracy_gap_A_minus_B"] = gap

    log.info(
        "Per-group accuracy",
        accuracy_A=f"{results['accuracy_A']:.3f}",
        accuracy_B=f"{results['accuracy_B']:.3f}",
        gap=f"{gap:.3f}",
    )
    return results


# --------------------------------------------------------------------------- #
# Full audit for one RM checkpoint
# --------------------------------------------------------------------------- #

def audit_rm(
    rm_checkpoint: str,
    audit_df: pd.DataFrame,
    max_length: int,
    batch_size: int,
    dtype: str = "float32",
    device_map: str | None = None,
    label: str = "rm",
) -> dict[str, Any]:
    """Load an RM and run the margin probe + group accuracy diagnostic."""
    log.info("Loading RM for audit", checkpoint=rm_checkpoint, label=label)
    rm = RewardModel.from_pretrained(rm_checkpoint, dtype=dtype, device_map=device_map)
    rm.eval()

    sc, sr, margins = score_pairs(rm, audit_df, max_length, batch_size)

    margin_results = run_margin_probe(audit_df, margins)
    accuracy_results = run_group_accuracy(audit_df, margins)

    del rm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "label": label,
        "margin_probe": margin_results,
        "group_accuracy": accuracy_results,
        "mean_margin_A": float(margins[audit_df["demographic_signal"] == "A"].mean()),
        "mean_margin_B": float(margins[audit_df["demographic_signal"] == "B"].mean()),
    }
