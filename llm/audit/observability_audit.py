"""Observability audit: linear probe, counterfactual delta, multi-step masking.

Analogue of audit.py from the CPM paper (Figure B5), adapted to reward model
bias in RLHF. Three diagnostics:

1. Linear probe — OLS: chosen_is_good ~ rm_margin + demo_B_indicator
2. Counterfactual delta — per (seniority × domain) stratum, Welch t-test
   comparing rm_margin between demographic groups A and B.
3. Multi-step masking — progressively mask k resume fields (k=0..10) and
   track mean score drift = score_k − score_0 per demographic group.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from scipy import stats

from llm.models.reward_model import RewardModel
from llm.utils.logging import get_logger
from llm.utils.seeding import set_seed

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = get_logger(__name__)

# Fields masked in order for k=1..5; k≥6 replaces the full candidate block.
_MASK_FIELDS: list[tuple[str, str]] = [
    ("Name", "Name: [CANDIDATE]"),
    ("Experience", "Experience: [MASKED]"),
    ("Domain", "Domain: [MASKED]"),
    ("Skills", "Skills: [MASKED]"),
    ("Projects", "Projects: [MASKED]"),
]

_GROUP_COLORS: dict[str, str] = {"A": "#4878d0", "B": "#ee854a"}


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


@torch.no_grad()
def _score_texts(
    rm: RewardModel,
    texts: list[str],
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """Return RM scalar scores for a list of texts as a 1-D float64 array."""
    device = next(rm.model.parameters()).device
    scores: list[float] = []
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
        scores.extend(s.cpu().float().tolist())
    return np.array(scores, dtype=np.float64)


def _mask_prompt(prompt: str, k: int) -> str:
    """Apply k-step progressive masking to a resume prompt string.

    k=0: unchanged.
    k=1: mask name.
    k=2: also mask experience.
    k=3: also mask domain.
    k=4: also mask skills.
    k=5: also mask projects.
    k≥6: replace everything from 'Name:' onward with '[candidate details masked]'.
    """
    if k == 0:
        return prompt
    if k >= 6:
        idx = prompt.find("Name:")
        return prompt[:idx] + "[candidate details masked]" if idx != -1 else prompt
    result = prompt
    for i, (field, replacement) in enumerate(_MASK_FIELDS, start=1):
        if k >= i:
            result = re.sub(rf"{field}: [^\n]+", replacement, result)
    return result


# --------------------------------------------------------------------------- #
# Analysis components
# --------------------------------------------------------------------------- #


def run_linear_probe(
    margins: np.ndarray,
    demo_b: np.ndarray,
    chosen_is_good: np.ndarray,
) -> dict[str, Any]:
    """OLS regression: chosen_is_good ~ rm_margin + demo_B_indicator.

    Returns coefficients, Wald t-stats, p-values, and R² — all computed
    analytically from the OLS covariance matrix (no statsmodels dependency).
    """
    n = len(chosen_is_good)
    y = chosen_is_good.astype(np.float64)
    X = np.column_stack([np.ones(n), margins, demo_b.astype(np.float64)])
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

    names = ["intercept", "rm_margin", "demo_B_indicator"]
    return {
        "coefficients": dict(zip(names, beta.tolist())),
        "t_stats": dict(zip(names[1:], t_vals[1:].tolist())),
        "p_values": dict(zip(names[1:], p_vals[1:].tolist())),
        "r_squared": float(r2),
        "n_obs": n,
    }


def _build_content_features(
    df: pd.DataFrame,
    margins: np.ndarray,
    demo_b: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Assemble the content-feature matrix used by both probes.

    Features (all float64):
        intercept, rm_margin, demo_B_indicator,
        response_length, prompt_length, quality_score

    response_length / prompt_length are word-count proxies (fast, no tokenizer).
    quality_score comes from df["quality_score"] when present; 0 otherwise.
    """
    n = len(df)
    resp_len = np.array([len(str(c).split()) for c in df["chosen"]], dtype=np.float64)
    prompt_len = np.array([len(str(p).split()) for p in df["prompt"]], dtype=np.float64)
    quality = df["quality_score"].values.astype(np.float64) if "quality_score" in df.columns else np.zeros(n)

    names = ["intercept", "rm_margin", "demo_B_indicator", "response_length", "prompt_length", "quality_score"]
    X = np.column_stack([
        np.ones(n),
        margins.astype(np.float64),
        demo_b.astype(np.float64),
        resp_len,
        prompt_len,
        quality,
    ])
    return X, names


def run_linear_probe_with_content(
    df: pd.DataFrame,
    margins: np.ndarray,
    demo_b: np.ndarray,
    chosen_is_good: np.ndarray,
) -> dict[str, Any]:
    """OLS: chosen_is_good ~ rm_margin + demo_B_indicator + response_length
                                        + prompt_length + quality_score.

    The key claim: demo_B_indicator coefficient remains negative and significant
    after controlling for content features, ruling out confounds where Group B
    responses are shorter or lower quality.
    """
    X, names = _build_content_features(df, margins, demo_b)
    n, p = X.shape
    y = chosen_is_good.astype(np.float64)

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

    return {
        "coefficients": dict(zip(names, beta.tolist())),
        "t_stats": dict(zip(names[1:], t_vals[1:].tolist())),
        "p_values": dict(zip(names[1:], p_vals[1:].tolist())),
        "r_squared": float(r2),
        "n_obs": n,
        "coverage_indicator_significant": bool(
            p_vals[names.index("demo_B_indicator")] < 0.05
            and beta[names.index("demo_B_indicator")] < 0
        ),
    }


def run_mlp_probe(
    df: pd.DataFrame,
    margins: np.ndarray,
    demo_b: np.ndarray,
    chosen_is_good: np.ndarray,
    n_background: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Non-linear MLP probe + SHAP feature attribution.

    Trains a small sklearn MLP on the content feature set, then uses SHAP
    PermutationExplainer to rank features by mean |SHAP| value.  Confirms the
    coverage indicator (demo_B_indicator) is among the top predictors in a
    non-linear model, ruling out a linear-specification artifact.

    Returns:
        mlp_accuracy: held-out accuracy of the MLP.
        shap_importance: dict {feature_name: mean_abs_shap} for class 1
            (chosen_is_good=1), sorted descending.
        coverage_indicator_rank: 1-based rank of demo_B_indicator by |SHAP|.
    """
    import shap
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    X, names = _build_content_features(df, margins, demo_b)
    # Drop the intercept column — sklearn adds its own bias
    X_feat = X[:, 1:]
    feat_names = names[1:]
    y = chosen_is_good.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, random_state=seed, stratify=y if y.sum() > 1 else None
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=seed)
    mlp.fit(X_train_s, y_train)
    accuracy = float(mlp.score(X_test_s, y_test))
    log.info("MLP probe trained", accuracy=f"{accuracy:.4f}")

    # SHAP via PermutationExplainer (faster than KernelExplainer for sklearn)
    bg_n = min(n_background, len(X_train_s))
    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(X_train_s), size=bg_n, replace=False)
    background = X_train_s[bg_idx]

    def _predict_proba_class1(X: np.ndarray) -> np.ndarray:
        return mlp.predict_proba(X)[:, 1]

    explainer = shap.PermutationExplainer(_predict_proba_class1, background)
    explain_n = min(200, len(X_test_s))
    shap_vals = explainer(X_test_s[:explain_n])  # Explanation object
    mean_abs = np.abs(shap_vals.values).mean(axis=0)  # (n_features,)

    shap_importance = dict(zip(feat_names, mean_abs.tolist()))
    ranked = sorted(shap_importance.items(), key=lambda kv: kv[1], reverse=True)
    rank = next((i + 1 for i, (k, _) in enumerate(ranked) if k == "demo_B_indicator"), None)

    log.info(
        "SHAP attribution complete",
        top_feature=ranked[0][0],
        coverage_indicator_rank=rank,
        demo_B_shap=f"{shap_importance.get('demo_B_indicator', float('nan')):.4f}",
    )

    return {
        "mlp_accuracy": accuracy,
        "shap_importance": dict(ranked),
        "coverage_indicator_rank": rank,
    }


def run_counterfactual_delta(
    df: pd.DataFrame,
    margins: np.ndarray,
) -> dict[str, Any]:
    """Within each (seniority × domain) stratum, compare rm_margin A vs B.

    Welch's t-test (unequal variances). Null: matched pairs treated equally.
    Rejection indicates demographic bias in the margin within that stratum.
    """
    df = df.copy()
    df["rm_margin"] = margins
    results: dict[str, Any] = {}
    for (sen, dom), group in df.groupby(["seniority", "domain"]):
        a_vals = group.loc[group["demographic_signal"] == "A", "rm_margin"].values
        b_vals = group.loc[group["demographic_signal"] == "B", "rm_margin"].values
        if len(a_vals) < 2 or len(b_vals) < 2:
            log.warning("Skipping stratum — too few samples", seniority=sen, domain=dom)
            continue
        t_stat, p_val = stats.ttest_ind(a_vals, b_vals, equal_var=False)
        results[f"{sen}__{dom}"] = {
            "n_A": int(len(a_vals)),
            "n_B": int(len(b_vals)),
            "mean_A": float(a_vals.mean()),
            "mean_B": float(b_vals.mean()),
            "delta": float(a_vals.mean() - b_vals.mean()),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
        }
    return results


def run_masking_analysis(
    df: pd.DataFrame,
    rm: RewardModel,
    max_length: int,
    batch_size: int,
    n_steps: int = 11,
) -> dict[str, Any]:
    """Progressive masking of k=0..n_steps-1 resume fields.

    score_k = RM score of (masked_prompt_k + chosen).
    drift_k = score_k - score_0.
    Reports mean drift ± SEM per demographic group per step.
    """
    prompts = df["prompt"].tolist()
    chosen = df["chosen"].tolist()
    demo = df["demographic_signal"].values

    base_texts = [p + c for p, c in zip(prompts, chosen)]
    scores_0 = _score_texts(rm, base_texts, max_length, batch_size)

    step_results: list[dict[str, Any]] = []
    for k in range(n_steps):
        masked = [_mask_prompt(p, k) for p in prompts]
        texts_k = [mp + c for mp, c in zip(masked, chosen)]
        scores_k = _score_texts(rm, texts_k, max_length, batch_size)
        drift = scores_k - scores_0

        row: dict[str, Any] = {"k": k}
        for grp in ("A", "B"):
            vals = drift[demo == grp]
            row[f"mean_drift_{grp}"] = float(vals.mean()) if len(vals) > 0 else 0.0
            row[f"sem_drift_{grp}"] = float(stats.sem(vals)) if len(vals) > 1 else 0.0
        step_results.append(row)
        log.info(
            "Masking step done",
            k=k,
            drift_A=f"{row['mean_drift_A']:.4f}",
            drift_B=f"{row['mean_drift_B']:.4f}",
        )

    return {"steps": step_results}


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #


def _save_figure(masking_results: dict[str, Any], output_path: pathlib.Path) -> None:
    steps = masking_results["steps"]
    fig, ax = plt.subplots(figsize=(8, 4))

    for grp in ("A", "B"):
        k_vals = [s["k"] for s in steps]
        means = [s[f"mean_drift_{grp}"] for s in steps]
        sems = [s[f"sem_drift_{grp}"] for s in steps]
        color = _GROUP_COLORS[grp]
        ax.plot(k_vals, means, marker="o", label=f"Group {grp}", color=color)
        ax.errorbar(k_vals, means, yerr=sems, fmt="none", color=color, capsize=3, alpha=0.6)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Masking step k (0 = unmasked, 5 = all fields masked, 6+ = fully masked)")
    ax.set_ylabel("Mean score drift (score_k − score_0)")
    ax.set_title("RM sensitivity to progressive demographic masking")
    ax.legend(title="Demographic group")
    ax.set_xticks(range(max(s["k"] for s in steps) + 1))
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #


def run_audit(cfg: DictConfig) -> dict[str, Any]:
    """Run the full observability audit and save results to disk.

    Args:
        cfg: OmegaConf config matching llm/configs/audit/*.yaml.

    Returns:
        Dict with keys: linear_probe, linear_probe_content, mlp_probe,
        counterfactual_delta, masking.
    """
    set_seed(cfg.seed, deterministic=False)

    log.info("Loading audit dataset", data_path=cfg.data_path)
    df = pd.read_parquet(cfg.data_path)
    max_samples: int | None = cfg.get("max_samples") or None
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=cfg.seed).reset_index(drop=True)
        log.info("Subsampled audit dataset", n=len(df))
    else:
        log.info("Audit dataset loaded", n=len(df))

    device_map = cfg.get("device_map") or None
    if device_map == "null":
        device_map = None

    log.info("Loading reward model", checkpoint=cfg.rm_checkpoint)
    rm = RewardModel.from_pretrained(
        cfg.rm_checkpoint,
        dtype=cfg.dtype,
        device_map=device_map,
    )
    rm.eval()

    log.info("Scoring chosen and rejected texts")
    chosen_texts = [p + c for p, c in zip(df["prompt"], df["chosen"])]
    rejected_texts = [p + r for p, r in zip(df["prompt"], df["rejected"])]
    scores_chosen = _score_texts(rm, chosen_texts, cfg.max_length, cfg.batch_size)
    scores_rejected = _score_texts(rm, rejected_texts, cfg.max_length, cfg.batch_size)
    margins = scores_chosen - scores_rejected

    demo_b = (df["demographic_signal"].values == "B").astype(np.float64)
    chosen_is_good = df["chosen_is_good"].values.astype(np.float64)

    log.info("Running linear probe")
    linear_probe = run_linear_probe(margins, demo_b, chosen_is_good)
    log.info(
        "Linear probe complete",
        r2=f"{linear_probe['r_squared']:.4f}",
        p_demo_B=f"{linear_probe['p_values']['demo_B_indicator']:.4f}",
    )

    # Optional content-controlled linear probe
    linear_probe_content: dict[str, Any] = {}
    if bool(cfg.get("run_content_controlled_probe", False)):
        log.info("Running content-controlled linear probe")
        linear_probe_content = run_linear_probe_with_content(df, margins, demo_b, chosen_is_good)
        log.info(
            "Content-controlled probe complete",
            r2=f"{linear_probe_content['r_squared']:.4f}",
            p_demo_B=f"{linear_probe_content['p_values']['demo_B_indicator']:.4f}",
            significant=linear_probe_content["coverage_indicator_significant"],
        )

    # Optional MLP + SHAP probe
    mlp_probe: dict[str, Any] = {}
    if bool(cfg.get("run_mlp_probe", False)):
        log.info("Running MLP probe + SHAP")
        mlp_probe = run_mlp_probe(df, margins, demo_b, chosen_is_good, seed=cfg.seed)

    log.info("Running counterfactual delta")
    counterfactual_delta = run_counterfactual_delta(df, margins)
    for stratum, vals in counterfactual_delta.items():
        log.info(
            "Counterfactual delta",
            stratum=stratum,
            delta=f"{vals['delta']:.4f}",
            p_value=f"{vals['p_value']:.4f}",
        )

    log.info("Running masking analysis", n_steps=cfg.n_masking_steps)
    masking = run_masking_analysis(df, rm, cfg.max_length, cfg.batch_size, cfg.n_masking_steps)

    results: dict[str, Any] = {
        "linear_probe": linear_probe,
        "linear_probe_content": linear_probe_content,
        "mlp_probe": mlp_probe,
        "counterfactual_delta": counterfactual_delta,
        "masking": masking,
    }

    output_dir = pathlib.Path(cfg.output_dir) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "audit_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    log.info("Saved audit results", path=str(json_path))

    fig_path = output_dir / "figure_b5_analogue.png"
    _save_figure(masking, fig_path)
    log.info("Saved masking figure", path=str(fig_path))

    return results
