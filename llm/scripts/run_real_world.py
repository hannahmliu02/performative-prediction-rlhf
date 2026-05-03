"""Section 5.2 — Real-world evaluation on HH-RLHF (single-round design).

Implements the paper's two real-world claims:
  1. Preconditions hold  — the observability shortcut is detectable in a biased
     RM trained on HH-RLHF (margin probe: demo_B coefficient < 0, p < 0.05).
  2. Mitigation transfers — an IPW-corrected RM eliminates the shortcut, and
     a policy trained against it improves DecodingTrust fairness/stereotype
     scores relative to the biased baseline.

Design decision (docs/decisions.md §2026-04-22):
  Do NOT simulate a multi-round loop on real data.  One biased RM, one
  corrected RM, one policy each, one DecodingTrust evaluation each.

Usage:
    uv run python -m llm.scripts.run_real_world \
        --config llm/configs/data/real_world_hh_rlhf.yaml
"""

from __future__ import annotations

import argparse
import json
import pathlib
import textwrap

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from llm.eval.decoding_trust import run_decoding_trust
from llm.eval.real_world.hh_rlhf_eval import audit_rm
from llm.mitigation.propensity import OraclePropensityModel as OraclePropensity
from llm.training.dpo import train_dpo
from llm.training.train_rm import train_rm
from llm.utils.logging import get_logger

log = get_logger(__name__)

_OUT = pathlib.Path("llm/outputs/real_world")


# --------------------------------------------------------------------------- #
# Config builders
# --------------------------------------------------------------------------- #

def _rm_cfg(
    data_path: str,
    exp_name: str,
    base: DictConfig,
    *,
    mitigation: dict | None = None,
) -> DictConfig:
    cfg: dict = {
        "experiment_name": exp_name,
        "model_name_or_path": str(base.sft_checkpoint),
        "dtype": str(base.dtype),
        "device_map": base.get("device_map") or None,
        "data_path": data_path,
        "max_length": int(base.max_length),
        "num_train_epochs": 1,
        "max_steps": int(base.rm_max_steps),
        "per_device_train_batch_size": int(base.rm_batch_size),
        "per_device_eval_batch_size": int(base.rm_batch_size) * 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": float(base.rm_lr),
        "warmup_ratio": 0.0,
        "eval_fraction": float(base.rm_eval_fraction),
        "report_to": "none",
        "seed": int(base.seed),
        "output_dir": str(_OUT / "checkpoints"),
    }
    if mitigation:
        cfg["mitigation"] = mitigation
    return OmegaConf.create(cfg)


def _dpo_cfg(
    data_path: str,
    exp_name: str,
    base: DictConfig,
    sft_checkpoint: str,
    *,
    mitigation: dict | None = None,
) -> DictConfig:
    cfg: dict = {
        "experiment_name": exp_name,
        "sft_checkpoint": sft_checkpoint,
        "model_name_or_path": sft_checkpoint,
        "dtype": str(base.get("policy_dtype", base.dtype)),
        "device_map": base.get("device_map") or None,
        "data_path": data_path,
        "max_length": int(base.get("dpo_max_length", base.max_length)),
        "num_train_epochs": 1,
        "max_steps": int(base.policy_max_steps),
        "per_device_train_batch_size": int(base.policy_batch_size),
        "per_device_eval_batch_size": int(base.policy_batch_size) * 2,
        "gradient_accumulation_steps": int(base.get("gradient_accumulation_steps", 1)),
        "gradient_checkpointing": bool(base.get("gradient_checkpointing", False)),
        "learning_rate": float(base.policy_lr),
        "warmup_steps": 0,
        "eval_fraction": 0.1,
        "beta": float(base.policy_beta),
        "length_norm_beta": 0.0,
        "report_to": "none",
        "seed": int(base.seed),
        "output_dir": str(_OUT / "checkpoints"),
    }
    if mitigation:
        cfg["mitigation"] = mitigation
    return OmegaConf.create(cfg)


def _dt_cfg(policy_checkpoint: str, exp_name: str, base: DictConfig) -> DictConfig:
    return OmegaConf.create({
        "experiment_name": exp_name,
        "model_checkpoint": policy_checkpoint,
        "subsets": ["stereotype", "fairness"],
        "max_samples_per_subset": int(base.get("dt_max_samples", 50)),
        "max_new_tokens": 30,
        "batch_size": int(base.get("generation_batch_size", 8)),
        "dtype": str(base.dtype),
        "device_map": base.get("device_map") or None,
        "seed": int(base.seed),
        "output_dir": str(_OUT / "dt_cache"),
    })


# --------------------------------------------------------------------------- #
# IPW weight helper
# --------------------------------------------------------------------------- #

def _compute_ipw_weights(
    train_df: pd.DataFrame,
    obs_probs: dict[str, float],
    eps: float = 0.05,
) -> list[float]:
    """Oracle IPW weights: 1 / p_obs(demographic_signal), clipped to [eps, 1/eps]."""
    prop = OraclePropensity(obs_probs)
    psi = prop.predict_proba(train_df)
    weights = 1.0 / np.clip(psi, eps, 1.0 - eps)
    # Normalise so mean weight = 1 (preserves loss scale)
    weights = weights / weights.mean()
    return weights.tolist()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run(cfg: DictConfig) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)

    data_dir = pathlib.Path(cfg.output_dir)
    train_path = str(data_dir / "train.parquet")
    audit_path = str(data_dir / "audit.parquet")

    if not pathlib.Path(train_path).exists():
        raise FileNotFoundError(
            f"HH-RLHF train data not found at {train_path}. "
            "Run: uv run python -m llm.data.load_real_world "
            "--config llm/configs/data/real_world_hh_rlhf.yaml"
        )

    log.info("Loading datasets", train=train_path, audit=audit_path)
    train_df = pd.read_parquet(train_path)
    audit_df = pd.read_parquet(audit_path)
    log.info(
        "Dataset loaded",
        train_A=int((train_df["demographic_signal"] == "A").sum()),
        train_B=int((train_df["demographic_signal"] == "B").sum()),
        audit_A=int((audit_df["demographic_signal"] == "A").sum()),
        audit_B=int((audit_df["demographic_signal"] == "B").sum()),
    )

    obs_probs: dict[str, float] = OmegaConf.to_container(
        cfg.obs_probs.demographic_signal, resolve=True
    )  # type: ignore[assignment]

    dtype = str(cfg.dtype)
    device_map = cfg.get("device_map") or None
    max_length = int(cfg.max_length)
    batch_size = int(cfg.rm_batch_size) * 2

    results: dict = {}

    # ── Step 1: Train biased RM ───────────────────────────────────────────────
    log.info("=== Step 1: Training biased RM ===")
    biased_rm_ckpt = _OUT / "checkpoints" / "rw_rm_biased"
    if not (biased_rm_ckpt / "config.json").exists():
        rm_ckpt = train_rm(_rm_cfg(train_path, "rw_rm_biased", cfg))
        biased_rm_ckpt = rm_ckpt
    else:
        log.info("Biased RM checkpoint exists, skipping training")
    biased_rm_ckpt = str(biased_rm_ckpt)

    # ── Step 2: Audit biased RM (precondition probe) ──────────────────────────
    log.info("=== Step 2: Auditing biased RM (precondition probe) ===")
    biased_audit = audit_rm(
        biased_rm_ckpt, audit_df,
        max_length=max_length, batch_size=batch_size,
        dtype=dtype, device_map=device_map, label="biased",
    )
    results["biased_rm_audit"] = biased_audit
    shortcut_detected = biased_audit["margin_probe"]["shortcut_detected"]
    log.info(
        "Precondition check",
        shortcut_detected=shortcut_detected,
        demo_B_coeff=f"{biased_audit['margin_probe']['coefficients'].get('demo_B', float('nan')):.4f}",
        p_value=f"{biased_audit['margin_probe']['p_values'].get('demo_B', float('nan')):.4f}",
        accuracy_gap=f"{biased_audit['group_accuracy']['accuracy_gap_A_minus_B']:.3f}",
    )

    # ── Step 3: Train IPW-corrected RM ───────────────────────────────────────
    log.info("=== Step 3: Training IPW-corrected RM ===")
    corrected_rm_ckpt = _OUT / "checkpoints" / "rw_rm_corrected"
    ipw_mitigation = {
        "type": "ipw_counterfactual",
        "propensity_variant": "oracle",
        "obs_probs": obs_probs,
    }
    if not (corrected_rm_ckpt / "config.json").exists():
        rm_ckpt = train_rm(_rm_cfg(
            train_path, "rw_rm_corrected", cfg,
            mitigation=ipw_mitigation,
        ))
        corrected_rm_ckpt = rm_ckpt
    else:
        log.info("Corrected RM checkpoint exists, skipping training")
    corrected_rm_ckpt = str(corrected_rm_ckpt)

    # ── Step 4: Audit corrected RM (verify shortcut removed) ─────────────────
    log.info("=== Step 4: Auditing corrected RM ===")
    corrected_audit = audit_rm(
        corrected_rm_ckpt, audit_df,
        max_length=max_length, batch_size=batch_size,
        dtype=dtype, device_map=device_map, label="corrected",
    )
    results["corrected_rm_audit"] = corrected_audit
    log.info(
        "Corrected RM audit",
        shortcut_detected=corrected_audit["margin_probe"]["shortcut_detected"],
        demo_B_coeff=f"{corrected_audit['margin_probe']['coefficients'].get('demo_B', float('nan')):.4f}",
        accuracy_gap=f"{corrected_audit['group_accuracy']['accuracy_gap_A_minus_B']:.3f}",
    )

    # ── Step 5: Train policies (biased and corrected) ─────────────────────────
    log.info("=== Step 5: Training policies ===")
    sft_checkpoint = str(cfg.sft_checkpoint)

    biased_policy_ckpt = _OUT / "checkpoints" / "rw_policy_biased"
    if not (biased_policy_ckpt / "config.json").exists():
        ckpt = train_dpo(_dpo_cfg(train_path, "rw_policy_biased", cfg, sft_checkpoint))
        biased_policy_ckpt = ckpt
    else:
        log.info("Biased policy checkpoint exists, skipping training")

    corrected_policy_ckpt = _OUT / "checkpoints" / "rw_policy_corrected"
    if not (corrected_policy_ckpt / "config.json").exists():
        dpo_ipw_mitigation = {
            "type": "ours_ipw",
            "propensity_variant": "oracle",
            "obs_probs": obs_probs,
        }
        ckpt = train_dpo(_dpo_cfg(train_path, "rw_policy_corrected", cfg, sft_checkpoint, mitigation=dpo_ipw_mitigation))
        corrected_policy_ckpt = ckpt
    else:
        log.info("Corrected policy checkpoint exists, skipping training")

    # ── Step 6: DecodingTrust evaluation ─────────────────────────────────────
    log.info("=== Step 6: DecodingTrust evaluation ===")
    dt_biased = run_decoding_trust(_dt_cfg(str(biased_policy_ckpt), "rw_dt_biased", cfg))
    dt_corrected = run_decoding_trust(_dt_cfg(str(corrected_policy_ckpt), "rw_dt_corrected", cfg))
    results["dt_biased"] = dt_biased
    results["dt_corrected"] = dt_corrected

    def _dt_scores(dt_out: dict) -> tuple[float, float]:
        subs = dt_out.get("subsets", {})
        fairness = float(subs.get("fairness", {}).get("demographic_parity_gap", float("nan")))
        stereotype = float(subs.get("stereotype", {}).get("agreement_rate", float("nan")))
        return fairness, stereotype

    fair_b, stereo_b = _dt_scores(dt_biased)
    fair_c, stereo_c = _dt_scores(dt_corrected)
    log.info(
        "DecodingTrust comparison",
        fairness_biased=f"{fair_b:.3f}",
        fairness_corrected=f"{fair_c:.3f}",
        stereotype_biased=f"{stereo_b:.3f}",
        stereotype_corrected=f"{stereo_c:.3f}",
    )

    # ── Step 7: Save results and write summary ────────────────────────────────
    log.info("=== Step 7: Writing summary ===")

    json_path = _OUT / "real_world_results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))

    gap_biased = biased_audit["group_accuracy"]["accuracy_gap_A_minus_B"]
    gap_corrected = corrected_audit["group_accuracy"]["accuracy_gap_A_minus_B"]

    summary = textwrap.dedent(f"""
    # Real-World Evaluation Summary (HH-RLHF)

    ## Precondition probe (Section 5.2, Claim 1)

    | Metric | Biased RM | Corrected RM (IPW) |
    |---|---|---|
    | `demo_B` margin probe coefficient | {biased_audit['margin_probe']['coefficients'].get('demo_B', float('nan')):.4f} | {corrected_audit['margin_probe']['coefficients'].get('demo_B', float('nan')):.4f} |
    | p-value | {biased_audit['margin_probe']['p_values'].get('demo_B', float('nan')):.4f} | {corrected_audit['margin_probe']['p_values'].get('demo_B', float('nan')):.4f} |
    | Shortcut detected (p < 0.05, coeff < 0) | {'✓' if shortcut_detected else '✗'} | {'✓' if corrected_audit['margin_probe']['shortcut_detected'] else '✗'} |
    | Group-A accuracy | {biased_audit['group_accuracy']['accuracy_A']:.3f} | {corrected_audit['group_accuracy']['accuracy_A']:.3f} |
    | Group-B accuracy | {biased_audit['group_accuracy']['accuracy_B']:.3f} | {corrected_audit['group_accuracy']['accuracy_B']:.3f} |
    | Accuracy gap (A − B) | {gap_biased:.3f} | {gap_corrected:.3f} |

    **Takeaway**: The biased RM {"shows" if shortcut_detected else "does NOT show"} the coverage shortcut
    (demo_B coefficient {"negative and significant" if shortcut_detected else "not significant"}).
    IPW {"eliminates" if not corrected_audit['margin_probe']['shortcut_detected'] else "reduces"} it.

    ## Mitigation transfer (Section 5.2, Claim 2)

    | Metric | Biased policy | Corrected policy (DPO⁺) | Δ |
    |---|---|---|---|
    | DecodingTrust fairness ↓ | {fair_b:.3f} | {fair_c:.3f} | {fair_b - fair_c:+.3f} |
    | DecodingTrust stereotype ↓ | {stereo_b:.3f} | {stereo_c:.3f} | {stereo_b - stereo_c:+.3f} |

    **Takeaway**: {"Corrected policy improves on both fairness and stereotype." if (fair_c < fair_b and stereo_c < stereo_b) else "Results mixed — see full JSON for details."}

    Full results: {json_path}
    """).strip()

    summary_path = _OUT / "real_world_summary.md"
    summary_path.write_text(summary)
    log.info("Summary written", path=str(summary_path))
    print("\n" + summary + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="llm/configs/data/real_world_hh_rlhf.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
