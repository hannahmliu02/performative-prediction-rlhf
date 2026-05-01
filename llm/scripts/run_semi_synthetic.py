"""End-to-end semi-synthetic pipeline orchestrator.

Runs the full pipeline — data generation, SFT, RM training, policy training
(DPO or PPO), observability audit, DecodingTrust eval, and general-capability
benchmark — in sequence, using per-stage YAML sub-configs.

Usage:
    uv run python -m llm.scripts.run_semi_synthetic --config llm/configs/semi_synthetic_smoke.yaml
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from omegaconf import OmegaConf

from llm.utils.logging import get_logger
from llm.utils.seeding import set_seed

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full semi-synthetic pipeline.")
    parser.add_argument("--config", required=True, help="Path to top-level orchestration config.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(42, deterministic=False)

    results: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # 1. Data generation
    # ------------------------------------------------------------------ #
    if cfg.run_data:
        log.info("=== Stage: data generation")
        from llm.data.generate_resume_prefs import run as generate_prefs

        data_cfg = OmegaConf.load(cfg.data_config)
        generate_prefs(data_cfg)
        results["data"] = "ok"
        log.info("Data generation complete")

    # ------------------------------------------------------------------ #
    # 2. SFT
    # ------------------------------------------------------------------ #
    if cfg.run_sft:
        log.info("=== Stage: SFT")
        from llm.training.sft import train_sft

        sft_cfg = OmegaConf.load(cfg.sft_config)
        sft_ckpt = train_sft(sft_cfg)
        results["sft_checkpoint"] = str(sft_ckpt)
        log.info("SFT complete", checkpoint=str(sft_ckpt))

    # ------------------------------------------------------------------ #
    # 3. RM training
    # ------------------------------------------------------------------ #
    if cfg.run_rm:
        log.info("=== Stage: RM training")
        from llm.training.train_rm import train_rm

        rm_cfg = OmegaConf.load(cfg.rm_config)
        rm_ckpt = train_rm(rm_cfg)
        results["rm_checkpoint"] = str(rm_ckpt)
        log.info("RM training complete", checkpoint=str(rm_ckpt))

    # ------------------------------------------------------------------ #
    # 4. Policy training (DPO or PPO)
    # ------------------------------------------------------------------ #
    if cfg.run_policy:
        log.info("=== Stage: policy training", method=cfg.policy_method)
        if cfg.policy_method == "ppo":
            from llm.training.ppo import train_ppo

            policy_cfg = OmegaConf.load(cfg.ppo_config)
            policy_ckpt = train_ppo(policy_cfg)
        else:
            from llm.training.dpo import train_dpo

            policy_cfg = OmegaConf.load(cfg.dpo_config)
            policy_ckpt = train_dpo(policy_cfg)
        results["policy_checkpoint"] = str(policy_ckpt)
        results["policy_method"] = cfg.policy_method
        log.info("Policy training complete", checkpoint=str(policy_ckpt))

    # ------------------------------------------------------------------ #
    # 5. Observability audit
    # ------------------------------------------------------------------ #
    if cfg.run_audit:
        log.info("=== Stage: observability audit")
        from llm.audit.observability_audit import run_audit

        audit_cfg = OmegaConf.load(cfg.audit_config)
        audit_out = run_audit(audit_cfg)
        results["audit_r2"] = audit_out["linear_probe"].get("r_squared")
        results["audit_p_demo_B"] = audit_out["linear_probe"]["p_values"].get("demo_B_indicator")
        log.info(
            "Audit complete",
            r2=f"{results['audit_r2']:.4f}",
            p_demo_B=f"{results['audit_p_demo_B']:.4f}",
        )

    # ------------------------------------------------------------------ #
    # 6. DecodingTrust eval
    # ------------------------------------------------------------------ #
    if cfg.run_decoding_trust:
        log.info("=== Stage: DecodingTrust eval")
        from llm.eval.decoding_trust import run_decoding_trust

        dt_cfg = OmegaConf.load(cfg.dt_config)
        dt_out = run_decoding_trust(dt_cfg)
        results["decoding_trust"] = {
            k: v.get("agreement_rate", v.get("demographic_parity_gap"))
            for k, v in dt_out.get("subsets", {}).items()
            if isinstance(v, dict)
        }
        log.info("DecodingTrust eval complete", subsets=list(results["decoding_trust"].keys()))

    # ------------------------------------------------------------------ #
    # 7. General-capability benchmark
    # ------------------------------------------------------------------ #
    if cfg.run_general_capability:
        log.info("=== Stage: general-capability benchmark")
        from llm.eval.general_capability import run_general_capability

        gc_cfg = OmegaConf.load(cfg.gc_config)
        gc_out = run_general_capability(gc_cfg)
        results["general_capability_score"] = gc_out.get("overall_score")
        log.info(
            "General capability complete",
            score=f"{results['general_capability_score']:.2f}",
        )

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    summary_path = pathlib.Path(cfg.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2))
    log.info("Pipeline complete", summary=str(summary_path))
    log.info(
        "Results",
        **{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in results.items()},
    )


if __name__ == "__main__":
    main()
