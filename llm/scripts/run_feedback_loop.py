"""Entry point for the multi-round feedback-loop simulator.

Usage:
    uv run python -m llm.scripts.run_feedback_loop --config llm/configs/simulate/feedback_loop/smoke.yaml
"""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from llm.simulate.feedback_loop import run_feedback_loop
from llm.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-round RLHF feedback-loop simulation.")
    parser.add_argument("--config", required=True, help="Path to YAML simulate config.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    log.info("Starting feedback-loop simulation", config=args.config, num_rounds=cfg.num_rounds)
    metrics = run_feedback_loop(cfg)
    for m in metrics:
        log.info(
            "Round summary",
            round=m["round"],
            demo_A_accuracy=f"{m['demo_A_accuracy']:.3f}",
            demo_B_accuracy=f"{m['demo_B_accuracy']:.3f}",
            n_new=m["n_new"],
        )
    log.info("Simulation complete", num_rounds=len(metrics))


if __name__ == "__main__":
    main()
