"""Entry point for the general-capability benchmark.

Usage:
    uv run python -m llm.scripts.run_general_capability --config llm/configs/eval/general_capability/smoke.yaml
"""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from llm.eval.general_capability import run_general_capability
from llm.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the general-capability benchmark.")
    parser.add_argument("--config", required=True, help="Path to YAML eval config.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    log.info("Starting general-capability benchmark", config=args.config)
    results = run_general_capability(cfg)
    log.info(
        "Benchmark complete",
        overall_score=f"{results['overall_score']:.2f}",
        n_questions=results["n_questions"],
        judge=results["judge_model"],
    )


if __name__ == "__main__":
    main()
