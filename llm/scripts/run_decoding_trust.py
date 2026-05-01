"""Entry point for the DecodingTrust evaluation.

Usage:
    uv run python -m llm.scripts.run_decoding_trust --config llm/configs/eval/decoding_trust/smoke.yaml
"""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from llm.eval.decoding_trust import run_decoding_trust
from llm.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DecodingTrust evaluation subsets.")
    parser.add_argument("--config", required=True, help="Path to YAML eval config.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    log.info("Starting DecodingTrust evaluation", config=args.config)
    results = run_decoding_trust(cfg)
    for subset, scores in results["subsets"].items():
        log.info("Result", subset=subset, **{k: v for k, v in scores.items() if isinstance(v, (int, float, str))})
    log.info("DecodingTrust evaluation complete")


if __name__ == "__main__":
    main()
