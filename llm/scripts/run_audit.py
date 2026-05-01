"""Entry point for the observability audit.

Usage:
    uv run python -m llm.scripts.run_audit --config llm/configs/audit/smoke.yaml
"""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from llm.audit.observability_audit import run_audit
from llm.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RM observability audit.")
    parser.add_argument("--config", required=True, help="Path to YAML audit config.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    log.info("Starting observability audit", config=args.config)
    run_audit(cfg)
    log.info("Audit complete")


if __name__ == "__main__":
    main()
