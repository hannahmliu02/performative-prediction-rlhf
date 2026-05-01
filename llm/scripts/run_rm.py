"""Entry point for reward model training.

Usage:
    uv run python -m llm.scripts.run_rm --config llm/configs/rm/semi_synthetic.yaml
"""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from llm.training.train_rm import train_rm
from llm.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a reward model.")
    parser.add_argument("--config", required=True, help="Path to RM YAML config.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    log.info("Loaded config", config_path=args.config, experiment=cfg.experiment_name)

    checkpoint_dir = train_rm(cfg)
    log.info("RM training complete", checkpoint_dir=str(checkpoint_dir))


if __name__ == "__main__":
    main()
