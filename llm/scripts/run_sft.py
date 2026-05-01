"""Entry point for SFT training.

Usage:
    uv run python -m llm.scripts.run_sft --config llm/configs/sft/semi_synthetic.yaml
"""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from llm.training.sft import train_sft
from llm.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run supervised fine-tuning.")
    parser.add_argument("--config", required=True, help="Path to SFT YAML config.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    log.info("Loaded config", config_path=args.config, experiment=cfg.experiment_name)

    checkpoint_dir = train_sft(cfg)
    log.info("SFT complete", checkpoint_dir=str(checkpoint_dir))


if __name__ == "__main__":
    main()
