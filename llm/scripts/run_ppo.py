"""Entry point for PPO (RLOO) training.

Usage:
    uv run python -m llm.scripts.run_ppo --config llm/configs/ppo/semi_synthetic.yaml
"""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from llm.training.ppo import train_ppo
from llm.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PPO (RLOO) policy training.")
    parser.add_argument("--config", required=True, help="Path to PPO YAML config.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    log.info("Loaded config", config_path=args.config, experiment=cfg.experiment_name)

    checkpoint_dir = train_ppo(cfg)
    log.info("PPO training complete", checkpoint_dir=str(checkpoint_dir))


if __name__ == "__main__":
    main()
