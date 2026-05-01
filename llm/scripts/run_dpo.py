"""Entry point for DPO training.

Usage:
    uv run python -m llm.scripts.run_dpo --config llm/configs/dpo/semi_synthetic.yaml
    uv run python -m llm.scripts.run_dpo --config llm/configs/dpo/semi_synthetic.yaml --skip_sft
"""

from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from llm.training.dpo import train_dpo
from llm.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DPO policy training.")
    parser.add_argument("--config", required=True, help="Path to DPO YAML config.")
    parser.add_argument(
        "--skip_sft",
        action="store_true",
        default=False,
        help="Load base backbone instead of SFT checkpoint (experimental ablation).",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    log.info(
        "Loaded config",
        config_path=args.config,
        experiment=cfg.experiment_name,
        skip_sft=args.skip_sft,
    )

    checkpoint_dir = train_dpo(cfg, skip_sft=args.skip_sft)
    log.info("DPO training complete", checkpoint_dir=str(checkpoint_dir))


if __name__ == "__main__":
    main()
