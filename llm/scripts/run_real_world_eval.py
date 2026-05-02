"""Section 5.2: real-world evaluation on HH-RLHF.

Runs baseline and IPW feedback loops on the real-world dataset, then
plots per-group RM accuracy across rounds alongside the synthetic results.

Usage:
    # Step 1 — generate data (once)
    uv run python -m llm.data.load_real_world \
        --config llm/configs/data/real_world_hh_rlhf.yaml

    # Step 2 — run this script
    uv run python -m llm.scripts.run_real_world_eval
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

from llm.utils.logging import get_logger

log = get_logger(__name__)

_UV = pathlib.Path.home() / ".local" / "bin" / "uv"
_PYTHON = [str(_UV), "run", "python"]

_CONFIGS = {
    "baseline": "llm/configs/simulate/feedback_loop/m1_experiment_rw.yaml",
    "ipw": "llm/configs/simulate/feedback_loop/m1_experiment_rw_ipw.yaml",
}

_METRICS = {
    "baseline": "llm/outputs/simulate/m1_experiment_rw/per_round_metrics.csv",
    "ipw": "llm/outputs/simulate/m1_experiment_rw_ipw/per_round_metrics.csv",
}


def _run(cmd: list[str]) -> None:
    log.info("Running", cmd=" ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log.error("Command failed", returncode=result.returncode)
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data loading if train.parquet already exists")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-ipw", action="store_true")
    args = parser.parse_args()

    data_path = pathlib.Path("llm/outputs/data/real_world_hh_rlhf/train.parquet")

    if not args.skip_data and not data_path.exists():
        log.info("Generating real-world dataset")
        _run(_PYTHON + [
            "-m", "llm.data.load_real_world",
            "--config", "llm/configs/data/real_world_hh_rlhf.yaml",
        ])
    else:
        log.info("Data exists, skipping generation")

    if not args.skip_baseline:
        log.info("Running baseline (no mitigation)")
        _run(_PYTHON + ["-m", "llm.scripts.run_feedback_loop",
                        "--config", _CONFIGS["baseline"]])

    if not args.skip_ipw:
        log.info("Running IPW")
        _run(_PYTHON + ["-m", "llm.scripts.run_feedback_loop",
                        "--config", _CONFIGS["ipw"]])

    # Plot
    metrics_files = [v for v in _METRICS.values() if pathlib.Path(v).exists()]
    if len(metrics_files) >= 2:
        log.info("Plotting real-world results")
        _run(_PYTHON + [
            "-m", "llm.simulate.plot_rounds",
            "--mode", "three",
            "--inputs", *metrics_files,
            "--output", "llm/outputs/figures/real_world_panel.png",
        ])
    else:
        log.warning("Not enough metrics files to plot yet", found=metrics_files)

    log.info("Real-world evaluation complete")


if __name__ == "__main__":
    main()
