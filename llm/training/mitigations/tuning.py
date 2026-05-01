"""Tuning protocol for baseline mitigations.

For each baseline, `select_best_round1_config` runs one round of the feedback
loop for every candidate hyperparameter setting, evaluates round-1 disparity on
the held-out audit set, and returns the setting that minimises disparity.

This protocol deliberately favours the baselines: the best round-1 setting has
the strongest possible chance of also breaking the loop.  The winning config
is logged to docs/decisions.md and the per-config metrics are saved to
llm/outputs/tuning/<baseline_name>.csv.

Usage (called from the headline sweep launcher, S1-Task 4):
    from llm.training.mitigations.tuning import select_best_round1_config

    best_cfg = select_best_round1_config(
        mitigation_name="length_norm",
        sweep=[{"type": "length_norm", "beta": b} for b in [0.01, 0.1, 1.0]],
        base_loop_cfg=OmegaConf.load("llm/configs/simulate/feedback_loop/semi_synthetic.yaml"),
        output_dir="llm/outputs/tuning",
    )
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from llm.utils.logging import get_logger

log = get_logger(__name__)


def select_best_round1_config(
    mitigation_name: str,
    sweep: list[dict[str, Any]],
    base_loop_cfg: DictConfig,
    output_dir: str = "llm/outputs/tuning",
) -> dict[str, Any]:
    """Run one feedback-loop round per sweep config; return the config with
    minimum round-1 disparity (|acc_A − acc_B|).

    Saves per-config metrics to `output_dir/<mitigation_name>.csv`.

    Args:
        mitigation_name: Label for the mitigation (e.g. "length_norm").
        sweep: List of mitigation config dicts to sweep over.
            Each dict must contain at least {"type": "<mitigation_type>"}.
        base_loop_cfg: Base feedback-loop OmegaConf config.  num_rounds will
            be overridden to 1.
        output_dir: Directory for tuning CSV outputs.

    Returns:
        The sweep config dict that minimised round-1 disparity.
    """
    from llm.simulate.feedback_loop import run_feedback_loop

    out_path = pathlib.Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    for sweep_idx, m_cfg in enumerate(sweep):
        exp_name = f"{base_loop_cfg.experiment_name}_tune_{mitigation_name}_{sweep_idx}"
        log.info(
            "Tuning round",
            mitigation=mitigation_name,
            sweep_idx=sweep_idx,
            config=m_cfg,
            experiment=exp_name,
        )

        # Build a one-round version of the loop config with this mitigation
        cfg_dict = OmegaConf.to_container(base_loop_cfg, resolve=True)
        assert isinstance(cfg_dict, dict)
        cfg_dict["num_rounds"] = 1
        cfg_dict["experiment_name"] = exp_name
        cfg_dict["mitigation"] = copy.deepcopy(m_cfg)
        loop_cfg = OmegaConf.create(cfg_dict)

        try:
            metrics = run_feedback_loop(loop_cfg)
            r0 = metrics[0]
            disparity = abs(r0.get("demo_A_accuracy", 0.0) - r0.get("demo_B_accuracy", 0.0))
        except Exception as exc:
            log.warning("Tuning run failed", sweep_idx=sweep_idx, error=str(exc))
            disparity = float("nan")
            r0 = {}

        record: dict[str, Any] = {
            "sweep_idx": sweep_idx,
            "mitigation_name": mitigation_name,
            "disparity": disparity,
            **{f"mitigation_{k}": v for k, v in m_cfg.items()},
            **{k: r0.get(k) for k in ("demo_A_accuracy", "demo_B_accuracy", "per_cell_rm_accuracy_mean")},
        }
        records.append(record)
        log.info(
            "Tuning run complete",
            sweep_idx=sweep_idx,
            disparity=f"{disparity:.4f}",
        )

    df = pd.DataFrame(records)
    csv_path = out_path / f"{mitigation_name}.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved tuning results", path=str(csv_path))

    # Find the run with minimum disparity (ignoring NaN)
    valid = df.dropna(subset=["disparity"])
    if valid.empty:
        log.warning("All tuning runs failed; returning first sweep config")
        return sweep[0]

    best_idx = int(valid.loc[valid["disparity"].idxmin(), "sweep_idx"])
    best_cfg = sweep[best_idx]
    log.info(
        "Best config selected",
        sweep_idx=best_idx,
        disparity=f"{valid[valid['sweep_idx'] == best_idx]['disparity'].iloc[0]:.4f}",
        config=best_cfg,
    )
    return best_cfg
