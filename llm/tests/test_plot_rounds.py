"""Tests for llm/simulate/plot_rounds.py."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from llm.simulate.plot_rounds import load_runs, plot_panel


def _make_synthetic_csv(tmp_path: pathlib.Path, method: str, seed: int) -> pathlib.Path:
    """Create a minimal per_round_metrics.csv for testing."""
    rng = np.random.default_rng(seed)
    n_rounds = 3
    rows = []
    for r in range(n_rounds):
        disparity_0 = 0.1
        disparity_k = disparity_0 * (1.0 + 0.2 * r)
        acc_A = 0.7 + rng.normal(0, 0.02)
        acc_B = acc_A - disparity_k
        rows.append(
            {
                "round": r,
                "method": method,
                "seed": seed,
                "n_train": 100 + r * 50,
                "n_new": 40,
                "demo_A_accuracy": float(np.clip(acc_A, 0, 1)),
                "demo_B_accuracy": float(np.clip(acc_B, 0, 1)),
                "per_cell_rm_accuracy_mean": float(np.clip((acc_A + acc_B) / 2, 0, 1)),
                "per_cell_rm_accuracy_min": float(np.clip(acc_B - 0.05, 0, 1)),
                "policy_output_entropy": float(3.0 + rng.normal(0, 0.1)),
                "dt_fairness_score": float("nan"),
                "dt_stereotype_score": float("nan"),
                "mt_bench_score": float("nan"),
                "amplification_ratio": disparity_k / disparity_0,
                "rm_checkpoint": "/tmp/rm",
                "policy_checkpoint": "/tmp/policy",
            }
        )
    df = pd.DataFrame(rows)
    csv_path = tmp_path / f"metrics_{method}_{seed}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_load_runs(tmp_path: pathlib.Path) -> None:
    csv1 = _make_synthetic_csv(tmp_path, "no_mitigation", seed=1)
    csv2 = _make_synthetic_csv(tmp_path, "length_norm", seed=2)
    df = load_runs([str(csv1), str(csv2)])
    assert len(df) == 6  # 3 rounds × 2 methods
    assert set(df["method"].unique()) == {"no_mitigation", "length_norm"}
    assert set(df["round"].unique()) == {0, 1, 2}


def test_plot_panel_produces_png(tmp_path: pathlib.Path) -> None:
    """plot_panel must write a PNG without raising for a minimal synthetic CSV."""
    csvs = []
    for method in ("no_mitigation", "ours_ipw"):
        for seed in (1, 2, 3):
            csvs.append(str(_make_synthetic_csv(tmp_path, method, seed)))

    df = load_runs(csvs)
    out_png = tmp_path / "headline_figure.png"
    plot_panel(df, out_png)
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_plot_panel_single_method(tmp_path: pathlib.Path) -> None:
    """Degenerate case: single method, single seed — should not crash."""
    csv = _make_synthetic_csv(tmp_path, "no_mitigation", seed=42)
    df = load_runs([str(csv)])
    out_png = tmp_path / "single_method.png"
    plot_panel(df, out_png)
    assert out_png.exists()


def test_plot_panel_with_nan_metrics(tmp_path: pathlib.Path) -> None:
    """All DT / MT-Bench columns NaN — panels should show 'No data' without raising."""
    csv = _make_synthetic_csv(tmp_path, "no_mitigation", seed=7)
    df = load_runs([str(csv)])
    # NaN columns are already set by _make_synthetic_csv; just verify no crash
    out_png = tmp_path / "nan_metrics.png"
    plot_panel(df, out_png)
    assert out_png.exists()
