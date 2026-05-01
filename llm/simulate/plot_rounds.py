"""Per-round metrics visualiser for the feedback-loop experiment.

Reads one or more per_round_metrics.csv files (one per seed/run), groups by
(method, round), and renders a 2×3 panel figure suitable for the paper.

Panel layout:
  [0,0] Per-cell RM accuracy   [0,1] Policy output entropy  [0,2] Amplification ratio
  [1,0] DT fairness score      [1,1] DT stereotype score     [1,2] MT-Bench score

One line per method, shaded band = ±1 std across seeds.
For [0,0], Group A is solid and Group B is dashed with the same method colour.

Usage:
    python -m llm.simulate.plot_rounds \\
        --inputs llm/outputs/simulate/run1/per_round_metrics.csv \\
                 llm/outputs/simulate/run2/per_round_metrics.csv \\
        --output llm/outputs/figures/headline_figure.png
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from llm.utils.logging import get_logger

log = get_logger(__name__)

# Colorblind-safe palette (Okabe-Ito)
_METHOD_COLORS: list[str] = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink/purple
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
]

_PANEL_META: list[dict[str, Any]] = [
    # (row, col, y_col_or_special, ylabel, title)
    {"pos": (0, 0), "key": "_accuracy", "ylabel": "Accuracy", "title": "Per-cell RM accuracy"},
    {"pos": (0, 1), "key": "policy_output_entropy", "ylabel": "Entropy (nats)", "title": "Policy output entropy"},
    {"pos": (0, 2), "key": "amplification_ratio", "ylabel": "Ratio", "title": "Amplification ratio"},
    {"pos": (1, 0), "key": "dt_fairness_score", "ylabel": "Parity gap", "title": "DT fairness score"},
    {"pos": (1, 1), "key": "dt_stereotype_score", "ylabel": "Agreement rate", "title": "DT stereotype score"},
    {"pos": (1, 2), "key": "mt_bench_score", "ylabel": "Score (1–10)", "title": "MT-Bench score"},
]


def load_runs(csv_paths: list[str]) -> pd.DataFrame:
    """Load and concatenate per-round CSVs from multiple runs."""
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    # Ensure method is always a string
    combined["method"] = combined["method"].fillna("no_mitigation").astype(str)
    return combined


def _plot_accuracy_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    methods: list[str],
    color_map: dict[str, str],
) -> None:
    """Special handler for the accuracy panel: two lines per method (A solid, B dashed)."""
    for method in methods:
        color = color_map[method]
        sub = df[df["method"] == method]
        for grp, ls, marker in [("demo_A_accuracy", "-", "o"), ("demo_B_accuracy", "--", "s")]:
            label_group = "A" if "A" in grp else "B"
            grouped = sub.groupby("round")[grp]
            means = grouped.mean()
            stds = grouped.std().fillna(0)
            rounds = means.index.tolist()
            ax.plot(rounds, means.values, color=color, linestyle=ls, marker=marker,
                    label=f"{method} / {label_group}", linewidth=1.5, markersize=5)
            ax.fill_between(rounds,
                            means.values - stds.values,
                            means.values + stds.values,
                            alpha=0.15, color=color)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-cell RM accuracy")
    ax.legend(fontsize=7, ncol=2)


def _plot_scalar_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    col: str,
    ylabel: str,
    title: str,
    methods: list[str],
    color_map: dict[str, str],
) -> None:
    """Generic handler for panels with a single scalar metric per (method, round)."""
    all_nan = df[col].isna().all() if col in df.columns else True
    for method in methods:
        color = color_map[method]
        sub = df[df["method"] == method]
        if col not in sub.columns or sub[col].isna().all():
            continue
        grouped = sub.groupby("round")[col]
        means = grouped.mean()
        stds = grouped.std().fillna(0)
        rounds = means.index.tolist()
        ax.plot(rounds, means.values, color=color, marker="o", label=method,
                linewidth=1.5, markersize=5)
        ax.fill_between(rounds,
                        means.values - stds.values,
                        means.values + stds.values,
                        alpha=0.15, color=color)
    if col == "amplification_ratio":
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, label="baseline (=1)")
    if all_nan:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes,
                color="gray", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7)


def plot_panel(df: pd.DataFrame, output_path: str | pathlib.Path) -> None:
    """Render the 2×3 headline figure and save to output_path."""
    methods = sorted(df["method"].unique().tolist())
    color_map = {m: _METHOD_COLORS[i % len(_METHOD_COLORS)] for i, m in enumerate(methods)}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Feedback-loop metrics across training rounds", fontsize=12)

    all_rounds = sorted(df["round"].unique().tolist())

    for meta in _PANEL_META:
        r, c = meta["pos"]
        ax = axes[r][c]
        ax.set_xlabel("Round")
        ax.set_xticks(all_rounds)

        if meta["key"] == "_accuracy":
            _plot_accuracy_panel(ax, df, methods, color_map)
        else:
            _plot_scalar_panel(
                ax, df, meta["key"], meta["ylabel"], meta["title"], methods, color_map
            )

    plt.tight_layout()
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    log.info("Saved headline figure", path=str(output_path))


def plot_three_metrics(df: pd.DataFrame, output_path: str | pathlib.Path) -> None:
    """Render a clean 1×3 figure: accuracy divergence, disparity, amplification.

    Panel 1 — RM accuracy: Group A (solid) and Group B (dashed) per method.
    Panel 2 — Accuracy disparity |A − B| over rounds (should grow without mitigation).
    Panel 3 — Amplification ratio (disparity_k / disparity_0; baseline = 1).
    """
    methods = sorted(df["method"].unique().tolist())
    color_map = {m: _METHOD_COLORS[i % len(_METHOD_COLORS)] for i, m in enumerate(methods)}
    all_rounds = sorted(df["round"].unique().tolist())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Feedback-loop bias dynamics", fontsize=12, y=1.01)

    # --- Panel 1: accuracy divergence ---
    ax = axes[0]
    for method in methods:
        color = color_map[method]
        sub = df[df["method"] == method]
        for col, ls, marker, label_sfx in [
            ("demo_A_accuracy", "-", "o", "/ A"),
            ("demo_B_accuracy", "--", "s", "/ B"),
        ]:
            grouped = sub.groupby("round")[col]
            means = grouped.mean()
            stds = grouped.std().fillna(0)
            rounds = means.index.tolist()
            ax.plot(rounds, means.values, color=color, linestyle=ls, marker=marker,
                    label=f"{method} {label_sfx}", linewidth=1.8, markersize=5)
            ax.fill_between(rounds,
                            means.values - stds.values,
                            means.values + stds.values,
                            alpha=0.12, color=color)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="chance")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Round")
    ax.set_ylabel("RM accuracy")
    ax.set_title("Accuracy by group")
    ax.set_xticks(all_rounds)
    ax.legend(fontsize=7, ncol=len(methods))

    # --- Panel 2: accuracy disparity |A − B| ---
    ax = axes[1]
    for method in methods:
        color = color_map[method]
        sub = df[df["method"] == method].copy()
        sub["disparity"] = (sub["demo_A_accuracy"] - sub["demo_B_accuracy"]).abs()
        grouped = sub.groupby("round")["disparity"]
        means = grouped.mean()
        stds = grouped.std().fillna(0)
        rounds = means.index.tolist()
        ax.plot(rounds, means.values, color=color, marker="o", label=method,
                linewidth=1.8, markersize=5)
        ax.fill_between(rounds,
                        means.values - stds.values,
                        means.values + stds.values,
                        alpha=0.12, color=color)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Round")
    ax.set_ylabel("|A accuracy − B accuracy|")
    ax.set_title("Accuracy disparity")
    ax.set_xticks(all_rounds)
    ax.legend(fontsize=7)

    # --- Panel 3: amplification ratio ---
    ax = axes[2]
    col = "amplification_ratio"
    all_nan = df[col].isna().all() if col in df.columns else True
    for method in methods:
        color = color_map[method]
        sub = df[df["method"] == method]
        if col not in sub.columns or sub[col].isna().all():
            continue
        grouped = sub.groupby("round")[col]
        means = grouped.mean()
        stds = grouped.std().fillna(0)
        rounds = means.index.tolist()
        ax.plot(rounds, means.values, color=color, marker="o", label=method,
                linewidth=1.8, markersize=5)
        ax.fill_between(rounds,
                        means.values - stds.values,
                        means.values + stds.values,
                        alpha=0.12, color=color)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, label="no amplification")
    if all_nan:
        ax.text(0.5, 0.5, "No data\n(computed post-hoc)", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=9)
    ax.set_xlabel("Round")
    ax.set_ylabel("Amplification ratio")
    ax.set_title("Bias amplification")
    ax.set_xticks(all_rounds)
    ax.legend(fontsize=7)

    plt.tight_layout()
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved three-metric figure", path=str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-round feedback-loop metrics.")
    parser.add_argument("--inputs", nargs="+", required=True, help="per_round_metrics.csv paths.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument(
        "--mode",
        choices=["three", "full"],
        default="three",
        help="'three' = 1×3 headline panel (default); 'full' = 2×3 panel.",
    )
    args = parser.parse_args()

    df = load_runs(args.inputs)
    log.info("Loaded runs", n_rows=len(df), methods=sorted(df["method"].unique().tolist()))
    if args.mode == "three":
        plot_three_metrics(df, args.output)
    else:
        plot_panel(df, args.output)


if __name__ == "__main__":
    main()
