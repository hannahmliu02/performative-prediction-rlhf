#!/usr/bin/env bash
set -euo pipefail

UV="$HOME/.local/bin/uv"

echo "=== Starting experiments at $(date) ==="

echo ">>> Baseline (no mitigation)"
$UV run python -m llm.scripts.run_feedback_loop \
    --config llm/configs/simulate/feedback_loop/m1_experiment_qt.yaml

echo ">>> Length norm"
$UV run python -m llm.scripts.run_feedback_loop \
    --config llm/configs/simulate/feedback_loop/m1_experiment_qt_length_norm.yaml

echo ">>> RM regularisation"
$UV run python -m llm.scripts.run_feedback_loop \
    --config llm/configs/simulate/feedback_loop/m1_experiment_qt_rm_reg.yaml

echo ">>> IPW (our method)"
$UV run python -m llm.scripts.run_feedback_loop \
    --config llm/configs/simulate/feedback_loop/m1_experiment_qt_ipw.yaml

echo ">>> Plotting"
$UV run python -m llm.simulate.plot_rounds \
    --mode three \
    --inputs \
        llm/outputs/simulate/m1_experiment_qt/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_qt_length_norm/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_qt_rm_reg/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_qt_ipw/per_round_metrics.csv \
    --output llm/outputs/figures/headline_three_metric.png

$UV run python -m llm.simulate.plot_rounds \
    --mode full \
    --inputs \
        llm/outputs/simulate/m1_experiment_qt/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_qt_length_norm/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_qt_rm_reg/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_qt_ipw/per_round_metrics.csv \
    --output llm/outputs/figures/headline_full_panel.png

echo "=== All done at $(date) ==="
echo "Figures: llm/outputs/figures/"
