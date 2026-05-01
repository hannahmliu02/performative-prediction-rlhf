#!/usr/bin/env bash
# Overnight pipeline: data generation → 4 feedback-loop experiments → plots.
# Prevents Mac sleep, logs everything to logs/overnight_TIMESTAMP.log.
#
# Usage (make sure `ollama serve` is already running in another terminal):
#   chmod +x run_overnight.sh
#   ./run_overnight.sh

set -euo pipefail

LOGDIR="logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/overnight_$(date +%Y%m%d_%H%M%S).log"

UV="$HOME/.local/bin/uv"
PYTHON="$UV run python"

echo "=== Overnight run started at $(date) ===" | tee -a "$LOGFILE"
echo "Log: $LOGFILE"
echo ""

run() {
    echo "" | tee -a "$LOGFILE"
    echo ">>> $(date +%H:%M:%S)  $*" | tee -a "$LOGFILE"
    "$@" 2>&1 | tee -a "$LOGFILE"
}

# caffeinate -i keeps the Mac awake for the duration of this script
caffeinate -i bash -c '

set -euo pipefail
UV="$HOME/.local/bin/uv"
PYTHON="$UV run python"
LOGFILE="'"$LOGFILE"'"

run() {
    echo "" | tee -a "$LOGFILE"
    echo ">>> $(date +%H:%M:%S)  $*" | tee -a "$LOGFILE"
    "$@" 2>&1 | tee -a "$LOGFILE"
}

# ── 1. Data generation (skip if already generated) ───────────────────────────
if [ ! -f llm/outputs/data/llm_annotated_quality_tiers/train.parquet ]; then
    run $PYTHON -m llm.data.generate_resume_prefs \
        --config llm/configs/data/llm_annotated_quality_tiers.yaml
else
    echo "Data already exists, skipping generation." | tee -a "$LOGFILE"
fi

# ── 2. Baseline (no mitigation) ───────────────────────────────────────────────
run $PYTHON -m llm.scripts.run_feedback_loop \
    --config llm/configs/simulate/feedback_loop/m1_experiment_qt.yaml

# ── 3. Length-norm baseline ───────────────────────────────────────────────────
run $PYTHON -m llm.scripts.run_feedback_loop \
    --config llm/configs/simulate/feedback_loop/m1_experiment_qt_length_norm.yaml

# ── 4. RM regularisation baseline ────────────────────────────────────────────
run $PYTHON -m llm.scripts.run_feedback_loop \
    --config llm/configs/simulate/feedback_loop/m1_experiment_qt_rm_reg.yaml

# ── 5. IPW (our method) ───────────────────────────────────────────────────────
run $PYTHON -m llm.scripts.run_feedback_loop \
    --config llm/configs/simulate/feedback_loop/m1_experiment_qt_ipw.yaml

# ── 6. Plots ──────────────────────────────────────────────────────────────────
run $PYTHON -m llm.simulate.plot_rounds \
    --mode three \
    --inputs \
        llm/outputs/simulate/m1_experiment_qt/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_qt_length_norm/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_qt_rm_reg/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_qt_ipw/per_round_metrics.csv \
    --output llm/outputs/figures/headline_three_metric.png

run $PYTHON -m llm.simulate.plot_rounds \
    --mode full \
    --inputs \
        llm/outputs/simulate/m1_experiment_qt/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_qt_length_norm/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_qt_rm_reg/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_qt_ipw/per_round_metrics.csv \
    --output llm/outputs/figures/headline_full_panel.png

echo "" | tee -a "$LOGFILE"
echo "=== All done at $(date) ===" | tee -a "$LOGFILE"
echo "Figures saved to llm/outputs/figures/" | tee -a "$LOGFILE"

' 2>&1 | tee -a "$LOGFILE"
