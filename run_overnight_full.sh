#!/usr/bin/env bash
# Full overnight pipeline — runs all three experiment tracks and plots.
#
# Tracks:
#   1. Quality-tier synthetic (qt)   — data via Ollama, 4 methods
#   2. Name-bias cross-group (nb)    — data via Ollama, 4 methods
#   3. Real-world HH-RLHF (rw)      — downloads + classifies, 2 methods
#
# Each track skips steps whose outputs already exist, so you can safely
# re-run after a partial failure without redoing completed work.
#
# Usage:
#   chmod +x run_overnight_full.sh
#   ./run_overnight_full.sh
#
# Prerequisites:
#   - ollama serve running in a separate terminal (Tracks 1 & 2)
#   - Internet access for HH-RLHF download (Track 3)
#   - uv installed at ~/.local/bin/uv

set -euo pipefail

UV="$HOME/.local/bin/uv"
LOGDIR="logs"
mkdir -p "$LOGDIR" llm/outputs/figures
LOGFILE="$LOGDIR/overnight_full_$(date +%Y%m%d_%H%M%S).log"
echo "Log: $LOGFILE"

# ── caffeinate keeps the Mac awake for the duration ──────────────────────────
caffeinate -i bash -s << CAFFEINATED
set -euo pipefail

PYTHON="$UV run python"
LOGFILE="$LOGFILE"

# ── helpers ──────────────────────────────────────────────────────────────────

log() { echo "" | tee -a "\$LOGFILE"; echo ">>> \$(date +%H:%M:%S)  \$*" | tee -a "\$LOGFILE"; }
err() { echo "!!! \$(date +%H:%M:%S)  ERROR: \$*" | tee -a "\$LOGFILE" >&2; }

run() {
    log "\$*"
    if ! "\$@" 2>&1 | tee -a "\$LOGFILE"; then
        err "Command failed: \$*"
        exit 1
    fi
}

skip_if_exists() {
    local file="\$1"; shift
    if [ -f "\$file" ]; then
        log "SKIP (already exists): \$file"
        return 0
    fi
    run "\$@"
}

feedback_loop() {
    local config="\$1"
    local name
    name=\$(grep '^experiment_name:' "\$config" | awk '{print \$2}')
    local metrics="llm/outputs/simulate/\${name}/per_round_metrics.csv"
    skip_if_exists "\$metrics" \
        \$PYTHON -m llm.scripts.run_feedback_loop --config "\$config"
}

ollama_check() {
    if ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        err "Ollama is not running — start it with: ollama serve"
        exit 1
    fi
}

log "=== Overnight full run started at \$(date) ==="

# ════════════════════════════════════════════════════════════════════════════ #
# TRACK 1 — Quality-tier synthetic (qt)
# ════════════════════════════════════════════════════════════════════════════ #

log "── Track 1: Quality-tier synthetic (qt) ──"
ollama_check

skip_if_exists llm/outputs/data/llm_annotated_quality_tiers/train.parquet \
    \$PYTHON -m llm.data.generate_resume_prefs \
        --config llm/configs/data/llm_annotated_quality_tiers.yaml

feedback_loop llm/configs/simulate/feedback_loop/m1_experiment_qt.yaml
feedback_loop llm/configs/simulate/feedback_loop/m1_experiment_qt_length_norm.yaml
feedback_loop llm/configs/simulate/feedback_loop/m1_experiment_qt_rm_reg.yaml
feedback_loop llm/configs/simulate/feedback_loop/m1_experiment_qt_ipw.yaml

log "Plotting qt track"
run \$PYTHON -m llm.simulate.plot_rounds \
    --mode three \
    --inputs \
        llm/outputs/simulate/m1_experiment_qt/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_qt_length_norm/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_qt_rm_reg/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_qt_ipw/per_round_metrics.csv \
    --output llm/outputs/figures/qt_three_metric.png

run \$PYTHON -m llm.simulate.plot_rounds \
    --mode full \
    --inputs \
        llm/outputs/simulate/m1_experiment_qt/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_qt_length_norm/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_qt_rm_reg/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_qt_ipw/per_round_metrics.csv \
    --output llm/outputs/figures/qt_full_panel.png

# ════════════════════════════════════════════════════════════════════════════ #
# TRACK 2 — Name-bias cross-group (nb)
# ════════════════════════════════════════════════════════════════════════════ #

log "── Track 2: Name-bias cross-group (nb) ──"
ollama_check

skip_if_exists llm/outputs/data/name_bias_cross_group/train.parquet \
    \$PYTHON -m llm.data.generate_resume_prefs \
        --config llm/configs/data/name_bias_cross_group.yaml

feedback_loop llm/configs/simulate/feedback_loop/m1_experiment_nb.yaml
feedback_loop llm/configs/simulate/feedback_loop/m1_experiment_nb_length_norm.yaml
feedback_loop llm/configs/simulate/feedback_loop/m1_experiment_nb_rm_reg.yaml
feedback_loop llm/configs/simulate/feedback_loop/m1_experiment_nb_ipw.yaml

log "Plotting nb track"
run \$PYTHON -m llm.simulate.plot_rounds \
    --mode three \
    --inputs \
        llm/outputs/simulate/m1_experiment_nb/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_nb_length_norm/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_nb_rm_reg/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_nb_ipw/per_round_metrics.csv \
    --output llm/outputs/figures/nb_three_metric.png

run \$PYTHON -m llm.simulate.plot_rounds \
    --mode full \
    --inputs \
        llm/outputs/simulate/m1_experiment_nb/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_nb_length_norm/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_nb_rm_reg/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_nb_ipw/per_round_metrics.csv \
    --output llm/outputs/figures/nb_full_panel.png

# ════════════════════════════════════════════════════════════════════════════ #
# TRACK 3 — Real-world HH-RLHF (rw)
# ════════════════════════════════════════════════════════════════════════════ #

log "── Track 3: Real-world HH-RLHF (rw) ──"

skip_if_exists llm/outputs/data/real_world_hh_rlhf/train.parquet \
    \$PYTHON -m llm.data.load_real_world \
        --config llm/configs/data/real_world_hh_rlhf.yaml

feedback_loop llm/configs/simulate/feedback_loop/m1_experiment_rw.yaml
feedback_loop llm/configs/simulate/feedback_loop/m1_experiment_rw_ipw.yaml

log "Plotting rw track"
run \$PYTHON -m llm.simulate.plot_rounds \
    --mode three \
    --inputs \
        llm/outputs/simulate/m1_experiment_rw/per_round_metrics.csv \
        llm/outputs/simulate/m1_experiment_rw_ipw/per_round_metrics.csv \
    --output llm/outputs/figures/rw_three_metric.png

# ════════════════════════════════════════════════════════════════════════════ #
# Done
# ════════════════════════════════════════════════════════════════════════════ #

log "=== All done at \$(date) ==="
log "Figures: llm/outputs/figures/"
log "Log:     \$LOGFILE"

CAFFEINATED
