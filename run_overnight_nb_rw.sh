#!/usr/bin/env bash
# Overnight pipeline — name-bias (nb) and real-world (rw) tracks only.
#
# Usage:
#   chmod +x run_overnight_nb_rw.sh
#   ./run_overnight_nb_rw.sh
#
# Prerequisites:
#   - ollama serve running in a separate terminal (nb data generation)
#   - Internet access for HH-RLHF download (rw track)
#   - uv installed and on PATH

set -euo pipefail

UV="$(command -v uv 2>/dev/null || echo "$HOME/.local/bin/uv")"
LOGDIR="logs"
mkdir -p "$LOGDIR" llm/outputs/figures
LOGFILE="$LOGDIR/overnight_nb_rw_$(date +%Y%m%d_%H%M%S).log"
echo "Log: $LOGFILE"

caffeinate -i bash -s << CAFFEINATED
set -euo pipefail

PYTHON="$UV run python"
LOGFILE="$LOGFILE"

log()  { echo ""; echo ">>> \$(date +%H:%M:%S)  \$*" | tee -a "\$LOGFILE"; }
err()  { echo "!!! \$(date +%H:%M:%S)  ERROR: \$*" | tee -a "\$LOGFILE" >&2; }

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

log "=== nb + rw overnight run started at \$(date) ==="

# ════════════════════════════════════════════════════════════════════════════ #
# TRACK 1 — Name-bias cross-group (nb)
# ════════════════════════════════════════════════════════════════════════════ #

log "── Track 1: Name-bias cross-group (nb) ──"
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
# TRACK 2 — Real-world HH-RLHF (rw)  §5.2 single-round design
# ════════════════════════════════════════════════════════════════════════════ #

log "── Track 2: Real-world HH-RLHF (rw, §5.2 single-round) ──"

skip_if_exists llm/outputs/data/real_world_hh_rlhf/train.parquet \
    \$PYTHON -m llm.data.load_real_world \
        --config llm/configs/data/real_world_hh_rlhf.yaml

skip_if_exists llm/outputs/real_world/real_world_results.json \
    \$PYTHON -m llm.scripts.run_real_world \
        --config llm/configs/data/real_world_hh_rlhf.yaml

log "Real-world summary written to llm/outputs/real_world/real_world_summary.md"

# ════════════════════════════════════════════════════════════════════════════ #
# Done
# ════════════════════════════════════════════════════════════════════════════ #

log "=== All done at \$(date) ==="
log "Figures: llm/outputs/figures/"
log "Log:     \$LOGFILE"

CAFFEINATED
