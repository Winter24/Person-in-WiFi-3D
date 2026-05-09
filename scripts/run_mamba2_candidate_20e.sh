#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

MODE="${1:-all}"
if (($# > 0)); then
  shift
fi

export ONLY_RUN_IDS="${ONLY_RUN_IDS:-M2F M2C M2FGP}"
export WORK_ROOT="${WORK_ROOT:-work_dirs/mamba2_encoder_ablation_20e}"
export EVAL_ROOT="${EVAL_ROOT:-work_dirs/mamba2_encoder_eval_20e}"
export LOG_ROOT="${LOG_ROOT:-paper_assets/logs/mamba2_encoder_ablation_20e}"
export CSV_PATH="${CSV_PATH:-$LOG_ROOT/experiment_log.csv}"
export SECTION_CSV_PATH="${SECTION_CSV_PATH:-$LOG_ROOT/section_latency_log.csv}"
export LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-$LOG_ROOT/launch_logs}"

exec bash scripts/run_mamba2_encoder_ablation.sh "$MODE" "$@"
