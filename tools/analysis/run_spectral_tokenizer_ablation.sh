#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
BENCHMARK_DEVICE="${BENCHMARK_DEVICE:-cuda:0}"
BENCHMARK_TIMES="${BENCHMARK_TIMES:-100}"
BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-10}"
ENDPOINT_TOLERANCE_MM="${ENDPOINT_TOLERANCE_MM:-1.0}"
WORK_ROOT="${WORK_ROOT:-work_dirs/spectral_tokenizer_ablation}"
LOG_ROOT="${LOG_ROOT:-paper_assets/logs/spectral_tokenizer_ablation}"
MODE="${1:-all}"

RUN_IDS=(T0 T1 T2 T3 T4)
declare -A CONFIGS=(
  [T0]="configs/wifi/tokenizer_ablation/t0_linear.py"
  [T1]="configs/wifi/tokenizer_ablation/t1_linear_ln.py"
  [T2]="configs/wifi/tokenizer_ablation/t2_temporal_residual.py"
  [T3]="configs/wifi/tokenizer_ablation/t3_spectral_gate_residual.py"
  [T4]="configs/wifi/tokenizer_ablation/t4_spectral.py"
)

checkpoint_for() {
  local run_id="$1"
  printf '%s/latest.pth\n' "$WORK_ROOT/$run_id"
}

train_one() {
  local run_id="$1"
  mkdir -p "$WORK_ROOT/$run_id" "$LOG_ROOT"
  env CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED="$SEED" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    "$PYTHON_BIN" tools/train.py "${CONFIGS[$run_id]}" \
      --gpu-id 0 --seed "$SEED" --deterministic \
      --work-dir "$WORK_ROOT/$run_id"
}

evaluate_one() {
  local run_id="$1"
  local checkpoint
  checkpoint="$(checkpoint_for "$run_id")"
  test -f "$checkpoint"
  env CUDA_VISIBLE_DEVICES="$GPU" \
    "$PYTHON_BIN" tools/test.py "${CONFIGS[$run_id]}" "$checkpoint" \
      --eval mpjpe --gpu-id 0 \
      --work-dir "$WORK_ROOT/$run_id/eval" \
      --metrics-out "$LOG_ROOT/${run_id}_eval.json"
}

benchmark_one() {
  local run_id="$1"
  "$PYTHON_BIN" tools/analysis/benchmark.py "${CONFIGS[$run_id]}" \
    --checkpoint "$(checkpoint_for "$run_id")" \
    --device "$BENCHMARK_DEVICE" \
    --times "$BENCHMARK_TIMES" --warmup "$BENCHMARK_WARMUP" \
    --out "$LOG_ROOT/${run_id}_benchmark.json"
}

audit_results() {
  "$PYTHON_BIN" tools/analysis/audit_tokenizer_ablation.py \
    --log-root "$LOG_ROOT" \
    --csv-out "$LOG_ROOT/tokenizer_component_ablation.csv" \
    --json-out "$LOG_ROOT/tokenizer_component_ablation_manifest.json" \
    --endpoint-tolerance-mm "$ENDPOINT_TOLERANCE_MM"
  "$PYTHON_BIN" tools/analysis/render_tokenizer_ablation.py \
    "$LOG_ROOT/tokenizer_component_ablation_manifest.json" \
    paper_assets/manuscript_latex/resfes2026_witidar/tables/tokenizer_component_ablation.tex
}

case "$MODE" in
  train)
    for run_id in "${RUN_IDS[@]}"; do train_one "$run_id"; done
    ;;
  evaluate)
    for run_id in "${RUN_IDS[@]}"; do evaluate_one "$run_id"; done
    ;;
  benchmark)
    for run_id in "${RUN_IDS[@]}"; do benchmark_one "$run_id"; done
    ;;
  audit)
    audit_results
    ;;
  all)
    for run_id in "${RUN_IDS[@]}"; do
      train_one "$run_id"
      evaluate_one "$run_id"
      benchmark_one "$run_id"
    done
    audit_results
    ;;
  *)
    echo "Usage: $0 [all|train|evaluate|benchmark|audit]" >&2
    exit 2
    ;;
esac
