#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-all}"
if (($# > 0)); then
  shift
fi

RUN_IDS=(A_DRAFT C_RF1 D_RF4 D_RF8 D_RF10)
if [[ -n "${ONLY_RUN_IDS:-}" ]]; then
  read -r -a RUN_IDS <<<"$ONLY_RUN_IDS"
fi

TRAIN_GPU="${TRAIN_GPU:-0}"
TEST_GPU="${TEST_GPU:-0}"
SEED="${SEED:-42}"
PYTHONHASHSEED_VALUE="${PYTHONHASHSEED_VALUE:-42}"
CUBLAS_WORKSPACE_CONFIG_VALUE="${CUBLAS_WORKSPACE_CONFIG_VALUE:-:4096:8}"
BENCHMARK_DEVICE="${BENCHMARK_DEVICE:-cuda:0}"
BENCHMARK_TIMES="${BENCHMARK_TIMES:-100}"
BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-10}"
PROFILE_SECTIONS="${PROFILE_SECTIONS:-1}"
AUTO_FIX_TRITON_LIBCUDA="${AUTO_FIX_TRITON_LIBCUDA:-1}"
DRY_RUN="${DRY_RUN:-0}"

WORK_ROOT="${WORK_ROOT:-work_dirs/flow_numstep_ablation}"
EVAL_ROOT="${EVAL_ROOT:-work_dirs/flow_numstep_eval}"
LOG_ROOT="${LOG_ROOT:-paper_assets/logs/flow_numstep_ablation}"
CSV_PATH="${CSV_PATH:-$LOG_ROOT/experiment_log.csv}"
SECTION_CSV_PATH="${SECTION_CSV_PATH:-$LOG_ROOT/section_latency_log.csv}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-$LOG_ROOT/launch_logs}"

declare -A CONFIG_PATHS=(
  [A_DRAFT]="configs/wifi/wi_tidir_wifi_flow_draft.py"
  [C_RF1]="configs/wifi/wi_tidir_wifi_flow_rf_1step.py"
  [D_RF4]="configs/wifi/wi_tidir_wifi_flow_rf_4step.py"
  [D_RF8]="configs/wifi/wi_tidir_wifi_flow_rf_8step.py"
  [D_RF10]="configs/wifi/wi_tidir_wifi_flow_rf_10step.py"
)

declare -A RUN_NOTES=(
  [A_DRAFT]="A draft pose only: WiTiDAR draft regressor, no refinement head"
  [C_RF1]="C rectified flow: random t interpolation, Euler num_steps=1"
  [D_RF4]="D rectified flow: random t interpolation, Euler num_steps=4"
  [D_RF8]="D rectified flow: random t interpolation, Euler num_steps=8"
  [D_RF10]="D rectified flow: random t interpolation, Euler num_steps=10"
)

usage() {
  cat <<'EOF'
Usage:
  scripts/run_flow_numstep_ablation.sh [all|train|postprocess|benchmark|extract|smoke|help]

Modes:
  all          Train each flow ablation, then evaluate, benchmark, append CSV.
  train        Train selected flow ablations.
  postprocess  Evaluate + benchmark + append CSV for existing checkpoints.
  benchmark    Benchmark only, for existing checkpoints.
  extract      Rebuild CSV rows from existing *_eval.json and *_benchmark.json.
  smoke        Short benchmark with --times 5 --warmup 2.

Environment overrides:
  ONLY_RUN_IDS="A_DRAFT C_RF1 D_RF8"  Select a subset.
  TRAIN_GPU=0 TEST_GPU=0              GPU ids.
  SEED=42                             Fixed training seed.
  WORK_ROOT=work_dirs/...             Training output root.
  LOG_ROOT=paper_assets/logs/...      JSON/CSV output root.
  DRY_RUN=1                           Print commands without running.
EOF
}

print_header() {
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

setup_triton_libcuda_env() {
  export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-/tmp/cuda-driver}"
  export LD_LIBRARY_PATH="/tmp/cuda-driver:/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="/tmp/cuda-driver:/usr/lib64-nvidia:${LIBRARY_PATH:-}"

  if [[ "$AUTO_FIX_TRITON_LIBCUDA" != "1" ]]; then
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ bash scripts/auto_fix_triton_libcuda.sh"
    return 0
  fi
  bash scripts/auto_fix_triton_libcuda.sh
}

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: Missing required file: $path" >&2
    exit 1
  fi
}

validate_runs() {
  local run_id config_path
  require_file "tools/train.py"
  require_file "tools/test.py"
  require_file "tools/analysis/benchmark.py"
  require_file "tools/analysis/append_experiment_log.py"
  for run_id in "${RUN_IDS[@]}"; do
    config_path="${CONFIG_PATHS[$run_id]:-}"
    if [[ -z "$config_path" ]]; then
      echo "ERROR: Unknown run id: $run_id" >&2
      exit 1
    fi
    require_file "$config_path"
  done
}

run_work_dir() {
  printf '%s\n' "$WORK_ROOT/$1"
}

eval_work_dir() {
  printf '%s\n' "$EVAL_ROOT/$1"
}

eval_json_path() {
  printf '%s\n' "$LOG_ROOT/${1}_eval.json"
}

benchmark_json_path() {
  printf '%s\n' "$LOG_ROOT/${1}_benchmark.json"
}

resolve_checkpoint() {
  local run_id="$1"
  local run_dir
  run_dir="$(run_work_dir "$run_id")"

  local override_var="${run_id}_CKPT"
  local override="${!override_var:-}"
  if [[ -n "$override" ]]; then
    require_file "$override"
    printf '%s\n' "$override"
    return 0
  fi
  if [[ -f "$run_dir/latest.pth" ]]; then
    printf '%s\n' "$run_dir/latest.pth"
    return 0
  fi
  local epoch_files=("$run_dir"/epoch_*.pth)
  if [[ -e "${epoch_files[0]}" ]]; then
    printf '%s\n' "${epoch_files[@]}" | tr ' ' '\n' | sort -V | tail -n 1
    return 0
  fi
  echo "ERROR: No checkpoint found for $run_id in $run_dir" >&2
  return 1
}

resolve_checkpoint_or_empty() {
  local run_id="$1"
  if resolve_checkpoint "$run_id" 2>/dev/null; then
    return 0
  fi
  printf '%s\n' ""
}

train_one() {
  local run_id="$1"
  local config_path="${CONFIG_PATHS[$run_id]}"
  local work_dir
  local log_path
  work_dir="$(run_work_dir "$run_id")"
  log_path="$LAUNCH_LOG_DIR/${run_id}_train.log"
  mkdir -p "$work_dir" "$LAUNCH_LOG_DIR"

  local cmd=(
    env
    "CUDA_VISIBLE_DEVICES=$TRAIN_GPU"
    "PYTHONHASHSEED=$PYTHONHASHSEED_VALUE"
    "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG_VALUE"
    "$PYTHON_BIN"
    tools/train.py
    "$config_path"
    --gpu-id 0
    --seed "$SEED"
    --deterministic
    --work-dir "$work_dir"
  )
  if [[ -f "$work_dir/latest.pth" ]]; then
    cmd+=(--auto-resume)
  fi

  print_header "Train $run_id"
  echo "Config:   $config_path"
  echo "Work dir: $work_dir"
  echo "Log:      $log_path"

  if [[ "$DRY_RUN" == "1" ]]; then
    run_cmd "${cmd[@]}"
    return 0
  fi
  printf '+'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}" >"$log_path" 2>&1
}

evaluate_one() {
  local run_id="$1"
  local config_path="${CONFIG_PATHS[$run_id]}"
  local checkpoint_path
  local work_dir
  local eval_json
  checkpoint_path="$(resolve_checkpoint "$run_id")"
  work_dir="$(eval_work_dir "$run_id")"
  eval_json="$(eval_json_path "$run_id")"
  mkdir -p "$work_dir" "$LOG_ROOT"

  print_header "Evaluate $run_id"
  run_cmd env "CUDA_VISIBLE_DEVICES=$TEST_GPU" "$PYTHON_BIN" tools/test.py \
    "$config_path" "$checkpoint_path" \
    --eval mpjpe \
    --work-dir "$work_dir" \
    --metrics-out "$eval_json" \
    --gpu-id 0
}

benchmark_one() {
  local run_id="$1"
  local config_path="${CONFIG_PATHS[$run_id]}"
  local checkpoint_path
  local benchmark_json
  checkpoint_path="$(resolve_checkpoint "$run_id")"
  benchmark_json="$(benchmark_json_path "$run_id")"
  mkdir -p "$LOG_ROOT"

  local cmd=(
    "$PYTHON_BIN" tools/analysis/benchmark.py
    "$config_path"
    --checkpoint "$checkpoint_path"
    --device "$BENCHMARK_DEVICE"
    --times "$BENCHMARK_TIMES"
    --warmup "$BENCHMARK_WARMUP"
    --out "$benchmark_json"
  )
  if [[ "$PROFILE_SECTIONS" == "1" ]]; then
    cmd+=(--profile-sections)
  fi
  print_header "Benchmark $run_id"
  run_cmd "${cmd[@]}"
}

smoke_one() {
  local saved_times="$BENCHMARK_TIMES"
  local saved_warmup="$BENCHMARK_WARMUP"
  local run_id="$1"
  local config_path="${CONFIG_PATHS[$run_id]}"
  local checkpoint_path
  local benchmark_json
  BENCHMARK_TIMES=5
  BENCHMARK_WARMUP=2
  checkpoint_path="$(resolve_checkpoint_or_empty "$run_id")"
  benchmark_json="$(benchmark_json_path "$run_id")"
  mkdir -p "$LOG_ROOT"

  local cmd=(
    "$PYTHON_BIN" tools/analysis/benchmark.py
    "$config_path"
    --device "$BENCHMARK_DEVICE"
    --times "$BENCHMARK_TIMES"
    --warmup "$BENCHMARK_WARMUP"
    --out "$benchmark_json"
  )
  if [[ -n "$checkpoint_path" ]]; then
    cmd+=(--checkpoint "$checkpoint_path")
  fi
  if [[ "$PROFILE_SECTIONS" == "1" ]]; then
    cmd+=(--profile-sections)
  fi
  print_header "Smoke benchmark $run_id"
  run_cmd "${cmd[@]}"
  BENCHMARK_TIMES="$saved_times"
  BENCHMARK_WARMUP="$saved_warmup"
}

append_one() {
  local run_id="$1"
  local config_path="${CONFIG_PATHS[$run_id]}"
  local checkpoint_path
  local eval_json
  local benchmark_json
  checkpoint_path="$(resolve_checkpoint "$run_id")"
  eval_json="$(eval_json_path "$run_id")"
  benchmark_json="$(benchmark_json_path "$run_id")"
  require_file "$eval_json"
  require_file "$benchmark_json"

  run_cmd "$PYTHON_BIN" tools/analysis/append_experiment_log.py \
    --experiment-id "$run_id" \
    --config "$config_path" \
    --checkpoint "$checkpoint_path" \
    --eval-json "$eval_json" \
    --benchmark-json "$benchmark_json" \
    --csv "$CSV_PATH" \
    --notes "${RUN_NOTES[$run_id]}"
}

extract_section_latency() {
  mkdir -p "$(dirname "$SECTION_CSV_PATH")"
  "$PYTHON_BIN" - "$SECTION_CSV_PATH" "$LOG_ROOT" "${RUN_IDS[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

section_csv = Path(sys.argv[1])
log_root = Path(sys.argv[2])
run_ids = sys.argv[3:]
rows = []
section_names = []
for run_id in run_ids:
    path = log_root / f'{run_id}_benchmark.json'
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding='utf-8'))
    sections = data.get('section_latency_ms') or {}
    for name in sections:
        if name not in section_names:
            section_names.append(name)
    rows.append((run_id, data, sections))

fieldnames = [
    'experiment_id', 'latency_ms', 'fps', 'params_m',
    'peak_memory_allocated_mb',
] + section_names
section_csv.parent.mkdir(parents=True, exist_ok=True)
with section_csv.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for run_id, data, sections in rows:
        row = {
            'experiment_id': run_id,
            'latency_ms': data.get('latency_ms', ''),
            'fps': data.get('fps', ''),
            'params_m': data.get('params_m', ''),
            'peak_memory_allocated_mb': data.get('peak_memory_allocated_mb', ''),
        }
        row.update(sections)
        writer.writerow(row)
print(f'Wrote section latency CSV: {section_csv}')
PY
}

postprocess_one() {
  evaluate_one "$1"
  benchmark_one "$1"
  append_one "$1"
}

main() {
  local run_id
  validate_runs
  mkdir -p "$LOG_ROOT" "$LAUNCH_LOG_DIR"
  setup_triton_libcuda_env

  case "$MODE" in
    train)
      for run_id in "${RUN_IDS[@]}"; do train_one "$run_id"; done
      ;;
    postprocess)
      for run_id in "${RUN_IDS[@]}"; do postprocess_one "$run_id"; done
      extract_section_latency
      ;;
    benchmark)
      for run_id in "${RUN_IDS[@]}"; do benchmark_one "$run_id"; done
      extract_section_latency
      ;;
    extract)
      for run_id in "${RUN_IDS[@]}"; do append_one "$run_id"; done
      extract_section_latency
      ;;
    smoke)
      for run_id in "${RUN_IDS[@]}"; do smoke_one "$run_id"; done
      extract_section_latency
      ;;
    all)
      for run_id in "${RUN_IDS[@]}"; do
        train_one "$run_id"
        postprocess_one "$run_id"
      done
      extract_section_latency
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "ERROR: Unknown mode: $MODE" >&2
      usage >&2
      exit 1
      ;;
  esac

  if [[ "$MODE" != "-h" && "$MODE" != "--help" && "$MODE" != "help" ]]; then
    echo "Done."
    echo "Experiment CSV:       $CSV_PATH"
    echo "Section latency CSV:  $SECTION_CSV_PATH"
    echo "JSON/log root:        $LOG_ROOT"
  fi
}

main "$@"
