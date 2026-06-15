#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-all}"
if (($# > 0)); then
  shift
fi

RUN_IDS=(E_NOFLOW E_RF1 E_RF2 E_RF4 E_RF8 T_FW1 T_FW2 T_FW5 T_N0 T_N005)
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

M9_BASE_CKPT="${M9_BASE_CKPT:-work_dirs/full_alation_20e/M9/latest.pth}"
WORK_ROOT="${WORK_ROOT:-work_dirs/m9_flow_step2_ablation}"
EVAL_ROOT="${EVAL_ROOT:-work_dirs/m9_flow_step2_eval}"
LOG_ROOT="${LOG_ROOT:-paper_assets/logs/m9_flow_step2_ablation}"
CSV_PATH="${CSV_PATH:-$LOG_ROOT/experiment_log.csv}"
SECTION_CSV_PATH="${SECTION_CSV_PATH:-$LOG_ROOT/section_latency_log.csv}"
SUMMARY_CSV_PATH="${SUMMARY_CSV_PATH:-$LOG_ROOT/flow_diagnostic_summary.csv}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-$LOG_ROOT/launch_logs}"

declare -A CONFIG_PATHS=(
  [E_NOFLOW]="configs/wifi/wi_tidir_wifi_mamba2_flattened_eval_no_flow.py"
  [E_RF1]="configs/wifi/wi_tidir_wifi_mamba2_flattened.py"
  [E_RF2]="configs/wifi/wi_tidir_wifi_mamba2_flattened_eval_rf_2step.py"
  [E_RF4]="configs/wifi/wi_tidir_wifi_mamba2_flattened_eval_rf_4step.py"
  [E_RF8]="configs/wifi/wi_tidir_wifi_mamba2_flattened_eval_rf_8step.py"
  [T_FW1]="configs/wifi/wi_tidir_wifi_mamba2_flattened_flow_w1.py"
  [T_FW2]="configs/wifi/wi_tidir_wifi_mamba2_flattened_flow_w2.py"
  [T_FW5]="configs/wifi/wi_tidir_wifi_mamba2_flattened_flow_w5.py"
  [T_N0]="configs/wifi/wi_tidir_wifi_mamba2_flattened_flow_noise_0.py"
  [T_N005]="configs/wifi/wi_tidir_wifi_mamba2_flattened_flow_noise_005.py"
)

declare -A RUN_KIND=(
  [E_NOFLOW]="eval"
  [E_RF1]="eval"
  [E_RF2]="eval"
  [E_RF4]="eval"
  [E_RF8]="eval"
  [T_FW1]="train"
  [T_FW2]="train"
  [T_FW5]="train"
  [T_N0]="train"
  [T_N005]="train"
)

declare -A DEFAULT_CKPTS=(
  [E_NOFLOW]="$M9_BASE_CKPT"
  [E_RF1]="$M9_BASE_CKPT"
  [E_RF2]="$M9_BASE_CKPT"
  [E_RF4]="$M9_BASE_CKPT"
  [E_RF8]="$M9_BASE_CKPT"
)

declare -A RUN_NOTES=(
  [E_NOFLOW]="M9 checkpoint evaluated with flow disabled; isolates draft output after flow-trained checkpoint"
  [E_RF1]="M9 checkpoint evaluated with rectified flow, Euler num_steps=1"
  [E_RF2]="M9 checkpoint evaluated with rectified flow, Euler num_steps=2"
  [E_RF4]="M9 checkpoint evaluated with rectified flow, Euler num_steps=4"
  [E_RF8]="M9 checkpoint evaluated with rectified flow, Euler num_steps=8"
  [T_FW1]="Train M9 flattened with rectified flow loss weight 1.0"
  [T_FW2]="Train M9 flattened with rectified flow loss weight 2.0"
  [T_FW5]="Train M9 flattened with rectified flow loss weight 5.0"
  [T_N0]="Train M9 flattened with flow noise strength 0.0 and loss weight 10.0"
  [T_N005]="Train M9 flattened with flow noise strength 0.05 and loss weight 10.0"
)

usage() {
  cat <<'EOF'
Usage:
  scripts/run_m9_flow_step2_ablation.sh [all|train|postprocess|benchmark|extract|smoke|help]

Purpose:
  End-to-end Step 2 flow diagnostics for canonical M9 flattened Mamba2-CSI.

Run IDs:
  E_NOFLOW  Eval existing M9 checkpoint with flow disabled.
  E_RF1     Eval existing M9 checkpoint with rectified flow num_steps=1.
  E_RF2     Eval existing M9 checkpoint with rectified flow num_steps=2.
  E_RF4     Eval existing M9 checkpoint with rectified flow num_steps=4.
  E_RF8     Eval existing M9 checkpoint with rectified flow num_steps=8.
  T_FW1     Train/eval M9 with loss_flow_weight=1.0.
  T_FW2     Train/eval M9 with loss_flow_weight=2.0.
  T_FW5     Train/eval M9 with loss_flow_weight=5.0.
  T_N0      Train/eval M9 with flow_noise_strength=0.0.
  T_N005    Train/eval M9 with flow_noise_strength=0.05.

Modes:
  all          For E_*: evaluate+benchmark+append. For T_*: train then evaluate+benchmark+append.
  train        Train selected T_* runs only; E_* runs are skipped.
  postprocess  Evaluate + benchmark + append CSV for selected runs.
  benchmark    Benchmark selected runs only.
  extract      Rebuild CSV rows and diagnostic summary from existing JSON files.
  smoke        Short benchmark sanity check, usually times=5, warmup=2.

Environment overrides:
  ONLY_RUN_IDS="E_NOFLOW E_RF1 E_RF4"  Select a subset.
  M9_BASE_CKPT=work_dirs/full_alation_20e/M9/latest.pth
  TRAIN_GPU=0 TEST_GPU=0
  BENCHMARK_DEVICE=cuda:0
  WORK_ROOT=work_dirs/m9_flow_step2_ablation
  LOG_ROOT=paper_assets/logs/m9_flow_step2_ablation
  DRY_RUN=1
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
    if [[ -z "${RUN_KIND[$run_id]:-}" ]]; then
      echo "ERROR: Missing run kind for $run_id" >&2
      exit 1
    fi
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

resolve_eval_config() {
  local run_id="$1"
  local run_dir config_basename config_copy
  run_dir="$(run_work_dir "$run_id")"
  config_basename="$(basename "${CONFIG_PATHS[$run_id]}")"
  config_copy="$run_dir/$config_basename"

  if [[ -f "$config_copy" ]]; then
    printf '%s\n' "$config_copy"
    return 0
  fi
  printf '%s\n' "${CONFIG_PATHS[$run_id]}"
}

resolve_checkpoint() {
  local run_id="$1"
  local run_dir override_var override default_ckpt
  run_dir="$(run_work_dir "$run_id")"
  override_var="${run_id}_CKPT"
  override="${!override_var:-}"
  default_ckpt="${DEFAULT_CKPTS[$run_id]:-}"

  if [[ -n "$override" ]]; then
    require_file "$override"
    printf '%s\n' "$override"
    return 0
  fi
  if [[ -n "$default_ckpt" ]]; then
    require_file "$default_ckpt"
    printf '%s\n' "$default_ckpt"
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
  local work_dir log_path

  if [[ "${RUN_KIND[$run_id]}" == "eval" ]]; then
    echo "Skip train for eval-only run: $run_id"
    return 0
  fi

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
  local config_path checkpoint_path work_dir eval_json
  config_path="$(resolve_eval_config "$run_id")"
  checkpoint_path="$(resolve_checkpoint "$run_id")"
  work_dir="$(eval_work_dir "$run_id")"
  eval_json="$(eval_json_path "$run_id")"
  mkdir -p "$work_dir" "$LOG_ROOT"

  print_header "Evaluate $run_id"
  echo "Config:     $config_path"
  echo "Checkpoint: $checkpoint_path"
  echo "Eval JSON:  $eval_json"

  run_cmd env "CUDA_VISIBLE_DEVICES=$TEST_GPU" "$PYTHON_BIN" tools/test.py \
    "$config_path" \
    "$checkpoint_path" \
    --eval mpjpe \
    --work-dir "$work_dir" \
    --metrics-out "$eval_json" \
    --gpu-id 0
}

benchmark_one() {
  local run_id="$1"
  local config_path checkpoint_path benchmark_json
  config_path="$(resolve_eval_config "$run_id")"
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
  echo "Benchmark JSON: $benchmark_json"
  run_cmd "${cmd[@]}"
}

smoke_one() {
  local run_id="$1"
  local config_path checkpoint_path benchmark_json
  local saved_times="$BENCHMARK_TIMES"
  local saved_warmup="$BENCHMARK_WARMUP"
  BENCHMARK_TIMES=5
  BENCHMARK_WARMUP=2
  config_path="$(resolve_eval_config "$run_id")"
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
  local config_path checkpoint_path eval_json benchmark_json
  config_path="$(resolve_eval_config "$run_id")"
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

extract_flow_diagnostic_summary() {
  mkdir -p "$(dirname "$SUMMARY_CSV_PATH")"
  "$PYTHON_BIN" - "$SUMMARY_CSV_PATH" "$LOG_ROOT" "${RUN_IDS[@]}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
log_root = Path(sys.argv[2])
run_ids = sys.argv[3:]
fieldnames = [
    'experiment_id', 'mpjpe', 'matched_only_mpjpe', 'missed_persons',
    'matched_persons', 'total_gt_persons', 'miss_rate_pct',
    'miss_penalty_mm', 'mpjpe_1p', 'mpjpe_2p', 'mpjpe_3p',
    'mpjpeh', 'mpjpev', 'mpjped', 'latency_ms', 'fps',
    'params_m', 'peak_memory_allocated_mb',
]
rows = []
for run_id in run_ids:
    eval_path = log_root / f'{run_id}_eval.json'
    bench_path = log_root / f'{run_id}_benchmark.json'
    if not eval_path.exists():
        continue
    ev = json.loads(eval_path.read_text(encoding='utf-8'))
    bm = {}
    if bench_path.exists():
        bm = json.loads(bench_path.read_text(encoding='utf-8'))
    total = int(ev.get('total_gt_persons') or 0)
    missed = int(ev.get('missed_persons') or 0)
    matched = int(ev.get('matched_persons') or 0)
    penalty = float(ev.get('miss_penalty_mm') or 0.0)
    mpjpe = float(ev.get('mpjpe') or math.nan)
    if matched > 0 and total > 0:
        matched_only = (mpjpe * total - missed * penalty) / matched
    else:
        matched_only = math.nan
    miss_rate = (missed / total * 100.0) if total else math.nan
    rows.append({
        'experiment_id': run_id,
        'mpjpe': mpjpe,
        'matched_only_mpjpe': matched_only,
        'missed_persons': missed,
        'matched_persons': matched,
        'total_gt_persons': total,
        'miss_rate_pct': miss_rate,
        'miss_penalty_mm': penalty,
        'mpjpe_1p': ev.get('mpjpe_1p', ''),
        'mpjpe_2p': ev.get('mpjpe_2p', ''),
        'mpjpe_3p': ev.get('mpjpe_3p', ''),
        'mpjpeh': ev.get('mpjpeh', ''),
        'mpjpev': ev.get('mpjpev', ''),
        'mpjped': ev.get('mpjped', ''),
        'latency_ms': bm.get('latency_ms', ''),
        'fps': bm.get('fps', ''),
        'params_m': bm.get('params_m', ''),
        'peak_memory_allocated_mb': bm.get('peak_memory_allocated_mb', ''),
    })

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f'Wrote flow diagnostic summary CSV: {summary_csv}')
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
      extract_flow_diagnostic_summary
      ;;
    benchmark)
      for run_id in "${RUN_IDS[@]}"; do benchmark_one "$run_id"; done
      extract_section_latency
      ;;
    extract)
      for run_id in "${RUN_IDS[@]}"; do append_one "$run_id"; done
      extract_section_latency
      extract_flow_diagnostic_summary
      ;;
    smoke)
      for run_id in "${RUN_IDS[@]}"; do smoke_one "$run_id"; done
      extract_section_latency
      ;;
    all)
      for run_id in "${RUN_IDS[@]}"; do
        if [[ "${RUN_KIND[$run_id]}" == "train" ]]; then
          train_one "$run_id"
        fi
        postprocess_one "$run_id"
      done
      extract_section_latency
      extract_flow_diagnostic_summary
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
    echo "Diagnostic CSV:       $SUMMARY_CSV_PATH"
    echo "JSON/log root:        $LOG_ROOT"
  fi
}

main "$@"
