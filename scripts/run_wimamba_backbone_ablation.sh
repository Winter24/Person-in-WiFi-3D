#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-all}"
if (($# > 0)); then
  shift
fi

# Full WiMamba encoder ladder inside opera.WiTiDARHead.
# M1V1 is intentionally not a separate run because wimamba_v1.py is a
# compatibility wrapper for the current factorized WiMamba implementation.
RUN_IDS=(M1FCT M1V2 M1V3 M1FLAT M2FLAT M2D M2CSI M2C M2CP M2CPA)
if [[ -n "${ONLY_RUN_IDS:-}" ]]; then
  read -r -a RUN_IDS <<<"$ONLY_RUN_IDS"
fi

TRAIN_GPU="${TRAIN_GPU:-0}"
TEST_GPU="${TEST_GPU:-0}"
SEED="${SEED:-42}"
PYTHONHASHSEED_VALUE="${PYTHONHASHSEED_VALUE:-42}"
CUBLAS_WORKSPACE_CONFIG_VALUE="${CUBLAS_WORKSPACE_CONFIG_VALUE:-:4096:8}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
BENCHMARK_DEVICE="${BENCHMARK_DEVICE:-cuda:0}"
BENCHMARK_TIMES="${BENCHMARK_TIMES:-100}"
BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-10}"
PROFILE_SECTIONS="${PROFILE_SECTIONS:-1}"
AUTO_FIX_TRITON_LIBCUDA="${AUTO_FIX_TRITON_LIBCUDA:-1}"
AUTO_RESUME="${AUTO_RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"

WORK_ROOT="${WORK_ROOT:-work_dirs/wimamba_backbone_ablation}"
EVAL_ROOT="${EVAL_ROOT:-work_dirs/wimamba_backbone_eval}"
LOG_ROOT="${LOG_ROOT:-paper_assets/logs/wimamba_backbone_ablation}"
CSV_PATH="${CSV_PATH:-$LOG_ROOT/experiment_log.csv}"
SECTION_CSV_PATH="${SECTION_CSV_PATH:-$LOG_ROOT/section_latency_log.csv}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-$LOG_ROOT/launch_logs}"

declare -A CONFIG_PATHS=(
  [M1FCT]="configs/wifi/wi_tidir_wifi.py"
  [M1V2]="configs/wifi/wi_tidir_wifi_mamba_v2.py"
  [M1V3]="configs/wifi/wi_tidir_wifi_mamba_v3.py"
  [M1FLAT]="configs/wifi/wi_tidir_wifi_mamba1_flatten.py"
  [M2FLAT]="configs/wifi/wi_tidir_wifi_mamba2_flatten.py"
  [M2D]="configs/wifi/wi_tidir_wifi_mamba2_dropin.py"
  [M2CSI]="configs/wifi/wi_tidir_wifi_mamba2_flattened.py"
  [M2C]="configs/wifi/wi_tidir_wifi_mamba2_crossscan.py"
  [M2CP]="configs/wifi/wi_tidir_wifi_mamba2_crossscan_pos.py"
  [M2CPA]="configs/wifi/wi_tidir_wifi_mamba2_crossscan_pos_attn.py"
)

declare -A RUN_NOTES=(
  [M1FCT]="Mamba1 current factorized: temporal Mamba + bidirectional spatial Mamba"
  [M1V2]="Mamba1 fast factorized: temporal Mamba + Conv1d antenna mixer"
  [M1V3]="Mamba1 fast factorized: temporal Mamba + Linear antenna mixer"
  [M1FLAT]="Mamba1 flattened single-route L=180 scan"
  [M2FLAT]="Mamba2 flattened single-route L=180 wrapper"
  [M2D]="Mamba2 drop-in factorized: temporal + bidirectional spatial scan"
  [M2CSI]="Mamba2 CSI flattened single-route via WiMamba2CSIEncoder"
  [M2C]="Mamba2 CSI two-route cross-scan"
  [M2CP]="Mamba2 CSI two-route cross-scan + CSI positional embedding"
  [M2CPA]="Mamba2 CSI cross-scan + CSI positional embedding + final attention"
)

usage() {
  cat <<'EOF'
Usage:
  scripts/run_wimamba_backbone_ablation.sh [all|train|postprocess|benchmark|extract|smoke|help]

Purpose:
  Run the full WiMamba encoder ladder inside opera.WiTiDARHead. This is the
  WiTiDAR/M4-style ablation path, not the PETRHead encoder path.

Modes:
  all          Train each selected backbone, then evaluate, benchmark, append CSV.
  train        Train selected backbones sequentially with fixed seed/repro env.
  postprocess  Evaluate + benchmark + append CSV for existing checkpoints.
  benchmark    Benchmark only for existing checkpoints.
  extract      Rebuild CSV rows from existing *_eval.json and *_benchmark.json.
  smoke        Short benchmark with --times 5 --warmup 2.

Run IDs:
  M1FCT   configs/wifi/wi_tidir_wifi.py
  M1V2    configs/wifi/wi_tidir_wifi_mamba_v2.py
  M1V3    configs/wifi/wi_tidir_wifi_mamba_v3.py
  M1FLAT  configs/wifi/wi_tidir_wifi_mamba1_flatten.py
  M2FLAT  configs/wifi/wi_tidir_wifi_mamba2_flatten.py
  M2D     configs/wifi/wi_tidir_wifi_mamba2_dropin.py
  M2CSI   configs/wifi/wi_tidir_wifi_mamba2_flattened.py
  M2C     configs/wifi/wi_tidir_wifi_mamba2_crossscan.py
  M2CP    configs/wifi/wi_tidir_wifi_mamba2_crossscan_pos.py
  M2CPA   configs/wifi/wi_tidir_wifi_mamba2_crossscan_pos_attn.py

Environment overrides:
  ONLY_RUN_IDS="M1V3 M1FLAT M2FLAT"  Select a subset.
  MAX_EPOCHS=5                       Override runner.max_epochs for quick ablation.
  TRAIN_GPU=0 TEST_GPU=0             GPU id exposed as CUDA_VISIBLE_DEVICES.
  SEED=42                            Train seed passed to tools/train.py.
  PYTHONHASHSEED_VALUE=42            Python hash seed matching docs/paper recipe.
  CUBLAS_WORKSPACE_CONFIG_VALUE=:4096:8
  BENCHMARK_DEVICE=cuda:0            Device used by tools/analysis/benchmark.py.
  PROFILE_SECTIONS=1                 Pass --profile-sections to benchmark.
  AUTO_FIX_TRITON_LIBCUDA=1          Run scripts/auto_fix_triton_libcuda.sh first.
  AUTO_RESUME=0                      Set to 1 only to continue the exact same run.
  DRY_RUN=1                          Print commands without running them.
  M1V3_CKPT=/path/latest.pth         Per-run checkpoint override for postprocess.

Checkpoint lookup:
  1. Per-run override such as M1V3_CKPT=/path/to/checkpoint.pth.
  2. $WORK_ROOT/$RUN_ID/latest.pth.
  3. Highest version-sorted $WORK_ROOT/$RUN_ID/epoch_*.pth.
EOF
}

print_header() {
  echo "============================================================"
  echo "$1"
  echo "============================================================"
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
      usage >&2
      exit 2
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

resolve_eval_config() {
  local run_id="$1"
  local run_dir expected_config copied_config
  run_dir="$(run_work_dir "$run_id")"
  expected_config="${CONFIG_PATHS[$run_id]}"
  copied_config="$run_dir/$(basename "$expected_config")"

  if [[ -f "$copied_config" ]]; then
    printf '%s\n' "$copied_config"
    return 0
  fi

  printf '%s\n' "$expected_config"
}

resolve_checkpoint() {
  local run_id="$1"
  local run_dir override_var override
  run_dir="$(run_work_dir "$run_id")"
  override_var="${run_id}_CKPT"
  override="${!override_var:-}"

  if [[ -n "$override" ]]; then
    if [[ -f "$override" ]]; then
      printf '%s\n' "$override"
      return 0
    fi
    echo "ERROR: $override_var points to a missing checkpoint: $override" >&2
    return 1
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
  local cmd=()
  work_dir="$(run_work_dir "$run_id")"
  log_path="$LAUNCH_LOG_DIR/${run_id}_train.log"

  mkdir -p "$work_dir" "$LAUNCH_LOG_DIR"

  cmd=(
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
    --cfg-options
    "runner.max_epochs=$MAX_EPOCHS"
  )

  if [[ "$AUTO_RESUME" == "1" && -f "$work_dir/latest.pth" ]]; then
    cmd+=(--auto-resume)
  fi

  print_header "Train $run_id"
  echo "Config:   $config_path"
  echo "Seed:     $SEED"
  echo "Epochs:   $MAX_EPOCHS"
  echo "Work dir: $work_dir"
  echo "Log:      $log_path"
  if [[ "$AUTO_RESUME" != "1" && -f "$work_dir/latest.pth" ]]; then
    echo "Resume:   disabled; set AUTO_RESUME=1 only for the exact same config."
  fi

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

  print_header "Benchmark $run_id"
  echo "Config:         $config_path"
  echo "Checkpoint:     $checkpoint_path"
  echo "Benchmark JSON: $benchmark_json"

  local cmd=(
    env
    "CUDA_VISIBLE_DEVICES=$TEST_GPU"
    "$PYTHON_BIN"
    tools/analysis/benchmark.py
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

  print_header "Smoke benchmark $run_id"
  echo "Config:         $config_path"
  if [[ -n "$checkpoint_path" ]]; then
    echo "Checkpoint:     $checkpoint_path"
  else
    echo "Checkpoint:     <none; untrained model smoke test>"
  fi
  echo "Benchmark JSON: $benchmark_json"

  local cmd=(
    env
    "CUDA_VISIBLE_DEVICES=$TEST_GPU"
    "$PYTHON_BIN"
    tools/analysis/benchmark.py
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

  run_cmd "${cmd[@]}"
  BENCHMARK_TIMES="$saved_times"
  BENCHMARK_WARMUP="$saved_warmup"
}

append_one() {
  local run_id="$1"
  local config_path checkpoint_path eval_json benchmark_json notes
  config_path="$(resolve_eval_config "$run_id")"
  checkpoint_path="$(resolve_checkpoint "$run_id")"
  eval_json="$(eval_json_path "$run_id")"
  benchmark_json="$(benchmark_json_path "$run_id")"
  notes="${RUN_NOTES[$run_id]}"

  require_file "$eval_json"
  require_file "$benchmark_json"

  run_cmd "$PYTHON_BIN" tools/analysis/append_experiment_log.py \
    --experiment-id "$run_id" \
    --config "$config_path" \
    --checkpoint "$checkpoint_path" \
    --eval-json "$eval_json" \
    --benchmark-json "$benchmark_json" \
    --csv "$CSV_PATH" \
    --notes "$notes"
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
    'experiment_id',
    'latency_ms',
    'fps',
    'params_m',
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

  case "$MODE" in
    -h|--help|help)
      usage
      return 0
      ;;
  esac

  validate_runs
  mkdir -p "$LOG_ROOT" "$LAUNCH_LOG_DIR"

  case "$MODE" in
    train|postprocess|benchmark|smoke|all)
      setup_triton_libcuda_env
      ;;
    extract)
      ;;
    *)
      echo "ERROR: Unknown mode: $MODE" >&2
      usage >&2
      exit 1
      ;;
  esac

  case "$MODE" in
    train)
      for run_id in "${RUN_IDS[@]}"; do
        train_one "$run_id"
      done
      ;;
    postprocess)
      for run_id in "${RUN_IDS[@]}"; do
        postprocess_one "$run_id"
      done
      extract_section_latency
      ;;
    benchmark)
      for run_id in "${RUN_IDS[@]}"; do
        benchmark_one "$run_id"
      done
      extract_section_latency
      ;;
    extract)
      for run_id in "${RUN_IDS[@]}"; do
        append_one "$run_id"
      done
      extract_section_latency
      ;;
    smoke)
      for run_id in "${RUN_IDS[@]}"; do
        smoke_one "$run_id"
      done
      extract_section_latency
      ;;
    all)
      for run_id in "${RUN_IDS[@]}"; do
        train_one "$run_id"
        postprocess_one "$run_id"
      done
      extract_section_latency
      ;;
    *)
      echo "ERROR: Unknown mode: $MODE" >&2
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
