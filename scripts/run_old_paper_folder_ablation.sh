#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-postprocess}"
if (($# > 0)); then
  shift
fi

OLD_PAPER_ROOT="${OLD_PAPER_ROOT:-/root/Person-in-WiFi-3D/work_dirs/old_ver/root/Person-in-WiFi-3D/work_dirs/paper}"
LOG_ROOT="${LOG_ROOT:-paper_assets/logs/old_paper_folder_ablation}"
EVAL_ROOT="${EVAL_ROOT:-work_dirs/old_paper_folder_eval}"
CSV_PATH="${CSV_PATH:-$LOG_ROOT/experiment_log.csv}"
SECTION_CSV_PATH="${SECTION_CSV_PATH:-$LOG_ROOT/section_latency_log.csv}"
BENCHMARK_DEVICE="${BENCHMARK_DEVICE:-cuda:0}"
BENCHMARK_TIMES="${BENCHMARK_TIMES:-100}"
BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-10}"
PROFILE_SECTIONS="${PROFILE_SECTIONS:-1}"
TEST_GPU="${TEST_GPU:-0}"
AUTO_FIX_TRITON_LIBCUDA="${AUTO_FIX_TRITON_LIBCUDA:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -n "${ONLY_RUN_IDS:-}" ]]; then
  read -r -a RUN_IDS <<<"$ONLY_RUN_IDS"
else
  if [[ ! -d "$OLD_PAPER_ROOT" ]]; then
    echo "ERROR: OLD_PAPER_ROOT does not exist: $OLD_PAPER_ROOT" >&2
    echo "Set OLD_PAPER_ROOT=/path/to/work_dirs/paper" >&2
    exit 1
  fi
  mapfile -t RUN_IDS < <(find "$OLD_PAPER_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -V)
fi

usage() {
  cat <<'EOF'
Usage:
  scripts/run_old_paper_folder_ablation.sh [postprocess|benchmark|extract|help]

Purpose:
  Evaluate/benchmark an existing paper-style folder tree by reading each
  run's dumped config and checkpoint directly from OLD_PAPER_ROOT/$RUN_ID.
  This is intended for old_ver/work_dirs/paper folders with mixed run names
  such as M0, M1, M1FLAT, M1V2, M2, M2CSI, M2FLAT, M3, M4, M5.

Modes:
  postprocess  Evaluate + benchmark + append experiment CSV for every run.
  benchmark    Benchmark only and write *_benchmark.json + section latency CSV.
  extract      Rebuild experiment CSV from existing *_eval.json and *_benchmark.json.
  help         Print this help.

Environment:
  OLD_PAPER_ROOT=/root/Person-in-WiFi-3D/work_dirs/old_ver/root/Person-in-WiFi-3D/work_dirs/paper
  LOG_ROOT=paper_assets/logs/old_paper_folder_ablation
  EVAL_ROOT=work_dirs/old_paper_folder_eval
  ONLY_RUN_IDS="M0 M1 M1FLAT M1V2 M2 M2CSI M2FLAT M3 M4 M5"
  TEST_GPU=0
  BENCHMARK_DEVICE=cuda:0
  BENCHMARK_TIMES=100
  BENCHMARK_WARMUP=10
  PROFILE_SECTIONS=1
  AUTO_FIX_TRITON_LIBCUDA=1
  DRY_RUN=1

Checkpoint lookup per run:
  1. $OLD_PAPER_ROOT/$RUN_ID/latest.pth
  2. Highest version-sorted $OLD_PAPER_ROOT/$RUN_ID/epoch_*.pth

Config lookup per run:
  First sorted *.py file in $OLD_PAPER_ROOT/$RUN_ID.
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

run_dir() {
  printf '%s\n' "$OLD_PAPER_ROOT/$1"
}

eval_dir() {
  printf '%s\n' "$EVAL_ROOT/$1"
}

eval_json_path() {
  printf '%s\n' "$LOG_ROOT/${1}_eval.json"
}

benchmark_json_path() {
  printf '%s\n' "$LOG_ROOT/${1}_benchmark.json"
}

resolve_config() {
  local run_id="$1"
  local dir
  dir="$(run_dir "$run_id")"
  mapfile -t configs < <(find "$dir" -maxdepth 1 -type f -name '*.py' | sort)
  if [[ "${#configs[@]}" -eq 0 ]]; then
    echo "ERROR: No dumped config (*.py) found in $dir" >&2
    return 1
  fi
  printf '%s\n' "${configs[0]}"
}

resolve_checkpoint() {
  local run_id="$1"
  local dir
  dir="$(run_dir "$run_id")"
  if [[ -f "$dir/latest.pth" ]]; then
    printf '%s\n' "$dir/latest.pth"
    return 0
  fi
  mapfile -t checkpoints < <(find "$dir" -maxdepth 1 -type f -name 'epoch_*.pth' | sort -V)
  if [[ "${#checkpoints[@]}" -eq 0 ]]; then
    echo "ERROR: No checkpoint found in $dir" >&2
    return 1
  fi
  local last_index=$((${#checkpoints[@]} - 1))
  printf '%s\n' "${checkpoints[$last_index]}"
}

note_for_run() {
  local run_id="$1"
  case "$run_id" in
    M0) printf '%s\n' "old paper M0 checkpoint folder" ;;
    M1) printf '%s\n' "old paper M1 checkpoint folder" ;;
    M1FLAT) printf '%s\n' "old paper M1FLAT checkpoint folder" ;;
    M1V2) printf '%s\n' "old paper M1V2 checkpoint folder" ;;
    M2) printf '%s\n' "old paper M2 checkpoint folder" ;;
    M2CSI) printf '%s\n' "old paper M2CSI checkpoint folder" ;;
    M2FLAT) printf '%s\n' "old paper M2FLAT checkpoint folder" ;;
    M3) printf '%s\n' "old paper M3 checkpoint folder" ;;
    M4) printf '%s\n' "old paper M4 checkpoint folder" ;;
    M5) printf '%s\n' "old paper M5 checkpoint folder" ;;
    *) printf '%s\n' "old paper checkpoint folder: $run_id" ;;
  esac
}

validate_runs() {
  require_file "tools/test.py"
  require_file "tools/analysis/benchmark.py"
  require_file "tools/analysis/append_experiment_log.py"

  local run_id dir
  for run_id in "${RUN_IDS[@]}"; do
    dir="$(run_dir "$run_id")"
    if [[ ! -d "$dir" ]]; then
      echo "ERROR: Missing run directory: $dir" >&2
      exit 1
    fi
    resolve_config "$run_id" >/dev/null
    resolve_checkpoint "$run_id" >/dev/null
  done
}

evaluate_one() {
  local run_id="$1"
  local config checkpoint out_dir eval_json
  config="$(resolve_config "$run_id")"
  checkpoint="$(resolve_checkpoint "$run_id")"
  out_dir="$(eval_dir "$run_id")"
  eval_json="$(eval_json_path "$run_id")"
  mkdir -p "$out_dir" "$LOG_ROOT"

  print_header "Evaluate $run_id"
  echo "Config:     $config"
  echo "Checkpoint: $checkpoint"
  echo "Eval JSON:  $eval_json"

  run_cmd env "CUDA_VISIBLE_DEVICES=$TEST_GPU" "$PYTHON_BIN" tools/test.py \
    "$config" \
    "$checkpoint" \
    --eval mpjpe \
    --work-dir "$out_dir" \
    --metrics-out "$eval_json" \
    --gpu-id 0
}

benchmark_one() {
  local run_id="$1"
  local config checkpoint benchmark_json
  config="$(resolve_config "$run_id")"
  checkpoint="$(resolve_checkpoint "$run_id")"
  benchmark_json="$(benchmark_json_path "$run_id")"
  mkdir -p "$LOG_ROOT"

  print_header "Benchmark $run_id"
  echo "Config:         $config"
  echo "Checkpoint:     $checkpoint"
  echo "Benchmark JSON: $benchmark_json"

  local cmd=(
    env
    "CUDA_VISIBLE_DEVICES=$TEST_GPU"
    "$PYTHON_BIN"
    tools/analysis/benchmark.py
    "$config"
    --checkpoint "$checkpoint"
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

append_one() {
  local run_id="$1"
  local config checkpoint eval_json benchmark_json notes
  config="$(resolve_config "$run_id")"
  checkpoint="$(resolve_checkpoint "$run_id")"
  eval_json="$(eval_json_path "$run_id")"
  benchmark_json="$(benchmark_json_path "$run_id")"
  notes="$(note_for_run "$run_id")"
  require_file "$eval_json"
  require_file "$benchmark_json"

  run_cmd "$PYTHON_BIN" tools/analysis/append_experiment_log.py \
    --experiment-id "$run_id" \
    --config "$config" \
    --checkpoint "$checkpoint" \
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
  case "$MODE" in
    -h|--help|help)
      usage
      return 0
      ;;
  esac

  validate_runs
  mkdir -p "$LOG_ROOT"

  case "$MODE" in
    postprocess|benchmark)
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

  local run_id
  case "$MODE" in
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
  esac
}

main "$@"
