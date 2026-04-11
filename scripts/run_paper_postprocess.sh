#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

RUN_IDS=(M0 M1 M2 M3 M4 M5)

# Optional manual checkpoint overrides.
# Leave empty to use latest.pth, or the highest epoch_*.pth as fallback.
M0_CKPT=""
M1_CKPT=""
M2_CKPT=""
M3_CKPT=""
M4_CKPT=""
M5_CKPT=""

BENCHMARK_DEVICE="${BENCHMARK_DEVICE:-cuda:0}"
BENCHMARK_TIMES="${BENCHMARK_TIMES:-100}"
BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-10}"

LOG_ROOT="paper_assets/logs"
CSV_PATH="$LOG_ROOT/experiment_log.csv"
FIGURE_ROOT="paper_assets/figures"
FIGURE1_PREFIX="paper_assets/figures/figure1_teaser"

mkdir -p "$LOG_ROOT"
mkdir -p "$FIGURE_ROOT"

get_ckpt_override() {
  local run_id="$1"
  case "$run_id" in
    M0) printf '%s\n' "$M0_CKPT" ;;
    M1) printf '%s\n' "$M1_CKPT" ;;
    M2) printf '%s\n' "$M2_CKPT" ;;
    M3) printf '%s\n' "$M3_CKPT" ;;
    M4) printf '%s\n' "$M4_CKPT" ;;
    M5) printf '%s\n' "$M5_CKPT" ;;
    *) return 1 ;;
  esac
}

get_run_note() {
  local run_id="$1"
  case "$run_id" in
    M0) printf '%s\n' "M0 linear input adapter + Transformer backbone + DETR regression, no bone loss" ;;
    M1) printf '%s\n' "M1 spectral input adapter + Transformer backbone + DETR regression, no bone loss" ;;
    M2) printf '%s\n' "M2 spectral input adapter + Mamba backbone + DETR regression, no bone loss" ;;
    M3) printf '%s\n' "M3 spectral input adapter + Transformer backbone + draft and rectified flow decoder, no bone loss" ;;
    M4) printf '%s\n' "M4 spectral input adapter + Mamba backbone + draft and rectified flow decoder, no bone loss" ;;
    M5) printf '%s\n' "M5 spectral input adapter + Mamba backbone + draft and rectified flow decoder with bone loss (ours)" ;;
    *) printf '%s\n' "" ;;
  esac
}

resolve_run_dir() {
  local run_id="$1"
  local default_run_dir="work_dirs/paper/$run_id"
  local override
  override="$(get_ckpt_override "$run_id")"

  if [[ -n "$override" && -d "$override" ]]; then
    printf '%s\n' "$override"
    return 0
  fi

  if [[ -d "$default_run_dir" ]]; then
    printf '%s\n' "$default_run_dir"
    return 0
  fi

  if [[ -n "$override" && -f "$override" ]]; then
    dirname "$override"
    return 0
  fi

  return 1
}

resolve_config_dir() {
  local run_id="$1"
  local default_run_dir="work_dirs/paper/$run_id"
  local override
  override="$(get_ckpt_override "$run_id")"

  if [[ -n "$override" && -d "$override" ]]; then
    printf '%s\n' "$override"
    return 0
  fi

  if [[ -d "$default_run_dir" ]]; then
    printf '%s\n' "$default_run_dir"
    return 0
  fi

  if [[ -n "$override" && -f "$override" ]]; then
    dirname "$override"
    return 0
  fi

  return 1
}

resolve_config() {
  local run_dir="$1"
  local matches=("$run_dir"/*.py)
  if [[ ! -e "${matches[0]}" ]]; then
    echo "ERROR: No config dump found in $run_dir" >&2
    return 1
  fi
  printf '%s\n' "${matches[0]}"
}

resolve_checkpoint() {
  local run_id="$1"
  local run_dir="$2"
  local override
  override="$(get_ckpt_override "$run_id")"

  if [[ -n "$override" ]]; then
    if [[ -f "$override" ]]; then
      printf '%s\n' "$override"
      return 0
    fi

    if [[ -d "$override" ]]; then
      run_dir="$override"
    else
      echo "ERROR: Override checkpoint for $run_id not found: $override" >&2
      return 1
    fi
  fi

  if [[ -f "$run_dir/latest.pth" ]]; then
    printf '%s\n' "$run_dir/latest.pth"
    return 0
  fi

  local epoch_files=("$run_dir"/epoch_*.pth)
  if [[ ! -e "${epoch_files[0]}" ]]; then
    echo "ERROR: No checkpoint found in $run_dir (expected latest.pth or epoch_*.pth)" >&2
    return 1
  fi

  printf '%s\n' "${epoch_files[@]}" | tr ' ' '\n' | sort -V | tail -n 1
}

run_one() {
  local run_id="$1"
  local run_dir
  local config_dir
  local eval_work_dir="work_dirs/paper_eval/$run_id"
  local eval_json="$LOG_ROOT/${run_id}_eval.json"
  local benchmark_json="$LOG_ROOT/${run_id}_benchmark.json"
  local config_path
  local checkpoint_path
  local notes

  if ! run_dir="$(resolve_run_dir "$run_id")"; then
    echo "ERROR: Missing run directory and override for $run_id" >&2
    return 1
  fi

  config_dir="$(resolve_config_dir "$run_id")"
  config_path="$(resolve_config "$config_dir")"
  checkpoint_path="$(resolve_checkpoint "$run_id" "$run_dir")"
  notes="$(get_run_note "$run_id")"

  mkdir -p "$eval_work_dir"

  echo "============================================================"
  echo "[$run_id] Config:      $config_path"
  echo "[$run_id] Checkpoint:  $checkpoint_path"
  echo "[$run_id] Eval JSON:   $eval_json"
  echo "[$run_id] Bench JSON:  $benchmark_json"
  echo "============================================================"

  python tools/test.py \
    "$config_path" \
    "$checkpoint_path" \
    --eval mpjpe \
    --work-dir "$eval_work_dir" \
    --metrics-out "$eval_json"

  python tools/analysis/benchmark.py \
    "$config_path" \
    --checkpoint "$checkpoint_path" \
    --device "$BENCHMARK_DEVICE" \
    --times "$BENCHMARK_TIMES" \
    --warmup "$BENCHMARK_WARMUP" \
    --out "$benchmark_json"

  python tools/analysis/append_experiment_log.py \
    --experiment-id "$run_id" \
    --config "$config_path" \
    --checkpoint "$checkpoint_path" \
    --eval-json "$eval_json" \
    --benchmark-json "$benchmark_json" \
    --csv "$CSV_PATH" \
    --notes "$notes"
}

for run_id in "${RUN_IDS[@]}"; do
  if ! run_dir="$(resolve_run_dir "$run_id")"; then
    echo "WARNING: Missing run directory and override for $run_id. Skipping." >&2
    continue
  fi
  run_one "$run_id"
done

python tools/analysis/plot_teaser_figure.py \
  --csv "$CSV_PATH" \
  --out-prefix "$FIGURE1_PREFIX" \
  --runs M0 M1 M2 M3 M4 M5 \
  --highlight M5 \
  --xmax 200

echo "Done. Updated $CSV_PATH"
