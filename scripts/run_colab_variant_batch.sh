#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

PAPER_DIR="${PAPER_DIR:-/content/drive/MyDrive/RESFES2026/Test}"
OUTPUT_DIR="${OUTPUT_DIR:-$PAPER_DIR}"
EVAL_ROOT="${EVAL_ROOT:-work_dirs/paper_eval_colab}"
BENCHMARK_DEVICE="${BENCHMARK_DEVICE:-cuda:0}"
BENCHMARK_TIMES="${BENCHMARK_TIMES:-100}"
BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-10}"

BACKBONE_DIR="opera/models/backbones"
INIT_FILE="$BACKBONE_DIR/__init__.py"
LIVE_FILE="$BACKBONE_DIR/wimamba_v1.py"

RUNS=(
  "M1_v1"
  "M1_v3"
  "M5_v1"
  "M5_v2"
  "M5_v3"
)

if (($# > 0)); then
  RUNS=("$@")
fi

usage() {
  cat <<'EOF'
Usage:
  scripts/run_colab_variant_batch.sh
  scripts/run_colab_variant_batch.sh M1_v1 M5_v2

Environment overrides:
  PAPER_DIR, OUTPUT_DIR, EVAL_ROOT
  BENCHMARK_DEVICE, BENCHMARK_TIMES, BENCHMARK_WARMUP
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cp "$INIT_FILE" /tmp/backbones___init__.py.bak
cp "$LIVE_FILE" /tmp/wimamba_v1.py.bak

restore_backbone_files() {
  cp /tmp/backbones___init__.py.bak "$INIT_FILE"
  cp /tmp/wimamba_v1.py.bak "$LIVE_FILE"
}

trap restore_backbone_files EXIT

cat > "$INIT_FILE" <<'PY'
# Copyright (c) Hikvision Research Institute. All rights reserved.
from .wimamba_v1 import WiMambaEncoder
__all__ = ['WiMambaEncoder']
PY

select_variant_file() {
  local run_name="$1"
  case "$run_name" in
    *_v1*) echo "$BACKBONE_DIR/wimamba_v1.py" ;;
    *_v2*) echo "$BACKBONE_DIR/wimamba_v2.py" ;;
    *_v3*) echo "$BACKBONE_DIR/wimamba_v3.py" ;;
    *) echo "$BACKBONE_DIR/wimamba_v1.py" ;;
  esac
}

apply_variant_file() {
  local variant_file="$1"
  if [[ "$variant_file" == "$LIVE_FILE" ]]; then
    return 0
  fi
  cp "$variant_file" "$LIVE_FILE"
}

resolve_config_path() {
  local run_dir="$1"
  local config_path
  config_path=$(ls "$run_dir"/*.py | head -n 1)
  if [[ -z "$config_path" ]]; then
    echo "ERROR: No config dump found in $run_dir" >&2
    return 1
  fi
  printf '%s\n' "$config_path"
}

resolve_checkpoint_path() {
  local run_dir="$1"
  if [[ -f "$run_dir/latest.pth" ]]; then
    printf '%s\n' "$run_dir/latest.pth"
    return 0
  fi

  local ckpt_path
  ckpt_path=$(ls "$run_dir"/epoch_*.pth 2>/dev/null | sort -V | tail -n 1 || true)
  if [[ -z "$ckpt_path" ]]; then
    echo "ERROR: No checkpoint found in $run_dir" >&2
    return 1
  fi
  printf '%s\n' "$ckpt_path"
}

mkdir -p "$OUTPUT_DIR"

for RUN_NAME in "${RUNS[@]}"; do
  RUN_DIR="$PAPER_DIR/$RUN_NAME"
  if [[ ! -d "$RUN_DIR" ]]; then
    echo "WARNING: Missing run directory $RUN_DIR. Skipping." >&2
    continue
  fi

  echo "============================================================"
  echo "Processing $RUN_NAME"
  echo "Run dir: $RUN_DIR"
  echo "============================================================"

  CONFIG_PATH="$(resolve_config_path "$RUN_DIR")"
  CKPT_PATH="$(resolve_checkpoint_path "$RUN_DIR")"
  VARIANT_FILE="$(select_variant_file "$RUN_NAME")"

  apply_variant_file "$VARIANT_FILE"

  python tools/test.py \
    "$CONFIG_PATH" \
    "$CKPT_PATH" \
    --eval mpjpe \
    --work-dir "$EVAL_ROOT/$RUN_NAME" \
    --metrics-out "$OUTPUT_DIR/${RUN_NAME}_eval.json"

  python tools/analysis/benchmark.py \
    "$CONFIG_PATH" \
    --checkpoint "$CKPT_PATH" \
    --device "$BENCHMARK_DEVICE" \
    --times "$BENCHMARK_TIMES" \
    --warmup "$BENCHMARK_WARMUP" \
    --out "$OUTPUT_DIR/${RUN_NAME}_benchmark.json"

  python tools/analysis/append_experiment_log.py \
    --experiment-id "$RUN_NAME" \
    --config "$CONFIG_PATH" \
    --checkpoint "$CKPT_PATH" \
    --eval-json "$OUTPUT_DIR/${RUN_NAME}_eval.json" \
    --benchmark-json "$OUTPUT_DIR/${RUN_NAME}_benchmark.json" \
    --csv "$OUTPUT_DIR/experiment_log.csv" \
    --notes ""
done

echo "All runs finished."
echo "CSV: $OUTPUT_DIR/experiment_log.csv"
