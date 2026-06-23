#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash tools/analysis/run_full_paper_qualitative.sh <source-video-root>" >&2
  echo "The root must contain S11_06, S52_40, and S23_12 folders." >&2
  exit 2
fi

SOURCE_VIDEO_ROOT="$1"

python tools/analysis/extract_qualitative_rgb_frames.py \
  --source-video-root "$SOURCE_VIDEO_ROOT"

python tools/analysis/render_qualitative_figure.py \
  --full-paper \
  --device cuda:0 \
  --score-thr 0.2 \
  --match-threshold-mm 500 \
  --match-quality-thr-mm 500 \
  --dpi 300
