#!/usr/bin/env bash
# Pack archives for GPU rental upload.
# Run from repo root (D:/Resfes_2026/Person-in-WiFi-3D on Windows / project root on *nix).
# Produces two tarballs in OUT_DIR (default /tmp): code + dataset.
# Checkpoints stay in Google Drive — see gpu-runbook §2.

set -euo pipefail

OUT_DIR=${OUT_DIR:-/tmp}
STAMP=$(date +%Y%m%d_%H%M)
CODE_TGZ="${OUT_DIR}/pw3d_code_${STAMP}.tgz"
DATA_TGZ="${OUT_DIR}/pw3d_data_${STAMP}.tgz"

mkdir -p "${OUT_DIR}"

echo "==== Packing code (excluding work_dirs, data, .git, build dirs) ===="
tar \
  --exclude='./work_dirs' \
  --exclude='./data' \
  --exclude='./.git' \
  --exclude='./.claude' \
  --exclude='./paper_assets/manuscript_latex' \
  --exclude='./paper_assets/figures' \
  --exclude='./paper_assets/logs/full_alation_20e' \
  --exclude='./tmp' \
  --exclude='./third_party/*/build' \
  --exclude='./third_party/*/dist' \
  --exclude='*/__pycache__' \
  --exclude='*.egg-info' \
  --exclude='*.pyc' \
  --exclude='wifipose-dataset.zip' \
  -czf "${CODE_TGZ}" .
echo "  ${CODE_TGZ}"
du -sh "${CODE_TGZ}"

echo
echo "==== Packing dataset (whole wifipose dir — small ~14MB compressed) ===="
tar -czf "${DATA_TGZ}" data/wifipose/
echo "  ${DATA_TGZ}"
du -sh "${DATA_TGZ}"

echo
echo "==== Done. Upload to GPU machine: ===="
echo "  scp ${CODE_TGZ} ${DATA_TGZ} root@vast-host:/workspace/"
echo
echo "Then on GPU machine:"
cat <<'REMOTE'
cd /workspace
mkdir -p Person-in-WiFi-3D
tar -xzf pw3d_code_*.tgz -C Person-in-WiFi-3D/
tar -xzf pw3d_data_*.tgz -C Person-in-WiFi-3D/
cd Person-in-WiFi-3D
pip install -r requirements.txt
pip install -e .
# Fetch checkpoints from Google Drive:
#   mkdir -p /workspace/ckpt/M0 /workspace/ckpt/M9
#   gdown 'https://drive.google.com/uc?id=<M0_FILE_ID>' -O /workspace/ckpt/M0/latest.pth
#   gdown 'https://drive.google.com/uc?id=<M9_FILE_ID>' -O /workspace/ckpt/M9/latest.pth
# Run Tier 1:
CKPT_M0=/workspace/ckpt/M0/latest.pth \
CKPT_M9=/workspace/ckpt/M9/latest.pth \
  bash scripts/gpu_run_tier1.sh
# Then Tier 2:
CKPT_M9=/workspace/ckpt/M9/latest.pth \
  bash scripts/gpu_run_tier2.sh
# Bring results home:
tar -czf /tmp/rev2_results.tgz paper_assets/logs/rev2/
# (download /tmp/rev2_results.tgz back to local)
REMOTE
