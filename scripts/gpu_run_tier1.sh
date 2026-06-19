#!/usr/bin/env bash
# Tier 1 GPU run — G1 (M9_RF2 eval) + G2 (M0 eval) with per-joint MPJDLE export.
# See docs/superpowers/specs/2026-06-19-gpu-runbook-rev2.md for context.
#
# Usage on rented GPU machine (RTX 3090/4090, CUDA 11.8, PyTorch 2.0):
#   cd /workspace/Person-in-WiFi-3D
#   CKPT_M0=/workspace/ckpt/M0/latest.pth CKPT_M9=/workspace/ckpt/M9/latest.pth \
#     bash scripts/gpu_run_tier1.sh
#
# Expected wall time: ~10 minutes.
# Outputs go to paper_assets/logs/rev2/.

set -euo pipefail

CKPT_M0=${CKPT_M0:?Set CKPT_M0 to M0 checkpoint path (e.g. /workspace/ckpt/M0/latest.pth)}
CKPT_M9=${CKPT_M9:?Set CKPT_M9 to M9 checkpoint path}
OUT_DIR=${OUT_DIR:-paper_assets/logs/rev2}
M9_RF2_CFG=${M9_RF2_CFG:-configs/wifi/wi_tidir_wifi_mamba2_flattened_eval_rf_2step.py}
M0_CFG=${M0_CFG:-configs/wifi/petr_wifi.py}

mkdir -p "${OUT_DIR}"

echo "==== [G1] Eval M9_RF2 (Mamba2-CSI WiTiDAR + 2-step rectified flow) ===="
python tools/test.py \
  "${M9_RF2_CFG}" \
  "${CKPT_M9}" \
  --eval mpjpe \
  --metrics-out "${OUT_DIR}/M9_RF2_eval.json"

echo
echo "==== [G2] Eval M0 (original PETR-style baseline) ===="
python tools/test.py \
  "${M0_CFG}" \
  "${CKPT_M0}" \
  --eval mpjpe \
  --metrics-out "${OUT_DIR}/M0_eval.json"

echo
echo "==== Tier 1 done. Sanity check: ===="
python - "${OUT_DIR}" <<'PY'
import json, os, sys
TOL = 0.5
out_dir = sys.argv[1]
EXPECT = {
    os.path.join(out_dir, 'M9_RF2_eval.json'): 165.487,
    os.path.join(out_dir, 'M0_eval.json'):     172.540,
}
ok = True
for path, target in EXPECT.items():
    d = json.load(open(path))
    have = d['mpjpe']
    drift = abs(have - target)
    marker = 'OK ' if drift <= TOL else 'FAIL'
    print(f'[{marker}] {path:60s}  mpjpe={have:.3f}  expected={target:.3f}  drift={drift:.3f}')
    if drift > TOL:
        ok = False
    assert 'per_joint_mpjpe' in d, f'{path} missing per_joint_mpjpe'
    assert 'per_joint_mpjdle' in d, f'{path} missing per_joint_mpjdle'
    assert len(d['per_joint_mpjpe']) == 14, (
        f'{path} per_joint_mpjpe entries: {len(d["per_joint_mpjpe"])}')
    assert len(d['per_joint_mpjdle']) == 14, (
        f'{path} per_joint_mpjdle entries: {len(d["per_joint_mpjdle"])}')
    sample = list(d['per_joint_mpjdle'].values())[0]
    assert set(sample.keys()) == {'h','v','d'}, f'{path} per_joint_mpjdle keys: {sample.keys()}'
    print(f'         per_joint_mpjpe entries: {len(d["per_joint_mpjpe"])}; '
          f'per_joint_mpjdle entries: {len(d["per_joint_mpjdle"])}')
if not ok:
    print('SANITY FAILED — do not use these results.', file=sys.stderr)
    sys.exit(2)
print('Tier 1 sanity passed.')
PY

ls -la "${OUT_DIR}/"
