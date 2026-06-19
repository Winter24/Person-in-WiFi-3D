# ckey.vn Rev2 Eval Runbook

Use this runbook instead of copy-pasting an inline L1 patch. The expected path
on ckey.vn is `/root/Person-in-WiFi-3D`.

## 1. Sync Code

```bash
cd /root/Person-in-WiFi-3D
git status
git pull
git log -1 --oneline
```

Do not continue if `git pull` reports conflicts.

## 2. Verify L1 Patch

```bash
python3 scripts/verify_l1_per_joint_mpjdle.py
```

Expected output:

```text
OK: opera/datasets/wifi_pose.py contains the complete L1 per_joint_mpjdle patch.
```

This check is static and does not require a full GPU eval.

## 3. Run Tier 1 Eval

```bash
mkdir -p paper_assets/logs/rev2

CKPT_M0=work_dirs/full_alation_20e/M0/latest.pth \
CKPT_M9=work_dirs/full_alation_20e/M9/latest.pth \
OUT_DIR=paper_assets/logs/rev2 \
bash scripts/gpu_run_tier1.sh
```

The runner writes:

- `paper_assets/logs/rev2/M9_RF2_eval.json`
- `paper_assets/logs/rev2/M0_eval.json`
- `paper_assets/logs/rev2/G1_M9_RF2_eval.log`
- `paper_assets/logs/rev2/G2_M0_eval.log`

Expected sanity:

- `M9_RF2_eval.json`: MPJPE about `165.487` mm.
- `M0_eval.json`: MPJPE about `172.540` mm.
- Both JSON files have `per_joint_mpjpe` with 14 entries.
- Both JSON files have `per_joint_mpjdle` with 14 entries, each with `h/v/d`.

## 4. Optional Manual JSON Check

```bash
python3 scripts/verify_l1_per_joint_mpjdle.py \
  --eval-json paper_assets/logs/rev2/M9_RF2_eval.json

python3 scripts/verify_l1_per_joint_mpjdle.py \
  --eval-json paper_assets/logs/rev2/M0_eval.json
```

## 5. Bring Results Back

```bash
tar -czf /tmp/rev2_tier1_results.tgz paper_assets/logs/rev2/
```

Download `/tmp/rev2_tier1_results.tgz` to local, then generate
`tables/per_joint_table.tex` with `scripts/extract_per_joint_mpjpe.py`.
