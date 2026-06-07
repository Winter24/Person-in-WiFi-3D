# Scripts Usage Guide

Run all commands from the repository root:

```bash
cd /content/Person-in-WiFi-3D
```

Most experiment launchers support fixed-seed training with:

```bash
SEED=42
PYTHONHASHSEED_VALUE=42
CUBLAS_WORKSPACE_CONFIG_VALUE=:4096:8
```

On Colab/Linux, keep Triton CUDA lookup auto-fix enabled unless you know the environment does not need it:

```bash
AUTO_FIX_TRITON_LIBCUDA=1
```

## Common Modes

The newer experiment runners use these modes:

| Mode | Meaning |
| --- | --- |
| `all` | Train, evaluate, benchmark, append CSV, then extract section latency when supported. |
| `train` | Train selected run ids only. |
| `postprocess` | Evaluate and benchmark existing checkpoints, then append CSV. |
| `benchmark` | Benchmark existing checkpoints only. |
| `extract` | Rebuild CSV rows from existing eval/benchmark JSON files. |
| `smoke` | Short benchmark sanity check, usually `times=5`, `warmup=2`. |
| `help` | Print script usage. |

Common environment overrides:

```bash
ONLY_RUN_IDS="RUN_A RUN_B"
TRAIN_GPU=0
TEST_GPU=0
BENCHMARK_DEVICE=cuda:0
BENCHMARK_TIMES=100
BENCHMARK_WARMUP=10
DRY_RUN=1
```

Checkpoint override convention:

```bash
RUN_ID_CKPT="/path/to/checkpoint.pth"
```

Example:

```bash
ONLY_RUN_IDS="C_RF1 D_RF8" \
C_RF1_CKPT="/content/Person-in-WiFi-3D/work_dirs/flow_numstep_ablation/C_RF1/latest.pth" \
D_RF8_CKPT="/content/Person-in-WiFi-3D/work_dirs/flow_numstep_ablation/D_RF8/latest.pth" \
bash scripts/run_flow_numstep_ablation.sh postprocess
```

If no `_CKPT` override is provided, runners usually search:

```text
$WORK_ROOT/$RUN_ID/latest.pth
$WORK_ROOT/$RUN_ID/epoch_*.pth newest by version sort
```

## `auto_fix_triton_libcuda.sh`

Purpose: fix the common Colab/Triton error where `triton.common.build.libcuda_dirs()` cannot find `libcuda.so`.

Use manually:

```bash
bash scripts/auto_fix_triton_libcuda.sh
```

Most new runners call this automatically when:

```bash
AUTO_FIX_TRITON_LIBCUDA=1
```

Disable only if the environment is already healthy:

```bash
AUTO_FIX_TRITON_LIBCUDA=0 bash scripts/run_mamba2_encoder_ablation.sh smoke
```

## `run_mamba2_encoder_ablation.sh`

Purpose: run the full Mamba2 encoder ablation ladder.

Default run ids:

```text
M2D M2F M2FGP M2C M2CP M2CPA
```

Meaning:

| Run ID | Summary |
| --- | --- |
| `M2D` | Mamba2 drop-in, factorized temporal/spatial layout. |
| `M2F` | Flattened single-route Mamba2, `time_major`, no position. |
| `M2FGP` | Flattened single-route Mamba2 with zero-init gated CSI position. |
| `M2C` | Two-route cross-scan, `time_major + serpentine`. |
| `M2CP` | Cross-scan with plain separable CSI position. |
| `M2CPA` | Cross-scan with position and final lightweight attention. |

Run all:

```bash
bash scripts/run_mamba2_encoder_ablation.sh all
```

Run only M2FGP:

```bash
ONLY_RUN_IDS="M2FGP" bash scripts/run_mamba2_encoder_ablation.sh all
```

Evaluate/benchmark trained checkpoints only:

```bash
bash scripts/run_mamba2_encoder_ablation.sh postprocess
```

Use explicit checkpoint:

```bash
ONLY_RUN_IDS="M2FGP" \
M2FGP_CKPT="/content/Person-in-WiFi-3D/work_dirs/mamba2_encoder_ablation/M2FGP/latest.pth" \
bash scripts/run_mamba2_encoder_ablation.sh postprocess
```

Default outputs:

```text
work_dirs/mamba2_encoder_ablation/
paper_assets/logs/mamba2_encoder_ablation/
paper_assets/logs/mamba2_encoder_ablation/experiment_log.csv
paper_assets/logs/mamba2_encoder_ablation/section_latency_log.csv
```

## `run_mamba2_candidate_20e.sh`

Purpose: wrapper around `run_mamba2_encoder_ablation.sh` for the long-run candidate set.

Default run ids:

```text
M2F M2C M2FGP
```

Run all candidates for 20 epochs:

```bash
bash scripts/run_mamba2_candidate_20e.sh all
```

Train only:

```bash
bash scripts/run_mamba2_candidate_20e.sh train
```

Run only M2FGP:

```bash
ONLY_RUN_IDS="M2FGP" bash scripts/run_mamba2_candidate_20e.sh all
```

Default outputs are isolated from short ablations:

```text
work_dirs/mamba2_encoder_ablation_20e/
work_dirs/mamba2_encoder_eval_20e/
paper_assets/logs/mamba2_encoder_ablation_20e/
```

## `run_flow_numstep_ablation.sh`

Purpose: test whether rectified flow refinement is useful, without the residual-MLP baseline.

Default run ids:

```text
A_DRAFT C_RF1 D_RF4 D_RF8 D_RF10
```

Meaning:

| Run ID | Summary |
| --- | --- |
| `A_DRAFT` | Draft pose only, no flow refinement. |
| `C_RF1` | Rectified flow, Euler inference `num_steps=1`. |
| `D_RF4` | Rectified flow, Euler inference `num_steps=4`. |
| `D_RF8` | Rectified flow, Euler inference `num_steps=8`. |
| `D_RF10` | Rectified flow, Euler inference `num_steps=10`. |

Run all:

```bash
bash scripts/run_flow_numstep_ablation.sh all
```

Evaluate/benchmark existing checkpoints:

```bash
bash scripts/run_flow_numstep_ablation.sh postprocess
```

Use explicit checkpoints:

```bash
ONLY_RUN_IDS="A_DRAFT C_RF1 D_RF8" \
A_DRAFT_CKPT="/content/Person-in-WiFi-3D/work_dirs/flow_numstep_ablation/A_DRAFT/latest.pth" \
C_RF1_CKPT="/content/Person-in-WiFi-3D/work_dirs/flow_numstep_ablation/C_RF1/latest.pth" \
D_RF8_CKPT="/content/Person-in-WiFi-3D/work_dirs/flow_numstep_ablation/D_RF8/latest.pth" \
bash scripts/run_flow_numstep_ablation.sh postprocess
```

Default outputs:

```text
work_dirs/flow_numstep_ablation/
work_dirs/flow_numstep_eval/
paper_assets/logs/flow_numstep_ablation/
```

## `run_paper_m0_m4_fixed_seed.sh`

Purpose: run the fixed-seed paper ladder for M0-M4.

Default run ids:

```text
M0 M1 M2 M3 M4
```

Meaning:

| Run ID | Summary |
| --- | --- |
| `M0` | Linear input adapter + Transformer + DETR regression. |
| `M1` | Spectral input adapter + Transformer + DETR regression. |
| `M2` | Spectral input adapter + Mamba + DETR regression. |
| `M3` | Spectral input adapter + Transformer + draft/flow decoder. |
| `M4` | Spectral input adapter + Mamba + draft/flow decoder, no bone loss. |

Run all:

```bash
bash scripts/run_paper_m0_m4_fixed_seed.sh all
```

Train only:

```bash
bash scripts/run_paper_m0_m4_fixed_seed.sh train
```

Postprocess existing checkpoints:

```bash
bash scripts/run_paper_m0_m4_fixed_seed.sh postprocess
```

Run subset:

```bash
ONLY_RUN_IDS="M2 M4" bash scripts/run_paper_m0_m4_fixed_seed.sh all
```

Use explicit checkpoint:

```bash
ONLY_RUN_IDS="M4" \
M4_CKPT="/content/Person-in-WiFi-3D/work_dirs/paper/M4/latest.pth" \
bash scripts/run_paper_m0_m4_fixed_seed.sh postprocess
```

Default outputs:

```text
work_dirs/paper/
work_dirs/paper_eval/
paper_assets/logs/
```

## `run_wimamba_backbone_ablation.sh`

Purpose: run the fixed-seed full WiMamba backbone ablation ladder inside
`opera.WiTiDARHead` from `opera/models/backbones/` without swapping files by
hand. This is the WiTiDAR/M4-style ablation, not the PETRHead encoder ablation.

Default run ids:

```text
M3 M1FCT M1V2 M1V3 M1FLAT M2FLAT M2CSI M2SA M2SAP M2D M2C M2CP M2CPA
```

Modes:

| Mode | What it does |
| --- | --- |
| `all` | Train each selected run, then evaluate, benchmark, append CSV, and export section latency CSV. |
| `train` | Train only. It writes checkpoints/logs under `WORK_ROOT`, but does not evaluate. |
| `postprocess` | Evaluate + benchmark existing checkpoints, append CSV, and export section latency CSV. |
| `benchmark` | Benchmark existing checkpoints only, then export section latency CSV. |
| `extract` | Rebuild CSV rows from existing eval/benchmark JSON files. It does not train/evaluate/benchmark. |
| `smoke` | Run a short benchmark with `times=5`, `warmup=2`; useful before long jobs. |
| `help` | Print script help only; it does not validate configs or patch Triton. |

`AUTO_RESUME` defaults to `0` on purpose. Enable `AUTO_RESUME=1` only when
continuing the exact same config in the same work directory. If you change
backbone/head/config, optimizer states from an older checkpoint can fail with
`loaded state dict has a different number of parameter groups`.

Meaning:

| Run ID | Config | Summary |
| --- | --- | --- |
| `M3` | `configs/wifi/wi_tidir_wifi_transformer.py` | WiTiDARHead baseline with Transformer encoder. |
| `M1FCT` | `configs/wifi/wi_tidir_wifi.py` | Current Mamba1 factorized temporal + bidirectional spatial scan. |
| `M1V2` | `configs/wifi/wi_tidir_wifi_mamba_v2.py` | Mamba1 temporal scan + Conv1d antenna mixer. |
| `M1V3` | `configs/wifi/wi_tidir_wifi_mamba_v3.py` | Mamba1 temporal scan + Linear antenna mixer. |
| `M1FLAT` | `configs/wifi/wi_tidir_wifi_mamba1_flatten.py` | Mamba1 flattened single-route `L=180` scan. |
| `M2FLAT` | `configs/wifi/wi_tidir_wifi_mamba2_flatten.py` | Mamba2 flattened single-route `L=180` wrapper. |
| `M2D` | `configs/wifi/wi_tidir_wifi_mamba2_dropin.py` | Mamba2 drop-in factorized temporal/spatial layout. |
| `M2CSI` | `configs/wifi/wi_tidir_wifi_mamba2_flattened.py` | Mamba2 CSI flattened single-route via `WiMamba2CSIEncoder`. |
| `M2SA` | `configs/wifi/wi_tidir_wifi_mamba2_spatial_attn.py` | Mamba2 flattened single-route with 9x9 spatial attention pre-mixer. |
| `M2SAP` | `configs/wifi/wi_tidir_wifi_mamba2_spatial_attn_pos.py` | `M2SA` plus gated CSI positional embedding. |
| `M2C` | `configs/wifi/wi_tidir_wifi_mamba2_crossscan.py` | Mamba2 two-route cross-scan. |
| `M2CP` | `configs/wifi/wi_tidir_wifi_mamba2_crossscan_pos.py` | Mamba2 cross-scan + CSI positional embedding. |
| `M2CPA` | `configs/wifi/wi_tidir_wifi_mamba2_crossscan_pos_attn.py` | Mamba2 cross-scan + position + final attention. |

Run all selected backbones:

```bash
bash scripts/run_wimamba_backbone_ablation.sh all
```

Run a fast 5-epoch candidate subset:

```bash
ONLY_RUN_IDS="M3 M2FLAT M2CSI M1FLAT M1V2 M2SA M2SAP" MAX_EPOCHS=5 \
bash scripts/run_wimamba_backbone_ablation.sh all
```

Train only:

```bash
ONLY_RUN_IDS="M3 M2FLAT M2CSI M1FLAT M1V2 M2SA M2SAP" \
bash scripts/run_wimamba_backbone_ablation.sh train
```

Smoke-test model construction/benchmark path:

```bash
ONLY_RUN_IDS="M3 M2FLAT M2CSI M2SA M2SAP" \
bash scripts/run_wimamba_backbone_ablation.sh smoke
```

Postprocess existing checkpoints:

```bash
bash scripts/run_wimamba_backbone_ablation.sh postprocess
```

Checkpoint lookup order for `postprocess`, `benchmark`, and `extract`:

```text
1. Per-run environment override, for example M1V3_CKPT=/path/to/checkpoint.pth.
2. $WORK_ROOT/$RUN_ID/latest.pth.
3. Highest version-sorted $WORK_ROOT/$RUN_ID/epoch_*.pth.
```

Eval/benchmark config lookup is similarly conservative: the runner uses the
current mapped config, or `$WORK_ROOT/$RUN_ID/$(basename config)` if that exact
copy exists. It does not scan arbitrary old `.py` files in the work directory,
which prevents stale PETR configs from being reused after switching to WiTiDAR.

Use explicit checkpoints:

```bash
ONLY_RUN_IDS="M1V3 M2FLAT" \
M1V3_CKPT="/content/Person-in-WiFi-3D/work_dirs/wimamba_backbone_ablation/M1V3/latest.pth" \
M2FLAT_CKPT="/content/Person-in-WiFi-3D/work_dirs/wimamba_backbone_ablation/M2FLAT/latest.pth" \
bash scripts/run_wimamba_backbone_ablation.sh postprocess
```

Dry run before launching long jobs:

```bash
DRY_RUN=1 ONLY_RUN_IDS="M1V3 M1FLAT M2FLAT" MAX_EPOCHS=5 \
bash scripts/run_wimamba_backbone_ablation.sh train
```

Useful overrides:

```bash
WORK_ROOT="work_dirs/wimamba_backbone_ablation"
EVAL_ROOT="work_dirs/wimamba_backbone_eval"
LOG_ROOT="paper_assets/logs/wimamba_backbone_ablation"
MAX_EPOCHS=20
SEED=42
TRAIN_GPU=0
TEST_GPU=0
PROFILE_SECTIONS=1
AUTO_FIX_TRITON_LIBCUDA=1
AUTO_RESUME=0
```

Default outputs:

```text
work_dirs/wimamba_backbone_ablation/
work_dirs/wimamba_backbone_eval/
paper_assets/logs/wimamba_backbone_ablation/
paper_assets/logs/wimamba_backbone_ablation/experiment_log.csv
paper_assets/logs/wimamba_backbone_ablation/section_latency_log.csv
```

## `run_paper_postprocess.sh`

Purpose: legacy postprocess helper for paper checkpoints M0-M5 and variant directories.

Modes:

```bash
bash scripts/run_paper_postprocess.sh default
bash scripts/run_paper_postprocess.sh root-paper
bash scripts/run_paper_postprocess.sh colab
```

Default mode processes:

```text
M0 M1 M2 M3 M4 M5
```

It expects run directories under:

```text
work_dirs/paper/$RUN_ID
```

or manual checkpoint variables inside the script. Prefer newer runners when possible because they support environment-based `_CKPT` overrides more cleanly.

## `run_old_paper_folder_ablation.sh`

Purpose: evaluate and benchmark an existing paper-style checkpoint folder tree by
reading each run's dumped config and checkpoint directly from
`OLD_PAPER_ROOT/$RUN_ID`. Use this for old-version folders that mix canonical
paper runs and backbone variants, for example:

```text
M0 M1 M1FLAT M1V2 M2 M2CSI M2FLAT M3 M4 M5
```

Run full postprocess and write logs into `paper_assets`:

```bash
OLD_PAPER_ROOT="/root/Person-in-WiFi-3D/work_dirs/old_ver/root/Person-in-WiFi-3D/work_dirs/paper" \
LOG_ROOT="paper_assets/logs/old_paper_folder_ablation" \
EVAL_ROOT="work_dirs/old_paper_folder_eval" \
ONLY_RUN_IDS="M0 M1 M1FLAT M1V2 M2 M2CSI M2FLAT M3 M4 M5" \
bash scripts/run_old_paper_folder_ablation.sh postprocess
```

Benchmark only:

```bash
OLD_PAPER_ROOT="/root/Person-in-WiFi-3D/work_dirs/old_ver/root/Person-in-WiFi-3D/work_dirs/paper" \
LOG_ROOT="paper_assets/logs/old_paper_folder_ablation" \
ONLY_RUN_IDS="M0 M1 M1FLAT M1V2 M2 M2CSI M2FLAT M3 M4 M5" \
bash scripts/run_old_paper_folder_ablation.sh benchmark
```

Outputs:

```text
paper_assets/logs/old_paper_folder_ablation/experiment_log.csv
paper_assets/logs/old_paper_folder_ablation/section_latency_log.csv
paper_assets/logs/old_paper_folder_ablation/*_eval.json
paper_assets/logs/old_paper_folder_ablation/*_benchmark.json
```

The script does not use hard-coded config mapping. For each run, it uses the
first sorted `*.py` config dump and `latest.pth`, or the highest `epoch_*.pth`
if `latest.pth` does not exist.

## `run_train_5gpu.sh`

Purpose: train multiple paper runs in parallel on separate GPUs.

Default run ids:

```text
M0 M1 M2 M4 M5
```

Run with automatic GPU discovery:

```bash
bash scripts/run_train_5gpu.sh
```

Specify exact GPUs:

```bash
GPU_IDS="0 1 2 3 4" bash scripts/run_train_5gpu.sh
```

Add optional run ids:

```bash
EXTRA_RUN_IDS="M5_linear" GPU_IDS="0 1 2 3 4 5" bash scripts/run_train_5gpu.sh
```

Dry run:

```bash
DRY_RUN=1 GPU_IDS="0 1 2 3 4" bash scripts/run_train_5gpu.sh
```

Logs:

```text
work_dirs/paper_launch_logs/
```

## `run_colab_variant_batch.sh`

Purpose: legacy Colab postprocess for Mamba variant checkpoints by temporarily swapping `wimamba_v1.py`.

Default run names:

```text
M1_v1 M1_v3 M5_v1 M5_v2 M5_v3
```

Run all defaults:

```bash
bash scripts/run_colab_variant_batch.sh
```

Run subset:

```bash
bash scripts/run_colab_variant_batch.sh M1_v1 M5_v2
```

Useful overrides:

```bash
PAPER_DIR="/content/drive/MyDrive/RESFES2026/Test" \
OUTPUT_DIR="/content/drive/MyDrive/RESFES2026/Test" \
bash scripts/run_colab_variant_batch.sh
```

This script modifies backbone files during execution and restores them on exit. Avoid interrupting the environment aggressively while it is copying files.

## Recommended Current Workflow

Mamba2 encoder long candidates:

```bash
bash scripts/run_mamba2_candidate_20e.sh all
```

Flow num-step ablation:

```bash
bash scripts/run_flow_numstep_ablation.sh all
```

Only postprocess existing flow checkpoints:

```bash
bash scripts/run_flow_numstep_ablation.sh postprocess
```

Quick sanity checks before a long run:

```bash
DRY_RUN=1 bash scripts/run_mamba2_candidate_20e.sh train
DRY_RUN=1 bash scripts/run_flow_numstep_ablation.sh train
```
