# Experiment Daily Checklist

Reference master plan:
[2026-03-31-balanced-arxiv-workshop-paper-plan.md](D:/Resfes_2026/Person-in-WiFi-3D/docs/paper/2026-03-31-balanced-arxiv-workshop-paper-plan.md)

## Usage

Use this file as the daily execution checklist while running experiments for the balanced `arXiv/workshop` paper.

Target scope:

- original CVPR split only
- `1-2 GPU`
- `about 1 week`
- priority ladder:
  `M0 -> M1 -> M2 -> M4 -> M5`

Current codebase convention:

- `M0` = `WifiInputAdapter(mode='linear')` baseline projection without `BoneLengthLoss`
- `M1+` = `WifiInputAdapter(mode='spectral')` improved tokenizer family
- do not report any run as `M0` if it was trained with `mode='spectral'`
- canonical `configs/wifi/petr_wifi.py` is now the current screening baseline recipe:
  `batch=32`, `20 epochs`, `lr=2e-5`, `step=[450]`, `AdamW`, `MSE`-based keypoint losses
- canonical `configs/wifi/petr_wifi.py` also currently follows the requested config-side conventions:
  `meta_keys=[]` and test-time `MultiScaleFlipAug`
- `BoneLossWarmupHook` is now installed globally in `configs/wifi/petr_wifi.py`
  and auto-bypasses when `loss_bone=None`
- default bone warmup policy for PETR-family bone runs:
  `target_weight=2.0`, `warmup_ratio=0.1`, `ramp_ratio=0.1`
- `M5` overrides the warmup target specifically for the Draft + Flow stack:
  `target_weight=1.0`, `warmup_ratio=0.1`, `ramp_ratio=0.1`
- train entrypoint default is fixed to `--seed 42 --deterministic`
- reproducibility env defaults are pinned to:
  `PYTHONHASHSEED=42` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- any `10e/20e/50e` run should be treated as a screening override, not the canonical baseline config
- exploratory side branch:
  `M0_bone = M0 + BoneLengthLoss`
  `M1_bone = M1 + BoneLengthLoss`
  `M2_bone = M2 + BoneLengthLoss`
- main paper ladder:
  `M3 = Spectral + Transformer + Draft + Flow`
  `M4 = Spectral + Mamba + Draft + Flow`
  `M5 = Spectral + Mamba + Draft + Flow + BoneLengthLoss`

## Global Rules

- [ ] Keep the train/test split fixed across all runs.
- [ ] Keep the evaluation script fixed across all runs.
- [ ] Keep the logging format fixed across all runs.
- [ ] Use one naming convention for experiment IDs.
- [ ] Record seed, config path, checkpoint path, and output folder for every run.
- [ ] Export both accuracy metrics and efficiency metrics for every key model.
- [ ] Export evaluation to a fixed JSON file via `--metrics-out`, not only timestamp JSON in `work_dir`.
- [ ] Export benchmark to a fixed JSON file via `tools/analysis/benchmark.py --out`.
- [ ] Append or update one row in `paper_assets/logs/experiment_log.csv` for every completed key run.
- [ ] Do not start extra side experiments until the main ladder is complete.

## Daily Run Template

For every experiment you launch:

- [ ] Confirm config file
- [ ] Confirm work directory
- [ ] Confirm seed
- [ ] Confirm output log path
- [ ] Confirm checkpoint interval
- [ ] Confirm evaluation JSON path
- [ ] Confirm benchmark JSON path if this is a key model
- [ ] Confirm experiment log CSV path

After every experiment finishes:

- [ ] Save final checkpoint path
- [ ] Save best checkpoint path
- [ ] Export main metrics to `paper_assets/logs/<ID>_eval.json`
- [ ] Confirm person-count breakdown exists in JSON: `mpjpe_1p/2p/3p`
- [ ] Confirm GT denominators exist in JSON: `count_1p/2p/3p`
- [ ] Confirm matched counters exist in JSON: `matched_1p/2p/3p`
- [ ] Export benchmark numbers to `paper_assets/logs/<ID>_benchmark.json` if this is a key model
- [ ] Append or update one row in `paper_assets/logs/experiment_log.csv`
- [ ] Mark run status as `done / failed / rerun needed`
- [ ] Write one short note on training stability

## Canonical Artifact Paths

For each key run, keep this naming stable:

- [ ] eval JSON: `paper_assets/logs/<ID>_eval.json`
- [ ] benchmark JSON: `paper_assets/logs/<ID>_benchmark.json`
- [ ] experiment log CSV: `paper_assets/logs/experiment_log.csv`
- [ ] eval work dir: `work_dirs/paper_eval/<ID>/`

## Canonical Commands

### Evaluation Command

Use this pattern for every key run:

```bash
python tools/test.py <config> <ckpt> --eval mpjpe \
    --work-dir work_dirs/paper_eval/<ID> \
    --metrics-out paper_assets/logs/<ID>_eval.json
```

Config note:

- [ ] `M0` should use `configs/wifi/petr_wifi.py`
- [ ] any improved run (`M1`, `M2`, `M3`, `M4`, `M5`) must use a config whose backbone is `WifiInputAdapter(mode='spectral')`
- [ ] `M0`, `M1`, and `M2` should not use `BoneLengthLoss`
- [ ] `M4` should use the Draft + Flow head (`opera.WiTiDARHead`) with `loss_bone=None`
- [ ] `M5` should use the Draft + Flow head (`opera.WiTiDARHead`) with `loss_bone=dict(...)`
- [ ] canonical `M0` config currently uses `AdamW`, `meta_keys=[]`, and test-time `MultiScaleFlipAug`
- [ ] legacy `M0` checkpoints from the original codebase must be evaluated with the current canonical `configs/wifi/petr_wifi.py`, not an old dumped config that still declares `ResNet`

What to check in `<ID>_eval.json`:

- [ ] `mpjpe`
- [ ] `mpjpeh`
- [ ] `mpjpev`
- [ ] `mpjped`
- [ ] `mpjpe_1p`
- [ ] `mpjpe_2p`
- [ ] `mpjpe_3p`
- [ ] `count_1p`
- [ ] `count_2p`
- [ ] `count_3p`
- [ ] `matched_1p`
- [ ] `matched_2p`
- [ ] `matched_3p`
- [ ] `per_joint_mpjpe`
- [ ] `bone_length_error`

Interpretation note:

- [ ] `count_Xp` = GT denominator of the split
- [ ] `matched_Xp` = frames that actually produced a valid matched evaluation result

### Benchmark Command

Use this pattern for `M0`, `M2`, `M5`, and any model you may put into the efficiency table:

```bash
python tools/analysis/benchmark.py <config> \
    --checkpoint <ckpt> \
    --device cuda:0 \
    --times 100 \
    --warmup 10 \
    --out paper_assets/logs/<ID>_benchmark.json
```

What to check in `<ID>_benchmark.json`:

- [ ] `params_m`
- [ ] `trainable_params_m`
- [ ] `flops_g`
- [ ] `latency_ms`
- [ ] `fps`
- [ ] `peak_memory_allocated_mb`
- [ ] `peak_memory_reserved_mb`

### Experiment Log Command

Use this pattern after eval JSON and benchmark JSON are both ready:

```bash
python tools/analysis/append_experiment_log.py \
    --experiment-id <ID> \
    --config <config> \
    --checkpoint <ckpt> \
    --eval-json paper_assets/logs/<ID>_eval.json \
    --benchmark-json paper_assets/logs/<ID>_benchmark.json \
    --csv paper_assets/logs/experiment_log.csv \
    --notes "<short note>"
```

What to check in `experiment_log.csv`:

- [ ] row exists for `<ID>`
- [ ] no duplicate row for the same `experiment_id`
- [ ] `created_at` is preserved on updates
- [ ] values match the latest eval and benchmark artifacts

## Per-Ablation Command Map

Important execution rule:

- [ ] if a run was created with `--cfg-options`, use the dumped config inside `work_dirs/paper/<ID>/` for later eval and benchmark
- [ ] this is especially important for `M1` and `M4`

### M0

```bash
python tools/train.py configs/wifi/petr_wifi.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/M0

python tools/test.py work_dirs/paper/M0/petr_wifi.py <CKPT_M0> --eval mpjpe \
    --work-dir work_dirs/paper_eval/M0 \
    --metrics-out paper_assets/logs/M0_eval.json

python tools/analysis/benchmark.py work_dirs/paper/M0/petr_wifi.py \
    --checkpoint <CKPT_M0> \
    --device cuda:0 \
    --times 100 \
    --warmup 10 \
    --out paper_assets/logs/M0_benchmark.json

python tools/analysis/append_experiment_log.py \
    --experiment-id M0 \
    --config work_dirs/paper/M0/petr_wifi.py \
    --checkpoint <CKPT_M0> \
    --eval-json paper_assets/logs/M0_eval.json \
    --benchmark-json paper_assets/logs/M0_benchmark.json \
    --csv paper_assets/logs/experiment_log.csv \
    --notes "M0 linear baseline"
```

Explicit shell form for maximum reproducibility:

```bash
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python tools/train.py configs/wifi/petr_wifi.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/M0
```

### Full Train Commands With Fixed Environment

Use these exact commands when you want the canonical reproducible shell form.

`M0`

```bash
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python tools/train.py configs/wifi/petr_wifi.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/M0
```

`M1`

```bash
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python tools/train.py configs/wifi/petr_wifi.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/M1 \
    --cfg-options model.backbone.mode=spectral
```

`M2`

```bash
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python tools/train.py configs/wifi/petr_wifi_mamba.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/M2
```

`M0_bone`

```bash
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python tools/train.py configs/wifi/petr_wifi_bone.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/M0_bone
```

`M1_bone`

```bash
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python tools/train.py configs/wifi/petr_wifi_bone.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/M1_bone \
    --cfg-options model.backbone.mode=spectral
```

`M2_bone`

```bash
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python tools/train.py configs/wifi/petr_wifi_bone_mamba.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/M2_bone
```

`M4`

```bash
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python tools/train.py configs/wifi/wi_tidir_wifi.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/M4 \
    --cfg-options model.bbox_head.loss_bone=None
```

`M5`

```bash
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python tools/train.py configs/wifi/wi_tidir_wifi.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/M5
```

`M3`

```bash
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python tools/train.py configs/wifi/wi_tidir_wifi_transformer.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/M3
```

### Strict Reproducibility Run

Use this workflow only for reproducibility verification runs, not for fast screening.

Strict train rerun for `M1`:

```bash
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python tools/train.py configs/wifi/petr_wifi.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/M1_rerun_strict_01 \
    --cfg-options model.backbone.mode=spectral data.workers_per_gpu=0
```

Strict evaluation against an explicit checkpoint:

```bash
python tools/test.py configs/wifi/petr_wifi.py \
    work_dirs/paper/M1_rerun_strict_01/epoch_5.pth \
    --eval mpjpe \
    --cfg-options model.backbone.mode=spectral data.workers_per_gpu=0
```

Strict-run notes:

- [ ] use a fresh `work_dir` for every rerun, never overwrite an older strict run
- [ ] use `epoch_X.pth` directly for comparison, not `latest.pth`
- [ ] set `data.workers_per_gpu=0` when the goal is reproducibility debugging rather than throughput
- [ ] with the current stack, deterministic warnings may still appear from attention/CUDA kernels even under `warn_only=True`
- [ ] the current WiFi PETR configs use `mmcv.MultiheadAttention`; do not attribute those warnings to `MultiScaleDeformableAttention`

Legacy M0 checkpoint note:

- [ ] if `<CKPT_M0>` comes from the original pre-refactor codebase, do not use its old dumped config for evaluation
- [ ] use the current canonical config `configs/wifi/petr_wifi.py`
- [ ] make sure `/content/Person-in-WiFi-3D/opera/models/detectors/petr.py` is the patched version that remaps legacy `head.weight/head.bias` and handles WiFi `forward_test()`
- [ ] make sure `/content/Person-in-WiFi-3D/opera/models/dense_heads/petr_head.py` and `/content/Person-in-WiFi-3D/opera/models/dense_heads/wi_tidar_head.py` are the patched versions that resolve `gt_bone_stats.json` from repo root

Legacy M0 eval rerun command:

```bash
python tools/test.py \
    /content/Person-in-WiFi-3D/configs/wifi/petr_wifi.py \
    <CKPT_M0> \
    --eval mpjpe \
    --work-dir work_dirs/paper_eval/M0 \
    --metrics-out paper_assets/logs/M0_eval.json
```

### M1

```bash
python tools/train.py configs/wifi/petr_wifi.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/M1 \
    --cfg-options model.backbone.mode=spectral

python tools/test.py work_dirs/paper/M1/petr_wifi.py <CKPT_M1> --eval mpjpe \
    --work-dir work_dirs/paper_eval/M1 \
    --metrics-out paper_assets/logs/M1_eval.json

python tools/analysis/benchmark.py work_dirs/paper/M1/petr_wifi.py \
    --checkpoint <CKPT_M1> \
    --device cuda:0 \
    --times 100 \
    --warmup 10 \
    --out paper_assets/logs/M1_benchmark.json

python tools/analysis/append_experiment_log.py \
    --experiment-id M1 \
    --config work_dirs/paper/M1/petr_wifi.py \
    --checkpoint <CKPT_M1> \
    --eval-json paper_assets/logs/M1_eval.json \
    --benchmark-json paper_assets/logs/M1_benchmark.json \
    --csv paper_assets/logs/experiment_log.csv \
    --notes "M1 spectral tokenizer only"
```

### M2

```bash
python tools/train.py configs/wifi/petr_wifi_mamba.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/M2

python tools/test.py work_dirs/paper/M2/petr_wifi_mamba.py <CKPT_M2> --eval mpjpe \
    --work-dir work_dirs/paper_eval/M2 \
    --metrics-out paper_assets/logs/M2_eval.json

python tools/analysis/benchmark.py work_dirs/paper/M2/petr_wifi_mamba.py \
    --checkpoint <CKPT_M2> \
    --device cuda:0 \
    --times 100 \
    --warmup 10 \
    --out paper_assets/logs/M2_benchmark.json

python tools/analysis/append_experiment_log.py \
    --experiment-id M2 \
    --config work_dirs/paper/M2/petr_wifi_mamba.py \
    --checkpoint <CKPT_M2> \
    --eval-json paper_assets/logs/M2_eval.json \
    --benchmark-json paper_assets/logs/M2_benchmark.json \
    --csv paper_assets/logs/experiment_log.csv \
    --notes "M2 spectral tokenizer + WiMamba"
```

### Bone Side Branch

Use this branch only if `M0_bone > M0` is promising enough to justify expanding `BoneLengthLoss` to `M1_bone` and `M2_bone`.

`M0_bone`

```bash
python tools/train.py configs/wifi/petr_wifi_bone.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/M0_bone
```

`M1_bone`

```bash
python tools/train.py configs/wifi/petr_wifi_bone.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/M1_bone \
    --cfg-options model.backbone.mode=spectral
```

`M2_bone`

```bash
python tools/train.py configs/wifi/petr_wifi_bone_mamba.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/M2_bone
```

### M3

```bash
python tools/train.py configs/wifi/wi_tidir_wifi_transformer.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/M3

python tools/test.py work_dirs/paper/M3/wi_tidir_wifi_transformer.py <CKPT_M3> --eval mpjpe \
    --work-dir work_dirs/paper_eval/M3 \
    --metrics-out paper_assets/logs/M3_eval.json

python tools/analysis/benchmark.py work_dirs/paper/M3/wi_tidir_wifi_transformer.py \
    --checkpoint <CKPT_M3> \
    --device cuda:0 \
    --times 100 \
    --warmup 10 \
    --out paper_assets/logs/M3_benchmark.json

python tools/analysis/append_experiment_log.py \
    --experiment-id M3 \
    --config work_dirs/paper/M3/wi_tidir_wifi_transformer.py \
    --checkpoint <CKPT_M3> \
    --eval-json paper_assets/logs/M3_eval.json \
    --benchmark-json paper_assets/logs/M3_benchmark.json \
    --csv paper_assets/logs/experiment_log.csv \
    --notes "M3 spectral + Transformer + draft + flow, no bone"
```

### M4

```bash
python tools/train.py configs/wifi/wi_tidir_wifi.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/M4 \
    --cfg-options model.bbox_head.loss_bone=None

python tools/test.py work_dirs/paper/M4/wi_tidir_wifi.py <CKPT_M4> --eval mpjpe \
    --work-dir work_dirs/paper_eval/M4 \
    --metrics-out paper_assets/logs/M4_eval.json

python tools/analysis/benchmark.py work_dirs/paper/M4/wi_tidir_wifi.py \
    --checkpoint <CKPT_M4> \
    --device cuda:0 \
    --times 100 \
    --warmup 10 \
    --out paper_assets/logs/M4_benchmark.json

python tools/analysis/append_experiment_log.py \
    --experiment-id M4 \
    --config work_dirs/paper/M4/wi_tidir_wifi.py \
    --checkpoint <CKPT_M4> \
    --eval-json paper_assets/logs/M4_eval.json \
    --benchmark-json paper_assets/logs/M4_benchmark.json \
    --csv paper_assets/logs/experiment_log.csv \
    --notes "M4 spectral + WiMamba(4) + draft + flow, no bone"
```

### M5

```bash
python tools/train.py configs/wifi/wi_tidir_wifi.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/M5

python tools/test.py work_dirs/paper/M5/wi_tidir_wifi.py <CKPT_M5> --eval mpjpe \
    --work-dir work_dirs/paper_eval/M5 \
    --metrics-out paper_assets/logs/M5_eval.json

python tools/analysis/benchmark.py work_dirs/paper/M5/wi_tidir_wifi.py \
    --checkpoint <CKPT_M5> \
    --device cuda:0 \
    --times 100 \
    --warmup 10 \
    --out paper_assets/logs/M5_benchmark.json

python tools/analysis/append_experiment_log.py \
    --experiment-id M5 \
    --config work_dirs/paper/M5/wi_tidir_wifi.py \
    --checkpoint <CKPT_M5> \
    --eval-json paper_assets/logs/M5_eval.json \
    --benchmark-json paper_assets/logs/M5_benchmark.json \
    --csv paper_assets/logs/experiment_log.csv \
    --notes "M5 spectral + WiMamba(4) + draft + flow + bone"
```

## Priority Order

### Tier 1

- [ ] `M0` reproduced CVPR baseline with `mode='linear'`
- [ ] `M1` switch `linear -> spectral` only
- [ ] `M2` spectral tokenizer + WiMamba
- [ ] `M4` spectral tokenizer + WiMamba + Draft + Flow
- [ ] `M5` full model with BoneLengthLoss

### Tier 2

- [ ] `M3` Draft-only ablation
- [ ] repeat best models for stability if budget allows

## Ablation Identity Check

Before launching any run, confirm the backbone mode matches the ablation claim:

- [ ] `M0` -> `WifiInputAdapter(mode='linear')`
- [ ] `M1` -> `WifiInputAdapter(mode='spectral')` with baseline encoder/decoder stack
- [ ] `M2/M3/M4/M5` -> `WifiInputAdapter(mode='spectral')`
- [ ] do not reuse an old spectral checkpoint and relabel it as `M0`

## Day 1 Checklist

### Goal

Lock baseline and evaluation pipeline.

### Tasks

- [ ] Verify dataset path and split integrity
- [ ] Verify `gt_bone_stats.json` exists at repo root
- [ ] Reproduce `M0`
- [ ] Confirm `M0` config really uses `WifiInputAdapter(mode='linear')`
- [ ] Run eval command for `M0`
- [ ] Confirm overall `MPJPE`
- [ ] Confirm `1-person / 2-person / 3-person` breakdown export works
- [ ] Confirm `count_1p/2p/3p` are GT denominators
- [ ] Confirm `matched_1p/2p/3p` are exported
- [ ] Run latency and memory benchmark command for `M0`
- [ ] Create or update one experiment log row for `M0`

### End-of-Day Deliverables

- [ ] `M0` final metrics
- [ ] `paper_assets/logs/M0_eval.json`
- [ ] `M0` checkpoint path
- [ ] `paper_assets/logs/M0_benchmark.json`
- [ ] `paper_assets/logs/experiment_log.csv` contains `M0`
- [ ] one short baseline note: stable or unstable

## Day 2 Checklist

### Goal

Measure representation and encoder gains.

### Tasks

- [ ] Confirm `M1` is the first spectral run, not `M0`
- [ ] Run `M1`
- [ ] Run `M2`
- [ ] Export metrics for `M1`
- [ ] Export metrics for `M2`
- [ ] Append or update CSV rows for `M1`
- [ ] Append or update CSV rows for `M2`
- [ ] Export benchmark numbers for `M1`
- [ ] Export benchmark numbers for `M2`
- [ ] Compare `M0 vs M1 vs M2`

### End-of-Day Deliverables

- [ ] tokenizer gain note
- [ ] WiMamba gain note
- [ ] first draft of accuracy-efficiency trend

## Day 3 Checklist

### Goal

Isolate draft and refinement behavior.

### Tasks

- [ ] Run `M3`
- [ ] Run `M4`
- [ ] Export metrics for `M3`
- [ ] Export metrics for `M4`
- [ ] Append or update CSV rows for `M3`
- [ ] Append or update CSV rows for `M4`
- [ ] Compare `M2 vs M3 vs M4`
- [ ] Inspect hard cases manually

### End-of-Day Deliverables

- [ ] note on whether draft head helps
- [ ] note on whether flow refinement helps
- [ ] 3-5 candidate qualitative cases saved

## Day 4 Checklist

### Goal

Run and inspect full model.

### Tasks

- [ ] Run `M5`
- [ ] Export metrics for `M5`
- [ ] Export benchmark for `M5`
- [ ] Append or update CSV row for `M5`
- [ ] Compare `M4 vs M5`
- [ ] Inspect bone-length realism qualitatively
- [ ] Save candidate figures for structure improvement

### End-of-Day Deliverables

- [ ] full model metrics
- [ ] full model benchmark
- [ ] first conclusion on structural loss usefulness

## Day 5 Checklist

### Goal

Stabilize and verify the core claims.

### Tasks

- [ ] Rerun any unstable key model if needed
- [ ] Confirm best checkpoint for `M0`
- [ ] Confirm best checkpoint for `M2`
- [ ] Confirm best checkpoint for `M5`
- [ ] Confirm `experiment_log.csv` rows for `M0`, `M2`, `M5` are final
- [ ] Export clean benchmark summaries
- [ ] Freeze the final numbers to be used in tables

### End-of-Day Deliverables

- [ ] final main-comparison numbers
- [ ] final efficiency numbers
- [ ] final ablation numbers

## Day 6 Checklist

### Goal

Convert experiment outputs into paper assets.

### Tasks

- [ ] Build `Table 1` data
- [ ] Build `Table 2` data
- [ ] Build `Table 3` data
- [ ] Build `Table 4` data
- [ ] Select final qualitative cases
- [ ] Prepare architecture figure draft
- [ ] Prepare trade-off plot
- [ ] Prepare draft-to-refine figure

### End-of-Day Deliverables

- [ ] all main tables in CSV or markdown form
- [ ] all primary figures in draft form

## Day 7 Checklist

### Goal

Write from frozen evidence only.

### Tasks

- [ ] Draft abstract from final numbers
- [ ] Draft introduction from the final claim
- [ ] Draft method section from official full pipeline
- [ ] Draft experiments section from frozen tables
- [ ] Draft limitation paragraph
- [ ] Cross-check every claim against actual metrics

### End-of-Day Deliverables

- [ ] working paper draft
- [ ] figure/table insertion list
- [ ] list of missing polish items only

## Per-Run Record Fields

For each model run, record:

- [ ] run ID
- [ ] ablation ID
- [ ] config path
- [ ] backbone mode (`linear` or `spectral`)
- [ ] eval JSON path
- [ ] benchmark JSON path
- [ ] seed
- [ ] start date/time
- [ ] end date/time
- [ ] GPU used
- [ ] best epoch
- [ ] best checkpoint
- [ ] final checkpoint
- [ ] MPJPE
- [ ] MPJPE 1-person
- [ ] MPJPE 2-person
- [ ] MPJPE 3-person
- [ ] count 1-person
- [ ] count 2-person
- [ ] count 3-person
- [ ] matched 1-person
- [ ] matched 2-person
- [ ] matched 3-person
- [ ] PJDLE(h)
- [ ] PJDLE(v)
- [ ] PJDLE(d)
- [ ] representative bone error summary
- [ ] params
- [ ] latency
- [ ] FPS
- [ ] peak memory
- [ ] notes

## Ablation Result Table

Fill this table as runs finish. Keep one row per frozen result.

| ID | Config | Input Adapter | Encoder | Draft | Flow | Bone | MPJPE | MPJPE 1P | MPJPE 2P | MPJPE 3P | PJDLE(h) | PJDLE(v) | PJDLE(d) | Bone Error | Latency (ms) | FPS | Params (M) | Peak Mem (MB) | Checkpoint | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | `configs/wifi/petr_wifi.py` | Linear | Transformer | PETR | No | No |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| M1 | `configs/wifi/petr_wifi.py + mode=spectral` | Spectral | Transformer | PETR | No | No |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| M2 | `configs/wifi/petr_wifi_mamba.py` | Spectral | Mamba-4 | PETR | No | No |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| M3 | `configs/wifi/wi_tidir_wifi_transformer.py` | Spectral | Transformer-6 | Draft + Flow | Yes | No |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| M0_bone | `configs/wifi/petr_wifi_bone.py` | Linear | Transformer | PETR | No | Yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| M1_bone | `configs/wifi/petr_wifi_bone.py + mode=spectral` | Spectral | Transformer | PETR | No | Yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| M2_bone | `configs/wifi/petr_wifi_bone_mamba.py` | Spectral | Mamba-4 | PETR | No | Yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| M4 | `configs/wifi/wi_tidir_wifi.py + loss_bone=None` | Spectral | Mamba-4 | Draft + Flow | Yes | No |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| M5 | `configs/wifi/wi_tidir_wifi.py` | Spectral | Mamba-4 | Draft + Flow | Yes | Yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Stop Conditions

Pause and reassess if any of these happen:

- [ ] baseline cannot be reproduced consistently
- [ ] `M1` and `M2` both fail to improve anything meaningful
- [ ] full model improves only visually but not numerically
- [ ] WiMamba is slower in practice than Transformer under your setup
- [ ] too many unstable runs consume more than 2 days

## Minimum Publishable Package

If time runs out, make sure these are complete:

- [ ] `M0`
- [ ] `M1`
- [ ] `M2`
- [ ] `M0` is verified as `linear`, not spectral
- [ ] `M0/M1/M2` eval JSON files frozen
- [ ] `M0/M2` benchmark JSON files frozen
- [ ] `experiment_log.csv` updated and checked
- [ ] `M4`
- [ ] `M5`
- [ ] one main comparison table
- [ ] one efficiency table
- [ ] one ablation table
- [ ] one qualitative comparison figure
- [ ] one architecture figure

