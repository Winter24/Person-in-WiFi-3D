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
  `B0 -> B1 -> B2 -> B4 -> B5`

Current codebase convention:

- `B0` = `WifiInputAdapter(mode='linear')` baseline projection without `BoneLengthLoss`
- `B1+` = `WifiInputAdapter(mode='spectral')` improved tokenizer family
- do not report any run as `B0` if it was trained with `mode='spectral'`
- canonical `configs/wifi/petr_wifi.py` is now paper-faithful for the CVPR baseline recipe:
  `batch=32`, `500 epochs`, `lr=2e-5`, `step=[450]`, `AdamW`, `MSE`-based keypoint losses
- canonical `configs/wifi/petr_wifi.py` also currently follows the requested config-side conventions:
  `meta_keys=[]` and test-time `MultiScaleFlipAug`
- `BoneLossWarmupHook` is now installed globally in `configs/wifi/petr_wifi.py`
  and auto-bypasses when `loss_bone=None`
- default bone warmup policy for PETR-family bone runs:
  `target_weight=2.0`, `warmup_ratio=0.1`, `ramp_ratio=0.1`
- `B5` overrides the warmup target specifically for WiTiDAR:
  `target_weight=1.0`, `warmup_ratio=0.1`, `ramp_ratio=0.1`
- train entrypoint default is fixed to `--seed 42 --deterministic`
- reproducibility env defaults are pinned to:
  `PYTHONHASHSEED=42` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- any `10e/20e/50e` run should be treated as a screening override, not the canonical baseline config
- exploratory side branch:
  `B0_bone = B0 + BoneLengthLoss`
  `B1_bone = B1 + BoneLengthLoss`
  `B2_bone = B2 + BoneLengthLoss`

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

- [ ] `B0` should use `configs/wifi/petr_wifi.py`
- [ ] any improved run (`B1`, `B2`, `B3`, `B4`, `B5`) must use a config whose backbone is `WifiInputAdapter(mode='spectral')`
- [ ] `B0`, `B1`, and `B2` should not use `BoneLengthLoss`
- [ ] `B4` should use `WiTiDARHead` with `loss_bone=None`
- [ ] `B5` should use `WiTiDARHead` with `loss_bone=dict(...)`
- [ ] canonical `B0` config currently uses `AdamW`, `meta_keys=[]`, and test-time `MultiScaleFlipAug`
- [ ] legacy `B0` checkpoints from the original codebase must be evaluated with the current canonical `configs/wifi/petr_wifi.py`, not an old dumped config that still declares `ResNet`

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

Use this pattern for `B0`, `B2`, `B5`, and any model you may put into the efficiency table:

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
- [ ] this is especially important for `B1` and `B4`

### B0

```bash
python tools/train.py configs/wifi/petr_wifi.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/B0

python tools/test.py work_dirs/paper/B0/petr_wifi.py <CKPT_B0> --eval mpjpe \
    --work-dir work_dirs/paper_eval/B0 \
    --metrics-out paper_assets/logs/B0_eval.json

python tools/analysis/benchmark.py work_dirs/paper/B0/petr_wifi.py \
    --checkpoint <CKPT_B0> \
    --device cuda:0 \
    --times 100 \
    --warmup 10 \
    --out paper_assets/logs/B0_benchmark.json

python tools/analysis/append_experiment_log.py \
    --experiment-id B0 \
    --config work_dirs/paper/B0/petr_wifi.py \
    --checkpoint <CKPT_B0> \
    --eval-json paper_assets/logs/B0_eval.json \
    --benchmark-json paper_assets/logs/B0_benchmark.json \
    --csv paper_assets/logs/experiment_log.csv \
    --notes "B0 linear baseline"
```

Explicit shell form for maximum reproducibility:

```bash
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python tools/train.py configs/wifi/petr_wifi.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/B0
```

Legacy B0 checkpoint note:

- [ ] if `<CKPT_B0>` comes from the original pre-refactor codebase, do not use its old dumped config for evaluation
- [ ] use the current canonical config `configs/wifi/petr_wifi.py`
- [ ] make sure `/content/Person-in-WiFi-3D/opera/models/detectors/petr.py` is the patched version that remaps legacy `head.weight/head.bias` and handles WiFi `forward_test()`
- [ ] make sure `/content/Person-in-WiFi-3D/opera/models/dense_heads/petr_head.py` and `/content/Person-in-WiFi-3D/opera/models/dense_heads/wi_tidar_head.py` are the patched versions that resolve `gt_bone_stats.json` from repo root

Legacy B0 eval rerun command:

```bash
python tools/test.py \
    /content/Person-in-WiFi-3D/configs/wifi/petr_wifi.py \
    <CKPT_B0> \
    --eval mpjpe \
    --work-dir work_dirs/paper_eval/B0 \
    --metrics-out paper_assets/logs/B0_eval.json
```

### B1

```bash
python tools/train.py configs/wifi/petr_wifi.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/B1 \
    --cfg-options model.backbone.mode=spectral

python tools/test.py work_dirs/paper/B1/petr_wifi.py <CKPT_B1> --eval mpjpe \
    --work-dir work_dirs/paper_eval/B1 \
    --metrics-out paper_assets/logs/B1_eval.json

python tools/analysis/benchmark.py work_dirs/paper/B1/petr_wifi.py \
    --checkpoint <CKPT_B1> \
    --device cuda:0 \
    --times 100 \
    --warmup 10 \
    --out paper_assets/logs/B1_benchmark.json

python tools/analysis/append_experiment_log.py \
    --experiment-id B1 \
    --config work_dirs/paper/B1/petr_wifi.py \
    --checkpoint <CKPT_B1> \
    --eval-json paper_assets/logs/B1_eval.json \
    --benchmark-json paper_assets/logs/B1_benchmark.json \
    --csv paper_assets/logs/experiment_log.csv \
    --notes "B1 spectral tokenizer only"
```

### B2

```bash
python tools/train.py configs/wifi/petr_wifi_mamba.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/B2

python tools/test.py work_dirs/paper/B2/petr_wifi_mamba.py <CKPT_B2> --eval mpjpe \
    --work-dir work_dirs/paper_eval/B2 \
    --metrics-out paper_assets/logs/B2_eval.json

python tools/analysis/benchmark.py work_dirs/paper/B2/petr_wifi_mamba.py \
    --checkpoint <CKPT_B2> \
    --device cuda:0 \
    --times 100 \
    --warmup 10 \
    --out paper_assets/logs/B2_benchmark.json

python tools/analysis/append_experiment_log.py \
    --experiment-id B2 \
    --config work_dirs/paper/B2/petr_wifi_mamba.py \
    --checkpoint <CKPT_B2> \
    --eval-json paper_assets/logs/B2_eval.json \
    --benchmark-json paper_assets/logs/B2_benchmark.json \
    --csv paper_assets/logs/experiment_log.csv \
    --notes "B2 spectral tokenizer + WiMamba"
```

### Bone Side Branch

Use this branch only if `B0_bone > B0` is promising enough to justify expanding `BoneLengthLoss` to `B1_bone` and `B2_bone`.

`B0_bone`

```bash
python tools/train.py configs/wifi/petr_wifi_bone.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/B0_bone
```

`B1_bone`

```bash
python tools/train.py configs/wifi/petr_wifi_bone.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/B1_bone \
    --cfg-options model.backbone.mode=spectral
```

`B2_bone`

```bash
python tools/train.py configs/wifi/petr_wifi_bone_mamba.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/B2_bone
```

### B3

Current caveat:

- [ ] `B3` is not yet a paper-clean config-only ablation in the current codebase
- [ ] do not freeze `B3` numbers for the paper until flow can be disabled cleanly in both training and inference

Temporary exploratory command only:

```bash
python tools/train.py configs/wifi/wi_tidir_wifi.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/B3_tmp \
    --cfg-options model.bbox_head.loss_flow_weight=0.0 model.bbox_head.loss_bone=None
```

### B4

```bash
python tools/train.py configs/wifi/wi_tidir_wifi.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/B4 \
    --cfg-options model.bbox_head.loss_bone=None

python tools/test.py work_dirs/paper/B4/wi_tidir_wifi.py <CKPT_B4> --eval mpjpe \
    --work-dir work_dirs/paper_eval/B4 \
    --metrics-out paper_assets/logs/B4_eval.json

python tools/analysis/benchmark.py work_dirs/paper/B4/wi_tidir_wifi.py \
    --checkpoint <CKPT_B4> \
    --device cuda:0 \
    --times 100 \
    --warmup 10 \
    --out paper_assets/logs/B4_benchmark.json

python tools/analysis/append_experiment_log.py \
    --experiment-id B4 \
    --config work_dirs/paper/B4/wi_tidir_wifi.py \
    --checkpoint <CKPT_B4> \
    --eval-json paper_assets/logs/B4_eval.json \
    --benchmark-json paper_assets/logs/B4_benchmark.json \
    --csv paper_assets/logs/experiment_log.csv \
    --notes "B4 spectral + WiMamba(6) + draft + flow, no bone"
```

### B5

```bash
python tools/train.py configs/wifi/wi_tidir_wifi.py \
    --seed 42 --deterministic \
    --work-dir work_dirs/paper/B5

python tools/test.py work_dirs/paper/B5/wi_tidir_wifi.py <CKPT_B5> --eval mpjpe \
    --work-dir work_dirs/paper_eval/B5 \
    --metrics-out paper_assets/logs/B5_eval.json

python tools/analysis/benchmark.py work_dirs/paper/B5/wi_tidir_wifi.py \
    --checkpoint <CKPT_B5> \
    --device cuda:0 \
    --times 100 \
    --warmup 10 \
    --out paper_assets/logs/B5_benchmark.json

python tools/analysis/append_experiment_log.py \
    --experiment-id B5 \
    --config work_dirs/paper/B5/wi_tidir_wifi.py \
    --checkpoint <CKPT_B5> \
    --eval-json paper_assets/logs/B5_eval.json \
    --benchmark-json paper_assets/logs/B5_benchmark.json \
    --csv paper_assets/logs/experiment_log.csv \
    --notes "B5 spectral + WiMamba(6) + draft + flow + bone"
```

## Priority Order

### Tier 1

- [ ] `B0` reproduced CVPR baseline with `mode='linear'`
- [ ] `B1` switch `linear -> spectral` only
- [ ] `B2` spectral tokenizer + WiMamba
- [ ] `B4` spectral tokenizer + WiMamba + Draft + Flow
- [ ] `B5` full model with BoneLengthLoss

### Tier 2

- [ ] `B3` Draft-only ablation
- [ ] repeat best models for stability if budget allows

## Ablation Identity Check

Before launching any run, confirm the backbone mode matches the ablation claim:

- [ ] `B0` -> `WifiInputAdapter(mode='linear')`
- [ ] `B1` -> `WifiInputAdapter(mode='spectral')` with baseline encoder/decoder stack
- [ ] `B2/B3/B4/B5` -> `WifiInputAdapter(mode='spectral')`
- [ ] do not reuse an old spectral checkpoint and relabel it as `B0`

## Day 1 Checklist

### Goal

Lock baseline and evaluation pipeline.

### Tasks

- [ ] Verify dataset path and split integrity
- [ ] Verify `gt_bone_stats.json` exists at repo root
- [ ] Reproduce `B0`
- [ ] Confirm `B0` config really uses `WifiInputAdapter(mode='linear')`
- [ ] Run eval command for `B0`
- [ ] Confirm overall `MPJPE`
- [ ] Confirm `1-person / 2-person / 3-person` breakdown export works
- [ ] Confirm `count_1p/2p/3p` are GT denominators
- [ ] Confirm `matched_1p/2p/3p` are exported
- [ ] Run latency and memory benchmark command for `B0`
- [ ] Create or update one experiment log row for `B0`

### End-of-Day Deliverables

- [ ] `B0` final metrics
- [ ] `paper_assets/logs/B0_eval.json`
- [ ] `B0` checkpoint path
- [ ] `paper_assets/logs/B0_benchmark.json`
- [ ] `paper_assets/logs/experiment_log.csv` contains `B0`
- [ ] one short baseline note: stable or unstable

## Day 2 Checklist

### Goal

Measure representation and encoder gains.

### Tasks

- [ ] Confirm `B1` is the first spectral run, not `B0`
- [ ] Run `B1`
- [ ] Run `B2`
- [ ] Export metrics for `B1`
- [ ] Export metrics for `B2`
- [ ] Append or update CSV rows for `B1`
- [ ] Append or update CSV rows for `B2`
- [ ] Export benchmark numbers for `B1`
- [ ] Export benchmark numbers for `B2`
- [ ] Compare `B0 vs B1 vs B2`

### End-of-Day Deliverables

- [ ] tokenizer gain note
- [ ] WiMamba gain note
- [ ] first draft of accuracy-efficiency trend

## Day 3 Checklist

### Goal

Isolate draft and refinement behavior.

### Tasks

- [ ] Run `B3`
- [ ] Run `B4`
- [ ] Export metrics for `B3`
- [ ] Export metrics for `B4`
- [ ] Append or update CSV rows for `B3`
- [ ] Append or update CSV rows for `B4`
- [ ] Compare `B2 vs B3 vs B4`
- [ ] Inspect hard cases manually

### End-of-Day Deliverables

- [ ] note on whether draft head helps
- [ ] note on whether flow refinement helps
- [ ] 3-5 candidate qualitative cases saved

## Day 4 Checklist

### Goal

Run and inspect full model.

### Tasks

- [ ] Run `B5`
- [ ] Export metrics for `B5`
- [ ] Export benchmark for `B5`
- [ ] Append or update CSV row for `B5`
- [ ] Compare `B4 vs B5`
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
- [ ] Confirm best checkpoint for `B0`
- [ ] Confirm best checkpoint for `B2`
- [ ] Confirm best checkpoint for `B5`
- [ ] Confirm `experiment_log.csv` rows for `B0`, `B2`, `B5` are final
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

## Stop Conditions

Pause and reassess if any of these happen:

- [ ] baseline cannot be reproduced consistently
- [ ] `B1` and `B2` both fail to improve anything meaningful
- [ ] full model improves only visually but not numerically
- [ ] WiMamba is slower in practice than Transformer under your setup
- [ ] too many unstable runs consume more than 2 days

## Minimum Publishable Package

If time runs out, make sure these are complete:

- [ ] `B0`
- [ ] `B1`
- [ ] `B2`
- [ ] `B0` is verified as `linear`, not spectral
- [ ] `B0/B1/B2` eval JSON files frozen
- [ ] `B0/B2` benchmark JSON files frozen
- [ ] `experiment_log.csv` updated and checked
- [ ] `B4`
- [ ] `B5`
- [ ] one main comparison table
- [ ] one efficiency table
- [ ] one ablation table
- [ ] one qualitative comparison figure
- [ ] one architecture figure
