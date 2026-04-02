# Balanced ArXiv Workshop Paper Plan

## Goal

Write a direct follow-up paper to *Person-in-WiFi 3D* that keeps the original CVPR split and tells one clear story:

`Raw CSI -> Linear Projection Baseline or Spectral Tokenizer -> WiMamba Encoder -> Cross-Attention + Draft MLP -> Rectified Flow Matching -> 3D Pose under BoneLengthLoss Constraints`

The target is a strong `arXiv/workshop first` paper with a balanced claim:

- improve 3D pose quality over the CVPR baseline
- improve the accuracy-efficiency trade-off
- show more realistic, anatomically plausible skeletons

## Target Positioning

### Venue Strategy

- first target: `arXiv + workshop`
- evaluation scope: `official Person-in-WiFi 3D split only`
- experimental budget: `1-2 GPU, about 1 week`

### Recommended Title Direction

- `Wi-FlowMamba: Spectral-Aware and Efficient Multi-Person 3D Pose Estimation from WiFi`
- `SpectraMambaFlow for Multi-Person 3D WiFi Pose Estimation`
- `Beyond Person-in-WiFi 3D: Spectral Tokenization, Linear Sequence Modeling, and Flow-Based Pose Refinement`

### One-Sentence Thesis

We improve Person-in-WiFi 3D by replacing its plain linear CSI projection with a WiFi-native spectral tokenizer, then adding a linear-complexity Mamba encoder and a draft-to-refine flow-based pose decoder, yielding a better balance between accuracy, efficiency, and skeletal realism on the original CVPR split.

## Reference Basis

This plan is grounded in:

- [2024CVPR_Person_in_WiFi_3D.pdf](D:/Resfes_2026/Person-in-WiFi-3D/Reference_src/2024CVPR_Person_in_WiFi_3D.pdf)
- [Towards Robust and Realistic Human Pose Estimation via WiFi Signals.pdf](D:/Resfes_2026/Person-in-WiFi-3D/Reference_src/Towards%20Robust%20and%20Realistic%20Human%20Pose%20Estimation%20via%20WiFi%20Signals.pdf)

Use the first paper as the baseline and primary comparison anchor. Use the second paper mainly as motivation for structural fidelity and realism, not as the main benchmark target.

## Paper Narrative

### Baseline Limitation Framing

The paper should motivate three gaps in the original CVPR baseline:

1. `Signal modeling gap`
   Raw CSI contains temporal motion and spectral Doppler cues, but the original plain linear projection is not explicitly designed to capture them.

2. `Efficiency gap`
   Transformer encoder complexity scales quadratically with sequence length, which is suboptimal for long CSI token sequences.

3. `Structural fidelity gap`
   Direct coordinate regression can produce anatomically inconsistent or unstable multi-person 3D poses, especially in extremities and crowded cases.

### Proposed Resolution

Map each gap to one corresponding module:

1. `Spectral Tokenizer`
   - local temporal branch via `1D Conv`
   - global spectral branch via `2D FFT`
   - target: stronger CSI representation than the baseline linear projector

2. `WiMamba Encoder`
   - linear-complexity sequence modeling
   - target: better speed-memory trade-off

3. `Draft-to-Refine Head`
   - cross-attention + draft MLP for coarse pose hypotheses
   - rectified flow matching for ODE-style correction
   - bone-length supervision for anatomical realism

### Core Contributions

Recommended contribution bullets:

1. A frequency-aware CSI tokenization module that upgrades the baseline linear CSI projection with local temporal dynamics and global spectral cues.
2. A linear-complexity WiMamba encoder for efficient long-sequence CSI modeling.
3. A draft-to-refine 3D pose pipeline using cross-attention and rectified flow matching.
4. Anatomy-aware regularization with BoneLengthLoss to improve structural plausibility.
5. A full benchmark on the original Person-in-WiFi 3D split, including accuracy, efficiency, and structural realism analyses.

## Model Definition For The Paper

### Official Full Model

The official full pipeline should be described as:

1. `Raw CSI`
2. `Linear Projection or Spectral Tokenizer`
3. `WiMamba Encoder`
4. `Cross-Attention Query Decoder`
5. `Draft MLP Pose Regressor`
6. `Rectified Flow Refiner`
7. `BoneLengthLoss Structural Supervision`

### Paper Naming Convention

Use the following naming consistently in the manuscript:

- `Baseline`: original CVPR-style Person-in-WiFi 3D
- `Linear Projection`: baseline `WifiInputAdapter(mode='linear')`
- `Tokenizer`: Spectral Tokenizer
- `Encoder`: WiMamba Encoder
- `Draft`: cross-attention + draft MLP
- `Refine`: Rectified Flow Matching
- `Structure`: BoneLengthLoss

Current codebase mapping:

- `B0` uses `WifiInputAdapter(mode='linear')` without `BoneLengthLoss`
- `B1+` use `WifiInputAdapter(mode='spectral')`
- canonical `configs/wifi/petr_wifi.py` now follows the paper-faithful baseline recipe:
  `batch=32`, `500 epochs`, `lr=2e-5`, `step=[450]`, `AdamW`, `MSE` regression losses
- canonical `configs/wifi/petr_wifi.py` also keeps the currently requested config-side conventions:
  `meta_keys=[]` and test-time `MultiScaleFlipAug`
- shorter `10e/20e/50e` runs are screening overrides for proposal-speed iteration, not the canonical CVPR-style baseline config
- exploratory side branch:
  `B0_bone = B0 + BoneLengthLoss`
  `B1_bone = B1 + BoneLengthLoss`
  `B2_bone = B2 + BoneLengthLoss`

## Experimental Protocol

### Dataset and Split

Only use the original `Person-in-WiFi 3D` train/test split for this first paper.

Keep:

- same train split
- same test split
- same evaluation logic
- same person-count grouping

Do not add cross-domain or cross-environment experiments in this first version.

### Main Metrics

The main paper should report:

- `MPJPE`
- `MPJPE (1-person)`
- `MPJPE (2-person)`
- `MPJPE (3-person)`
- `PJDLE(h)`
- `PJDLE(v)`
- `PJDLE(d)`
- `Per-joint MPJPE`
- `Mean bone length error`

### Efficiency Metrics

For the balanced claim, also report:

- `parameter count`
- `latency per sample`
- `throughput / FPS`
- `peak GPU memory`
- `optional FLOPs or MACs`

### Structural Metrics

For realism analysis, report:

- `mean bone length error`
- `bone-length consistency error`
- `qualitative structural failure rate` if you define a simple criterion

## Required Baselines And Ablations

The paper should not compare only `baseline vs full model`. It needs a clean ladder.

### Main Ablation Ladder

| ID | Input Adapter | Encoder | Draft | Flow Refine | BoneLengthLoss | Purpose |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | Linear Projection | Transformer | PETR original | No | No | Reproduced CVPR baseline |
| B1 | Spectral Tokenizer | Transformer | PETR original | No | No | Isolate tokenizer gain over linear B0 |
| B2 | Spectral Tokenizer | WiMamba (6 layers) | PETR original or PETR-compatible decoder | No | No | Isolate encoder gain |
| B3 | Spectral Tokenizer | WiMamba | Draft head | No | No | Isolate draft decoding |
| B4 | Spectral Tokenizer | WiMamba (6 layers) | Draft head | Yes | No | Isolate flow refinement |
| B5 | Spectral Tokenizer | WiMamba (6 layers) | Draft head | Yes | Yes | Full model with BoneLengthLoss |

### Minimum Viable Ladder If Time Is Tight

If the week gets compressed, keep at least:

- `B0`
- `B1`
- `B2`
- `B4`
- `B5`

This is the minimum set that still tells a convincing story.

### Current Runnable Command Strategy

To keep the ablations reproducible in the current codebase:

- use `work_dirs/paper/<ID>/` as the canonical run folder for each ablation
- for runs launched with `--cfg-options`, use the dumped config inside that work dir for later eval and benchmark
- treat `B3` as exploratory only for now; it is not yet a paper-clean config-only ablation because flow is not fully toggleable at inference

Practical mapping:

- `B0`: train from `configs/wifi/petr_wifi.py`
- `B1`: train from `configs/wifi/petr_wifi.py` with `--cfg-options model.backbone.mode=spectral`
- `B2`: train from `configs/wifi/petr_wifi_mamba.py`
- `B4`: train from `configs/wifi/wi_tidir_wifi.py` with `--cfg-options model.bbox_head.loss_bone=None`
- `B5`: train from `configs/wifi/wi_tidir_wifi.py`
- exploratory side branch:
  `B0_bone`: train from `configs/wifi/petr_wifi_bone.py`
  `B1_bone`: train from `configs/wifi/petr_wifi_bone.py` with `--cfg-options model.backbone.mode=spectral`
  `B2_bone`: train from `configs/wifi/petr_wifi_bone_mamba.py`

Legacy B0 evaluation policy:

- if a `B0` checkpoint was trained in the original pre-refactor codebase, still evaluate it with the current canonical `configs/wifi/petr_wifi.py`
- do not evaluate a legacy `B0` checkpoint with an old dumped config that still declares `ResNet`, because that config no longer reflects the actual linear-projection runtime path used by the old PETR WiFi baseline
- the patched current codebase is the source of truth for legacy `B0` evaluation because it remaps old `head.weight/head.bias` checkpoints onto `backbone.linear_proj.*`

## Hypotheses To Validate

### H1: Spectral Tokenizer Helps

Expected signs:

- lower MPJPE than the linear-projection baseline `B0`
- clearer gains on wrists, ankles, and depth-related errors
- better qualitative stability in hard multi-person cases

### H2: WiMamba Improves Accuracy-Efficiency Trade-Off

Expected signs:

- similar or better MPJPE than B1
- lower latency than Transformer encoder
- lower or more stable GPU memory footprint

### H3: Draft-to-Refine Improves Difficult Poses

Expected signs:

- B4 better than B3
- stronger gains in 2-person and 3-person settings
- visible correction of coarse joints after refinement

### H4: Structure Loss Improves Realism

Expected signs:

- B5 better bone-length consistency than B4
- cleaner bone-length consistency
- fewer qualitative failure examples

## Tables To Produce

### Table 1: Main Comparison With Baseline

Columns:

- Method
- MPJPE
- MPJPE 1-person
- MPJPE 2-person
- MPJPE 3-person
- PJDLE(h)
- PJDLE(v)
- PJDLE(d)

Rows:

- CVPR baseline
- B0
- B1
- B2
- B4
- B5

### Table 2: Accuracy-Efficiency Trade-Off

Columns:

- Method
- Params
- Latency
- FPS
- Peak Memory
- MPJPE

Rows:

- B0
- B2
- B5

### Table 3: Full Ablation

Columns:

- Input Adapter
- WiMamba
- Draft Head
- Flow Refine
- BoneLengthLoss
- MPJPE
- Bone Error
- Latency

Rows:

- B0 to B5

### Table 4: Structural Realism

Columns:

- Method
- Mean Bone Error
- Upper-body bone error
- Lower-body bone error
- Notes

Rows:

- B0
- B4
- B5

## Figures To Prepare

### Figure 1: Overall Architecture

The main system figure should show:

- raw CSI input
- optional baseline linear projection branch for comparison in caption or inset
- spectral tokenizer with `1D Conv` and `2D FFT` branches
- WiMamba encoder stack
- cross-attention query decoder
- draft pose output
- rectified flow refinement path
- final 3D pose with structural constraints

### Figure 2: Qualitative Comparison

Choose 3 categories:

- easy 1-person case
- moderate 2-person interaction case
- hard 3-person or overlapping case

For each case show:

- GT
- baseline prediction
- full model prediction

### Figure 3: Draft-To-Refine Visualization

Show:

- draft pose
- one or two intermediate refinement steps
- final refined pose

This will make the flow branch visually convincing.

### Figure 4: Accuracy-Efficiency Trade-Off Plot

Use:

- x-axis: latency or params
- y-axis: MPJPE

Plot:

- baseline
- B1
- B2
- B5

### Figure 5: Tokenizer / Frequency Cue Illustration

Show one of:

- CSI temporal trace
- spectral response after FFT
- branch outputs from tokenizer
- a visual explanation of why spectral cues help capture motion

This figure strengthens the motivation for the tokenizer.

## Qualitative Cases To Collect

Create a small curated set of examples:

- `Case A`: 1-person, easy
- `Case B`: 2-person, moderate separation
- `Case C`: 2-person, overlap or proximity
- `Case D`: 3-person, hardest case
- `Case E`: extremity failure under baseline but corrected by full model

For each case, save:

- sample ID
- GT pose
- B0 baseline pose
- full model pose
- short caption describing the failure mode

## One-Week Experiment Schedule

### Day 1

- lock evaluation script
- reproduce `B0` linear baseline result
- confirm logging format

### Day 2

- run `B1`
- run `B2`
- collect first efficiency measurements

### Day 3

- run `B3`
- run `B4`

### Day 4

- run `B5`
- inspect training stability
- export preliminary metrics

### Day 5

- rerun key models if needed
- benchmark latency, memory, throughput
- select qualitative samples

### Day 6

- finalize tables
- generate figures
- write experiment section draft

### Day 7

- write abstract
- write introduction
- write method section
- write conclusion and limitations

## Writing Outline

### Abstract

Should contain:

- problem
- baseline limitation
- your 3-part solution
- one-line result summary

### Introduction

Keep it simple:

1. Why WiFi-based multi-person 3D pose matters
2. Why Person-in-WiFi 3D is the right foundation
3. What its remaining limitations are
4. How your method addresses them
5. Contributions

### Related Work

Use 3 subsections:

- WiFi-based human pose estimation
- efficient sequence modeling for wireless signals
- structural / realistic pose refinement

### Method

Suggested subsections:

1. Problem formulation
2. Baseline Linear Projection and Spectral Tokenizer
3. WiMamba Encoder
4. Draft Pose Decoder
5. Rectified Flow Refinement
6. BoneLengthLoss Structural Loss
7. Training objective

### Experiments

Suggested subsections:

1. Dataset and protocol
2. Implementation details
3. Main comparison
4. Ablation study
5. Efficiency analysis
6. Qualitative analysis
7. Limitations

## Concrete Deliverables To Prepare

Before writing the paper, the following artifacts should exist:

- one CSV with all main metrics
- one CSV with efficiency metrics
- one CSV with ablation metrics
- one folder of curated qualitative images
- one architecture figure
- one trade-off plot
- one draft-to-refine visualization figure
- one per-joint or bone-error summary table

## Success Criteria For The First Release

This first arXiv/workshop version is successful if:

1. `B5` beats the reproduced baseline on overall MPJPE
   baseline here means `B0` with linear projection, not a spectral checkpoint
2. `B2` or `B5` clearly improves the accuracy-efficiency trade-off
3. qualitative examples visibly support the refinement and structure claims
4. the paper story stays clean and does not over-claim robustness beyond the official split

## Risks And Guardrails

### Risk 1: Too Many Moving Parts

Guardrail:

- keep the paper centered on one unified pipeline
- do not describe modules as unrelated tricks

### Risk 2: Efficiency Claim Without Evidence

Guardrail:

- always pair WiMamba with latency and memory numbers

### Risk 3: Flow Improves Little Quantitatively

Guardrail:

- use qualitative refinement figure
- use structural metrics, not only MPJPE

### Risk 4: One-Week Budget Overrun

Guardrail:

- prioritize `B0, B1, B2, B4, B5`
- drop extra robustness experiments

## Final Recommendation

For the first public version, position the paper as:

`an efficient and structurally aware upgrade of Person-in-WiFi 3D, not a fully generalized cross-domain WiFi pose framework`

That framing is the safest, clearest, and strongest match for the current codebase and the available experiment budget.
