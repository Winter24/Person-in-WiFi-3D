# Person-in-WiFi 3D Improvement Paper Plan

## Goal

Write a direct follow-up paper to *Person-in-WiFi 3D* with the unified pipeline:

`CSI -> SpectralTokenizer -> Mamba -> draft pose -> Flow Matching -> Bone Loss`

The paper should not read as a loose collection of tricks. It should tell one consistent story:

1. CSI needs a better tokenization stage than a plain linear projection.
2. Long WiFi sequences should be modeled by a sequence encoder that is more suitable than quadratic self-attention.
3. Multi-person 3D pose should be predicted in a coarse-to-fine way.
4. Final poses should be structurally plausible, not only numerically close.

## What Already Exists In This Repo

The current branch already contains most building blocks needed for the paper:

- `SpectralTokenizer` in [spectral_tokenizer.py](/D:/Resfes_2026/Person-in-WiFi-3D/opera/models/utils/spectral_tokenizer.py)
- `MambaEncoder` in [mamba_encoder.py](/D:/Resfes_2026/Person-in-WiFi-3D/opera/models/utils/mamba_encoder.py)
- `WiMambaEncoder` in [wimamba.py](/D:/Resfes_2026/Person-in-WiFi-3D/opera/models/backbones/wimamba.py)
- `RectifiedFlowWrapper` in [rectified_flow.py](/D:/Resfes_2026/Person-in-WiFi-3D/opera/models/utils/rectified_flow.py)
- `VelocityMLP` in [flow_components.py](/D:/Resfes_2026/Person-in-WiFi-3D/opera/models/dense_heads/flow_components.py)
- `BoneLengthLoss` in [bone_loss.py](/D:/Resfes_2026/Person-in-WiFi-3D/opera/models/losses/bone_loss.py)
- detailed evaluation with MPJPE, per-joint MPJPE, and bone-length error in [wifi_pose.py](/D:/Resfes_2026/Person-in-WiFi-3D/opera/datasets/wifi_pose.py)
- runtime and complexity benchmark script in [benchmark.py](/D:/Resfes_2026/Person-in-WiFi-3D/tools/analysis/benchmark.py)

Important code audit notes:

- The detector in [petr.py](/D:/Resfes_2026/Person-in-WiFi-3D/opera/models/detectors/petr.py) already uses `SpectralTokenizer`, so the current "baseline" is not the original CVPR baseline anymore.
- There are two Mamba implementations: `MambaEncoder` and `WiMambaEncoder`. For the paper, only one should be kept as the official implementation path.
- The full `draft -> flow refine` idea is implemented in [wi_tidar_head.py](/D:/Resfes_2026/Person-in-WiFi-3D/opera/models/dense_heads/wi_tidar_head.py), while the PETR branch still carries the original decoder structure plus `BoneLengthLoss`.

## Recommended Paper Positioning

Use the original CVPR paper as the base system paper and the robust/realistic WiFi pose paper as the motivation for refinement and topology.

Recommended contribution framing:

1. **Frequency-aware CSI tokenization**
   Replace plain token embedding with a local-global tokenizer that fuses temporal convolution and spectral gating.
2. **Linear-complexity sequence modeling**
   Replace the Transformer encoder with Mamba to better model long CSI sequences with lower inference cost.
3. **Draft-to-refine pose generation**
   Predict coarse pose hypotheses first, then refine them by conditional flow matching.
4. **Topology-aware structural regularization**
   Add bone-length constraints to reduce unrealistic skeleton distortion.

Recommended one-sentence claim:

> We improve Person-in-WiFi 3D by combining frequency-aware CSI tokenization, linear-time sequence modeling, coarse-to-fine flow-based pose refinement, and anatomy-aware structural supervision.

## Main Narrative

The paper should follow this logic:

- The original Person-in-WiFi 3D already solved direct multi-person 3D pose estimation from WiFi.
- However, three problems remain:
  - the CSI embedding stage is weak if it only uses simple projection;
  - Transformer encoding is expensive for long WiFi sequences;
  - direct coordinate regression still produces unstable extremities and unrealistic limb geometry.
- The proposed method addresses these three issues in a matched order:
  - `SpectralTokenizer` improves input representation,
  - `Mamba` improves sequence modeling efficiency,
  - `draft pose + Flow Matching + Bone Loss` improves final pose realism and accuracy.

## Experimental Scope

### Primary Dataset

Use the official Person-in-WiFi-3D split first.

From the original CVPR paper:

- train: 89,946 frames
- test: 7,824 frames
- breakdown:
  - 1-person train/test: 28,121 / 2,586
  - 2-person train/test: 36,242 / 3,184
  - 3-person train/test: 25,583 / 2,054

### Core Metrics

Use the metrics already aligned with the original paper and current repo:

- `MPJPE`
- `PJDLE(h)`
- `PJDLE(v)`
- `PJDLE(d)`
- per-joint MPJPE
- mean bone-length error

Add efficiency metrics for the Mamba claim:

- parameter count
- FLOPs
- latency per sample
- FPS

### Optional Secondary Metrics

Only add these if implementation time allows:

- separate errors for easy joints vs hard joints
- separate errors by person count: 1-person / 2-person / 3-person
- separate bone error by upper limb vs lower limb

## Fair Comparison Rules

These rules are mandatory if the paper is going to be convincing.

1. Rebuild a true original baseline.
   The current detector already uses `SpectralTokenizer`, so a fair baseline must restore the original plain embedding path.

2. Change one factor at a time.
   Do not compare "full model" directly against original PETR only.

3. Keep decoder/query count/training setup fixed where possible.
   Otherwise the ablation will be hard to interpret.

4. Use the same evaluation code for all models.
   The current evaluation in `WifiPoseDataset.evaluate()` should be the single source of truth.

5. Report both accuracy and efficiency.
   If Mamba is introduced, speed/memory evidence is not optional.

## Recommended Ablation Ladder

This is the most important table in the paper.

| ID | Tokenizer | Encoder | Head | Flow Refine | Bone Loss | Expected Role |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | Linear | Transformer | PETR original | No | No | Reproduced CVPR baseline |
| B1 | SpectralTokenizer | Transformer | PETR original | No | No | Isolate tokenizer gain |
| B2 | SpectralTokenizer | Mamba | PETR original | No | No | Isolate encoder gain |
| B3 | SpectralTokenizer | Mamba | Draft head | No | No | Isolate draft formulation |
| B4 | SpectralTokenizer | Mamba | Draft head | Yes | No | Isolate flow refinement gain |
| B5 | SpectralTokenizer | Mamba | Draft head | Yes | Yes | Full model |

If time is limited, the minimum publishable ladder is:

- B0
- B1
- B2
- B4
- B5

## Hypotheses To Verify

For each module, define in advance what evidence counts as success.

### H1: SpectralTokenizer helps the representation

Expected evidence:

- lower MPJPE than B0
- clearer gain on hard joints like wrists and ankles
- better depth error than plain linear projection

### H2: Mamba improves sequence modeling efficiency

Expected evidence:

- similar or better MPJPE than Transformer
- lower latency or better FPS
- competitive accuracy with fewer memory bottlenecks

### H3: Flow refinement improves difficult pose cases

Expected evidence:

- lower MPJPE than the draft-only model
- visible correction in extremities and multi-person overlap cases
- better performance on 2-person and 3-person scenes than 1-person-only gains

### H4: Bone Loss improves realism

Expected evidence:

- lower mean bone-length error
- fewer visibly distorted skeletons
- improvement in symmetry-related limbs

## Main Figures To Prepare

### Figure 1: Full Method Overview

Use this as the paper's system figure.

```mermaid
flowchart LR
    accTitle: Full WiFi pose pipeline
    accDescr: End-to-end pipeline from CSI input to refined multi-person 3D pose prediction with structural supervision.

    csi["CSI"]
    tok["SpectralTokenizer"]
    enc["Mamba Encoder"]
    draft["Draft Pose Head"]
    flow["Flow Matching Refiner"]
    pose["Refined 3D Pose"]
    bone["Bone Length Loss"]

    csi --> tok --> enc --> draft --> flow --> pose
    draft -. "coarse pose" .-> flow
    pose -. "topology regularization" .-> bone

    classDef core fill:#dbeafe,stroke:#2563eb,stroke-width:1px,color:#1e3a5f
    class csi,tok,enc,draft,flow,pose,bone core
```

What to show visually:

- CSI tensor shape
- tokenizer local/global branches
- Mamba sequence block
- draft pose outputs
- flow correction arrows
- bone regularization edges

### Figure 2: SpectralTokenizer Detail

Show:

- temporal branch
- frequency branch
- fusion block
- output tokens shape

Goal:

- make it obvious why tokenizer is not just a small preprocessing trick

### Figure 3: Draft vs Refined Pose

Show side-by-side:

- ground truth
- draft prediction
- refined prediction

Pick at least:

- one 1-person easy case
- one 2-person interaction case
- one 3-person crowded case

### Figure 4: Per-Joint Error Bar Chart

Plot:

- B0 vs B2 vs B5

Purpose:

- show that gains are concentrated in difficult joints, not only averaged out

### Figure 5: Bone-Length Error Comparison

Plot:

- average bone error per limb for model without bone loss vs full model

Purpose:

- convert the structural realism claim into a measurable figure

### Figure 6: Accuracy vs Efficiency

Scatter or table-like plot:

- x-axis: latency or FPS
- y-axis: MPJPE
- points: B0, B2, B5

Purpose:

- support the Mamba efficiency narrative

## Tables To Prepare

### Table A: Main Comparison

Compare against:

- original Person-in-WiFi 3D baseline reproduction
- your proposed full model

Columns:

- MPJPE
- PJDLE(h)
- PJDLE(v)
- PJDLE(d)
- Params
- FLOPs
- Latency

### Table B: Ablation Study

Use the B0-B5 ladder above.

Columns:

- Tokenizer
- Encoder
- Draft head
- Flow refine
- Bone loss
- MPJPE
- bone error
- FPS

### Table C: Breakdown By Number Of People

Rows:

- 1-person
- 2-person
- 3-person

Columns:

- baseline MPJPE
- proposed MPJPE
- relative gain

### Table D: Per-Joint Errors

Rows:

- 14 joints

Columns:

- baseline
- full model
- improvement

Sort by highest baseline error first.

### Table E: Bone Statistics / Bone Error

Use `gt_bone_stats.json` as the reference prior and report:

- ground-truth mean bone length
- predicted mean bone length
- mean absolute bone error

## Exact Numbers To Log During Training

Create one experiment sheet per run with these fields:

| run_id | config | tokenizer | encoder | head | flow_steps | bone_weight | epochs | best_epoch | mpjpe | mpjpe_h | mpjpe_v | mpjpe_d | wrist_err | ankle_err | bone_err | params_m | flops_g | latency_ms | fps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

At minimum, keep:

- final metrics
- best checkpoint epoch
- inference speed
- experiment notes

Also log per-epoch validation curves:

- `loss_cls`
- `loss_kpt`
- `loss_flow`
- `loss_bone`
- validation `MPJPE`

## Recommended Output Folder Structure

Use a paper-specific structure so assets do not get mixed with normal training runs.

```text
work_dirs/paper/
  baseline_b0/
  ablation_b1_tokenizer/
  ablation_b2_mamba/
  ablation_b3_draft/
  ablation_b4_flow/
  full_b5/
paper_assets/
  figures/
  tables/
  logs/
  qualitative/
```

## Commands To Reuse

### Bone statistics

```powershell
python tools/analysis/compute_bone_stats.py
```

### Complexity and speed

```powershell
python tools/analysis/benchmark.py configs/wifi/petr_wifi.py
python tools/analysis/benchmark.py configs/wifi/petr_wifi_mamba.py
python tools/analysis/benchmark.py configs/wifi/wi_tidir_wifi.py
```

### Qualitative visualization

```powershell
python tools/multiperson_visualize.py
python tools/analysis/visualize_wifi_keypoints.py
```

### Evaluation from saved results

```powershell
python tools/eval_metric.py <config> <result_pkl> --eval mpjpe
```

## Suggested Experiment Order

### Stage 0: Freeze the experimental protocol

- decide the official baseline config
- decide the official full-model config
- decide one Mamba implementation only
- verify evaluation script and units in millimeters

### Stage 1: Reproduce the real baseline

- restore linear tokenizer
- run original Transformer encoder
- confirm the reproduced MPJPE is close to the original paper

### Stage 2: Component ablations

- tokenizer only
- tokenizer + Mamba
- tokenizer + Mamba + draft
- tokenizer + Mamba + draft + flow
- tokenizer + Mamba + draft + flow + bone

### Stage 3: Efficiency benchmark

- run benchmark script on B0, B2, B5
- collect params, FLOPs, latency, FPS

### Stage 4: Qualitative collection

- save 10-15 representative examples
- keep 3-5 best publication-quality examples
- include both success cases and one failure case

### Stage 5: Writing

- write method after Fig. 1 is fixed
- write experiments only after Table B is complete
- write conclusion after the main claim is numerically supported

## Paper Writing Order

Recommended order:

1. Title and contribution statement
2. Fig. 1 and method overview
3. Table B ablation study
4. Table A main result
5. qualitative figures
6. abstract and introduction

This order helps keep the writing grounded in actual evidence.

## Recommended Title Candidates

- Person-in-WiFi 3D++: Frequency-Aware and Topology-Constrained Multi-Person 3D Pose Estimation with WiFi
- Beyond Person-in-WiFi 3D: Spectral Tokenization, Mamba Encoding, and Flow-Based Pose Refinement
- Improving Person-in-WiFi 3D with Spectral Tokenization, Linear-Time Sequence Modeling, and Structural Pose Refinement

## Risks To Watch

1. **Baseline contamination**
   The current repo already changed the tokenizer, so reporting current code as the "original baseline" would be unfair.

2. **Architecture inconsistency**
   `PETR + Mamba` and `WiTiDARHead + Flow` currently live in partially different paths. The final paper model should be described as one coherent system.

3. **Bone prior bias**
   Bone-length targets are dataset averages, so they may regularize away natural subject-specific variation.

4. **Weak flow evidence**
   If flow uses only one Euler step, you must verify that the gain is real and not only due to extra capacity.

5. **Efficiency claim without timing**
   If Mamba is used, runtime evidence is required, not optional.

## Immediate Next Actions

1. Build and freeze a true B0 baseline with plain linear tokenization.
2. Choose one official Mamba implementation path.
3. Run the B0/B1/B2/B5 ladder first.
4. Save qualitative draft-vs-refined examples early.
5. Start writing only after the ablation table is stable.

## Reference Anchors Used For This Plan

- `Reference_src/2024CVPR_Person_in_WiFi_3D.pdf`
  - baseline task definition
  - official train/test split
  - official metrics
  - official 1/2/3-person MPJPE numbers

- `Reference_src/Towards Robust and Realistic Human Pose Estimation via WiFi Signals.pdf`
  - motivation for robust and realistic decoding
  - structural fidelity argument
  - support for topology-aware pose design

