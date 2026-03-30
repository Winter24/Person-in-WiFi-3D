# Paper Experiment Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paper-clean experiment scaffold so the repo can reproduce the planned `B0 -> B5` ladder, export the required metrics, and generate paper-ready tables/figures.

**Architecture:** Keep the current repo evaluation path as the single source of truth for `MPJPE/PJDLE/per-joint/bone error`, but move ablation control to config-driven switches. Separate the work into three layers: model/config toggles, experiment configs, and paper-export tooling.

**Tech Stack:** MMCV/MMDetection configs, PyTorch modules, repo-local analysis scripts, CSV/JSON export, matplotlib-based plotting.

---

## Current Audit Summary

### Already available

- `SpectralTokenizer` is implemented and used by default in [petr.py](/D:/Resfes_2026/Person-in-WiFi-3D/opera/models/detectors/petr.py)
- `MambaEncoder` and `WiMambaEncoder` exist in the repo
- `WiTiDARHead` now supports Mamba + draft + flow + structural loss path
- `WifiPoseDataset.evaluate()` already computes:
  - `MPJPE`
  - `PJDLE(h/v/d)`
  - per-joint MPJPE
  - mean bone-length error
- `tools/analysis/benchmark.py` exists for params/FLOPs/latency/FPS
- `tools/analysis/compute_bone_stats.py` exists for `gt_bone_stats.json`
- qualitative scripts already exist

### Not yet sufficient for the paper plan

1. **B0 is still missing**
   - [petr.py](/D:/Resfes_2026/Person-in-WiFi-3D/opera/models/detectors/petr.py) still hard-codes `SpectralTokenizer`
   - there is no config-only path for `Linear -> Transformer -> PETR`

2. **The repo does not yet have a paper-clean config ladder**
   - no dedicated config files for `B0/B1/B2/B3/B4/B5`
   - current configs are a mix of old experiments and new branches
   - current [wi_tidir_wifi.py](/D:/Resfes_2026/Person-in-WiFi-3D/configs/wifi/wi_tidir_wifi.py) is closer to `B5`, but not enough to cover the whole ablation table

3. **B3 cannot be isolated cleanly by config today**
   - current [wi_tidar_head.py](/D:/Resfes_2026/Person-in-WiFi-3D/opera/models/dense_heads/wi_tidar_head.py) always builds the flow branch if flow modules import successfully
   - inference also always applies flow when `self.flow_model` exists

4. **The structural loss path is not aligned with the original plan wording**
   - the plan says `Bone Loss`
   - current `WiTiDARHead` uses `LimbLoss`
   - this must be made explicit and configurable, otherwise the paper claim and the code path will drift apart

5. **Paper export tooling is still missing**
   - there is no script to export evaluation results into `csv/json`
   - there is no script for `1-person / 2-person / 3-person` breakdown
   - there is no script to merge metrics + benchmark results into one ablation sheet
   - there is no script to generate the planned paper figures from saved outputs

## Target Deliverables

When this support work is complete, the repo should contain:

- a clean config family for `B0/B1/B2/B3/B4/B5`
- one official tokenizer switch for `Linear` vs `SpectralTokenizer`
- one official sequence encoder path for `Transformer` vs `Mamba`
- one official draft head path with config switches for:
  - draft only
  - draft + flow
  - draft + flow + structural loss
- one paper evaluation export script
- one person-count breakdown script
- one benchmark summary script
- one figure/table generation script family

## File Structure To Add

### Model / Config layer

- Modify: `opera/models/detectors/petr.py`
- Modify: `opera/models/dense_heads/wi_tidar_head.py`
- Create: `configs/wifi/paper/_paper_base.py`
- Create: `configs/wifi/paper/b0_linear_transformer_petr.py`
- Create: `configs/wifi/paper/b1_spectral_transformer_petr.py`
- Create: `configs/wifi/paper/b2_spectral_mamba_petr.py`
- Create: `configs/wifi/paper/b3_spectral_mamba_draft.py`
- Create: `configs/wifi/paper/b4_spectral_mamba_draft_flow.py`
- Create: `configs/wifi/paper/b5_spectral_mamba_draft_flow_bone.py`

### Analysis / export layer

- Create: `tools/analysis/export_wifi_pose_metrics.py`
- Create: `tools/analysis/export_person_count_breakdown.py`
- Create: `tools/analysis/export_benchmark_summary.py`
- Create: `tools/analysis/make_paper_tables.py`
- Create: `tools/analysis/make_paper_figures.py`

### Output / artifact layer

- Reuse: `docs/paper/experiment_log_template.csv`
- Create: `paper_assets/tables/.gitkeep`
- Create: `paper_assets/figures/.gitkeep`
- Create: `paper_assets/logs/.gitkeep`
- Create: `paper_assets/qualitative/.gitkeep`

## Task 1: Make The Detector Tokenizer Configurable

**Files:**
- Modify: `D:\Resfes_2026\Person-in-WiFi-3D\opera\models\detectors\petr.py`
- Test: config parsing via source compile and model build smoke test when environment is ready

- [ ] Add a detector-level config argument such as `tokenizer_cfg`
- [ ] Support two official modes:
  - `type='LinearProjection'`
  - `type='SpectralTokenizer'`
- [ ] Keep `SpectralTokenizer` as the default for backward compatibility
- [ ] Preserve current tensor reshape contract before the head
- [ ] Add a short code comment documenting that this switch exists to support fair paper ablations
- [ ] Verify source compiles cleanly

**Expected outcome:** `B0` and `B1/B2` can differ only in tokenizer choice without patching code again.

## Task 2: Make WiTiDARHead Truly Ablation-Friendly

**Files:**
- Modify: `D:\Resfes_2026\Person-in-WiFi-3D\opera\models\dense_heads\wi_tidar_head.py`
- Optionally modify: `D:\Resfes_2026\Person-in-WiFi-3D\opera\models\losses\bone_loss.py`
- Optionally modify: `D:\Resfes_2026\Person-in-WiFi-3D\opera\models\losses\limb_loss.py`

- [ ] Add an explicit flow toggle, e.g. `enable_flow=True/False` or `flow_cfg=None`
- [ ] Ensure `simple_test()` and `get_bboxes()` skip refinement when flow is disabled
- [ ] Add an explicit structural loss switch
- [ ] Decide the official paper path:
  - either `loss_bone`
  - or `loss_limb`
  - or support both and choose one per config
- [ ] If `BoneLengthLoss` is the official paper claim, wire `gt_bone_stats.json` into `WiTiDARHead`
- [ ] Keep zero-cost fallback behavior for disabled branches so logs remain stable
- [ ] Keep the current `mamba_cfg` config-driven interface
- [ ] Verify source compiles cleanly

**Expected outcome:** `B3/B4/B5` become true config-level ablations instead of code forks.

## Task 3: Build The Official Paper Config Ladder

**Files:**
- Create: `D:\Resfes_2026\Person-in-WiFi-3D\configs\wifi\paper\_paper_base.py`
- Create: `D:\Resfes_2026\Person-in-WiFi-3D\configs\wifi\paper\b0_linear_transformer_petr.py`
- Create: `D:\Resfes_2026\Person-in-WiFi-3D\configs\wifi\paper\b1_spectral_transformer_petr.py`
- Create: `D:\Resfes_2026\Person-in-WiFi-3D\configs\wifi\paper\b2_spectral_mamba_petr.py`
- Create: `D:\Resfes_2026\Person-in-WiFi-3D\configs\wifi\paper\b3_spectral_mamba_draft.py`
- Create: `D:\Resfes_2026\Person-in-WiFi-3D\configs\wifi\paper\b4_spectral_mamba_draft_flow.py`
- Create: `D:\Resfes_2026\Person-in-WiFi-3D\configs\wifi\paper\b5_spectral_mamba_draft_flow_bone.py`

- [ ] Move shared dataset/runtime/training settings into `_paper_base.py`
- [ ] Freeze one common experimental protocol:
  - batch size
  - optimizer
  - epochs
  - evaluation interval
  - checkpoint policy
  - work_dir naming
- [ ] Implement `B0` as `Linear + Transformer + PETR + no structural loss`
- [ ] Implement `B1` as `SpectralTokenizer + Transformer + PETR + no structural loss`
- [ ] Implement `B2` as `SpectralTokenizer + Mamba + PETR + no structural loss`
- [ ] Implement `B3` as `SpectralTokenizer + Mamba + Draft head + no flow + no structural loss`
- [ ] Implement `B4` as `SpectralTokenizer + Mamba + Draft head + flow + no structural loss`
- [ ] Implement `B5` as `SpectralTokenizer + Mamba + Draft head + flow + structural loss`
- [ ] Give every config a paper-specific `work_dir`
- [ ] Add one short header comment in each config mapping it to the ablation table ID
- [ ] Verify all config files parse as Python source

**Expected outcome:** the ablation table in the paper maps one-to-one to config filenames.

## Task 4: Export Canonical Metrics To JSON And CSV

**Files:**
- Create: `D:\Resfes_2026\Person-in-WiFi-3D\tools\analysis\export_wifi_pose_metrics.py`

- [ ] Accept inputs:
  - config path
  - result `.pkl`
  - optional output directory
- [ ] Reuse the same dataset/evaluate logic already used in the repo
- [ ] Export:
  - overall metrics
  - per-joint MPJPE
  - per-bone error
- [ ] Save both:
  - machine-readable `.json`
  - flat `.csv` for spreadsheets
- [ ] Match column names with `docs/paper/experiment_log_template.csv`
- [ ] Print a concise terminal summary for quick sanity checking

**Expected outcome:** one evaluation run produces artifacts that can be merged into tables without manual copy-paste.

## Task 5: Add Person-Count Breakdown Export

**Files:**
- Create: `D:\Resfes_2026\Person-in-WiFi-3D\tools\analysis\export_person_count_breakdown.py`
- Optionally modify: `D:\Resfes_2026\Person-in-WiFi-3D\opera\datasets\wifi_pose.py`

- [ ] Group samples by ground-truth person count:
  - 1-person
  - 2-person
  - 3-person
- [ ] Compute the same canonical metrics per group
- [ ] Export `.json` and `.csv`
- [ ] Keep this breakdown outside the canonical `evaluate()` summary unless there is a strong reason to bake it in
- [ ] Document the command in a header comment

**Expected outcome:** Table C can be generated automatically instead of manually re-running ad hoc scripts.

## Task 6: Add Benchmark Export For B0/B2/B5

**Files:**
- Create: `D:\Resfes_2026\Person-in-WiFi-3D\tools\analysis\export_benchmark_summary.py`
- Reuse: `D:\Resfes_2026\Person-in-WiFi-3D\tools\analysis\benchmark.py`

- [ ] Wrap or parse benchmark output into structured fields:
  - params
  - FLOPs
  - latency
  - FPS
- [ ] Standardize warmup/runs/device settings so benchmarks are comparable
- [ ] Save summary files per config
- [ ] Save one merged ablation benchmark CSV for paper tables

**Expected outcome:** Table A and Figure 6 can be regenerated directly from saved benchmark summaries.

## Task 7: Generate Paper Tables And Figures

**Files:**
- Create: `D:\Resfes_2026\Person-in-WiFi-3D\tools\analysis\make_paper_tables.py`
- Create: `D:\Resfes_2026\Person-in-WiFi-3D\tools\analysis\make_paper_figures.py`

- [ ] Read exported metrics and benchmark summaries from `paper_assets/logs`
- [ ] Generate:
  - main comparison table
  - ablation table
  - per-person breakdown table
  - per-joint table
  - bone/structural table
- [ ] Generate:
  - per-joint error bar chart
  - bone error chart
  - accuracy vs efficiency scatter
- [ ] Save plots into `paper_assets/figures`
- [ ] Save tables into `paper_assets/tables`
- [ ] Keep output filenames stable so the writing workflow is reproducible

**Expected outcome:** the paper can be updated by rerunning scripts, not by manually reformatting logs.

## Task 8: Documentation And Command Sheet

**Files:**
- Modify: `D:\Resfes_2026\Person-in-WiFi-3D\docs\paper\2026-03-30-person-in-wifi3d-improvement-plan.md`
- Optionally create: `D:\Resfes_2026\Person-in-WiFi-3D\docs\paper\paper_experiment_commands.md`

- [ ] Update the original paper plan to point to the final official config filenames
- [ ] Add the command sequence for:
  - bone stats
  - train
  - test with pkl export
  - metrics export
  - person-count breakdown export
  - benchmark export
  - figure/table generation
- [ ] Add a short note describing which structural loss is the official paper loss

**Expected outcome:** future writing sessions can rerun the full experiment pipeline without re-auditing the repo.

## Recommended Execution Order

1. Task 1: tokenizer switch
2. Task 2: `WiTiDARHead` toggles for flow + structural loss
3. Task 3: create `B0 -> B5` config family
4. Task 4: canonical metric export
5. Task 5: person-count breakdown
6. Task 6: benchmark export
7. Task 7: table/figure generation
8. Task 8: docs and command sheet

## Definition Of Done

This support work is complete only when all of the following are true:

- `B0/B1/B2/B3/B4/B5` each have one dedicated config file
- `B0` uses `Linear`, not `SpectralTokenizer`
- `B3` truly skips flow at both train and test time
- `B4` truly uses flow but no structural loss
- `B5` truly uses flow plus the official structural loss selected for the paper
- exported metrics exist as `.json` and `.csv`
- person-count breakdown exists as `.json` and `.csv`
- benchmark summaries exist as `.json` and `.csv`
- paper tables and figures can be generated from saved artifacts

## Practical First Milestone

If implementation time is tight, the minimum useful milestone is:

- tokenizer switch in `PETR`
- `WiTiDARHead` flow toggle
- clean configs for `B0/B1/B2/B4/B5`
- one metrics export script
- one benchmark export script

That is enough to start filling Table A, Table B, and Figure 6 while B3 and the person-count breakdown are still in progress.
