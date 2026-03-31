# Paper Asset Checklist

Reference master plan:
[2026-03-31-balanced-arxiv-workshop-paper-plan.md](D:/Resfes_2026/Person-in-WiFi-3D/docs/paper/2026-03-31-balanced-arxiv-workshop-paper-plan.md)

## Usage

Use this file to track all assets needed for the first `arXiv/workshop` version:

- tables
- figures
- CSV files
- qualitative examples
- benchmark summaries

## Asset Root Structure

Recommended folders:

- `paper_assets/tables/`
- `paper_assets/figures/`
- `paper_assets/logs/`
- `paper_assets/qualitative/`

## Command-To-Asset Mapping

For every reported ablation, the asset source of truth should come from the canonical run directory:

- `B0` -> `work_dirs/paper/B0/`
- `B1` -> `work_dirs/paper/B1/`
- `B2` -> `work_dirs/paper/B2/`
- `B4` -> `work_dirs/paper/B4/`
- `B5` -> `work_dirs/paper/B5/`

Expected fixed artifacts:

- `paper_assets/logs/<ID>_eval.json`
- `paper_assets/logs/<ID>_benchmark.json`
- `paper_assets/logs/experiment_log.csv`

Important note:

- if a run was launched with `--cfg-options`, use the dumped config inside `work_dirs/paper/<ID>/` as the official config reference for that ablation
- do not build final paper assets from `B3` until flow can be disabled cleanly for a paper-clean draft-only ablation

## Master Completion Check

- [ ] all tables have frozen numbers
- [ ] all figures have final captions
- [ ] all CSV sources are versioned
- [ ] every paper claim maps to at least one asset
- [ ] every asset filename is stable and human-readable

## Table Checklist

### Table 1: Main Comparison

Purpose:

- compare the proposed method with the reproduced CVPR baseline
- make it explicit that `B0` is the linear-projection baseline, not a spectral run

Required columns:

- [ ] Method
- [ ] MPJPE
- [ ] MPJPE 1-person
- [ ] MPJPE 2-person
- [ ] MPJPE 3-person
- [ ] PJDLE(h)
- [ ] PJDLE(v)
- [ ] PJDLE(d)

Required rows:

- [ ] Baseline / `B0` (`WifiInputAdapter(mode='linear')`)
- [ ] `B1`
- [ ] `B2`
- [ ] `B4`
- [ ] `B5`

Source files:

- [ ] main metrics CSV
- [ ] person-count breakdown CSV

Output files:

- [ ] markdown version
- [ ] paper-ready image or LaTeX table version

### Table 2: Accuracy-Efficiency Trade-Off

Required columns:

- [ ] Method
- [ ] Params
- [ ] Latency
- [ ] FPS
- [ ] Peak Memory
- [ ] MPJPE

Required rows:

- [ ] `B0` (`linear`)
- [ ] `B2`
- [ ] `B5`

Source files:

- [ ] benchmark summary CSV
- [ ] main metrics CSV

Output files:

- [ ] markdown version
- [ ] paper-ready version

### Table 3: Full Ablation

Required columns:

- [ ] Input Adapter
- [ ] WiMamba
- [ ] Draft Head
- [ ] Flow Refine
- [ ] Bone/Limb Loss
- [ ] MPJPE
- [ ] Bone Error
- [ ] Latency

Required rows:

- [ ] `B0` (`linear`)
- [ ] `B1` (`spectral`)
- [ ] `B2`
- [ ] `B3`
- [ ] `B4`
- [ ] `B5`

Source files:

- [ ] all ablation CSV exports

Output files:

- [ ] markdown version
- [ ] paper-ready version

### Table 4: Structural Realism

Required columns:

- [ ] Method
- [ ] Mean Bone Error
- [ ] Upper-Limb Error
- [ ] Lower-Limb Error
- [ ] Note

Required rows:

- [ ] `B0` (`linear baseline`)
- [ ] `B4`
- [ ] `B5`

Source files:

- [ ] bone error CSV
- [ ] per-bone or grouped structural summary

Output files:

- [ ] markdown version
- [ ] paper-ready version

## Figure Checklist

### Figure 1: Overall Architecture

Must show:

- [ ] Raw CSI
- [ ] optional inset or caption note for the baseline linear projection branch
- [ ] Spectral Tokenizer
- [ ] WiMamba Encoder
- [ ] Cross-Attention Decoder
- [ ] Draft Pose
- [ ] Rectified Flow Refinement
- [ ] Bone/Limb Constraints
- [ ] Final 3D Pose

Asset requirements:

- [ ] editable source
- [ ] exported PNG
- [ ] exported PDF or SVG
- [ ] final caption

### Figure 2: Qualitative Comparison

Required cases:

- [ ] easy 1-person case
- [ ] medium 2-person case
- [ ] hard 3-person or overlap case

For each case, include:

- [ ] GT
- [ ] `B0` linear-baseline prediction
- [ ] full model prediction
- [ ] sample ID recorded

Asset requirements:

- [ ] combined panel figure
- [ ] high-resolution export
- [ ] short caption explaining improvement

### Figure 3: Draft-to-Refine Visualization

Must show:

- [ ] draft pose
- [ ] at least one intermediate refinement step
- [ ] final refined pose

Asset requirements:

- [ ] selected sample ID
- [ ] consistent camera/view angle
- [ ] panel labels
- [ ] caption linking to flow refinement claim

### Figure 4: Accuracy-Efficiency Trade-Off Plot

Must show:

- [ ] `B0` linear-baseline point
- [ ] `B1`
- [ ] `B2`
- [ ] `B5`

Plot requirements:

- [ ] x-axis chosen and fixed
- [ ] y-axis chosen and fixed
- [ ] model labels readable
- [ ] workshop-friendly styling

### Figure 5: Spectral Tokenizer Illustration

Choose at least one of:

- [ ] raw CSI temporal pattern
- [ ] FFT / spectral response
- [ ] local-global branch visualization
- [ ] token feature comparison

Purpose:

- [ ] justify why spectral tokenization is meaningful for WiFi
- [ ] make clear that the comparison target is the baseline linear projection in `B0`

## CSV Checklist

### Main Metrics CSV

- [ ] one row per model
- [ ] overall metrics
- [ ] matching ablation IDs
- [ ] `B0` row explicitly marked as `linear`
- [ ] no duplicated or stale runs

### Person-Count Breakdown CSV

- [ ] `1-person`
- [ ] `2-person`
- [ ] `3-person`
- [ ] `B0` row explicitly marked as `linear`
- [ ] same model naming as main metrics CSV

### Benchmark CSV

- [ ] Params
- [ ] Latency
- [ ] FPS
- [ ] Peak Memory
- [ ] optional FLOPs

### Structural Metrics CSV

- [ ] mean bone error
- [ ] grouped structural fields if used
- [ ] same model naming as other CSVs

## Qualitative Asset Checklist

For every chosen sample:

- [ ] sample ID
- [ ] GT file saved
- [ ] `B0` linear-baseline result saved
- [ ] full model result saved
- [ ] failure type tagged
- [ ] caption note written

Suggested tags:

- [ ] depth ambiguity
- [ ] limb distortion
- [ ] crowd overlap
- [ ] extremity correction
- [ ] multi-person separation

## Caption Checklist

Every figure/table needs:

- [ ] short title
- [ ] one-sentence takeaway
- [ ] no unsupported claims
- [ ] terminology consistent with paper text

## Final Paper Mapping

### Abstract Support

- [ ] one final main-comparison number
- [ ] one efficiency number
- [ ] one realism statement supported by figure/table

### Introduction Support

- [ ] architecture figure
- [ ] one motivation example

### Method Support

- [ ] pipeline figure
- [ ] tokenizer illustration
- [ ] draft-to-refine figure

### Experiments Support

- [ ] main comparison table
- [ ] efficiency table
- [ ] ablation table
- [ ] structural realism table
- [ ] qualitative comparison figure

## Freeze Checklist Before Writing

- [ ] no placeholder numbers remain
- [ ] no duplicate versions of the same table remain
- [ ] filenames are final
- [ ] captions are final enough for drafting
- [ ] all source CSV files are backed up
- [ ] best checkpoints for all reported models are recorded
