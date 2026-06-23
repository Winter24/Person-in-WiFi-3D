# Full-Paper Qualitative Figure

Run this workflow on the rented Linux/CUDA server from the repository root.
The local Windows machine is not expected to provide the MMCV, Mamba-SSM, or
CUDA runtime required by the checkpoints.

## Required source videos

Prepare one root directory with this structure:

~~~text
SOURCE_VIDEO_ROOT/
  S11_06/
    output.mkv
    time_list.txt
  S52_40/
    output.mkv
    time_list.txt
  S23_12/
    output.mkv
    time_list.txt
~~~

The files are stored under Google Drive RESEARCH/RESFES2026.

## Model inputs

The command uses these fixed model assets:

~~~text
M0:
  work_dirs/full_alation_20e/M0/petr_wifi.py
  work_dirs/full_alation_20e/M0/latest.pth

M7:
  work_dirs/full_alation_20e/M7/wi_tidir_wifi_transformer.py
  work_dirs/full_alation_20e/M7/latest.pth

M9_RF2:
  configs/wifi/wi_tidir_wifi_mamba2_flattened_eval_rf_2step.py
  work_dirs/full_alation_20e/M9/latest.pth
~~~

The test dataset must exist at data/wifipose/test_data.

## Run

~~~bash
bash tools/analysis/run_full_paper_qualitative.sh /absolute/path/to/SOURCE_VIDEO_ROOT
~~~

The workflow extracts synchronized RGB frames for S11_06_319, S52_40_322, and
S23_12_337. It then runs inference with a fixed score threshold of 0.2 and
thresholded Hungarian matching at 500 mm. Predictions above the score threshold
that remain unmatched are shown in gray.

## Outputs

~~~text
paper_assets/manuscript_latex/resfes2026_witidar/figures/
  qualitative_rgb/
    S11_06_319.png
    S52_40_322.png
    S23_12_337.png
  fig_qualitative_full_paper.png
  fig_qualitative_full_paper.pdf
  fig_qualitative_full_paper_diagnostics.json
~~~

main.tex automatically prefers fig_qualitative_full_paper.png when the file
exists. The JSON file records the fixed thresholds and per-model M/GT, FP, FN,
and matched sample MPJPE diagnostics. Rebuild the manuscript after the command
finishes:

~~~bash
cd paper_assets/manuscript_latex/resfes2026_witidar
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
~~~
