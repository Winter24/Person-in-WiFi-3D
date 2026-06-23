# Manuscript Figure Refresh Design

## Objective

Refresh three manuscript elements without changing reported quantitative results:

1. use the clean flow-refinement diagram;
2. replace the acquisition-setup figure with an original prose description grounded in the CVPR 2024 Person-in-WiFi 3D paper;
3. regenerate the qualitative comparison from local test data and checkpoints for the same three samples already shown in the manuscript.

## Scope

The implementation is limited to:

- paper_assets/manuscript_latex/resfes2026_witidar/main.tex;
- figure assets under paper_assets/manuscript_latex/resfes2026_witidar/figures;
- qualitative rendering support under tools/analysis;
- temporary source-video downloads and extracted RGB frames required for Figure 8.

No training, metric-table changes, checkpoint modification, or sample reselection is in scope.

## Flow-Refinement Figure

Change the LaTeX include from figures/fig_flow_refinement.pdf to
figures/flow_refined_clean.pdf. Preserve the existing caption and label unless
visual inspection shows that the new image requires a narrowly scoped caption
correction.

## Acquisition Setup

Remove the fig_dataset_setup.pdf figure environment and its caption. Expand the
Dataset and Acquisition prose with a paraphrased description based on
Template/2024CVPR_Person_in_WiFi_3D.pdf, cited as
\\cite{yan2024personwifi3d}.

The paragraph will retain the setup facts needed for reproducibility:

- four ThinkPad X201 laptops with Intel 5300 adapters;
- one transmitter and three receivers;
- channel 128 at 5.64 GHz with 30 subcarriers;
- transmitter rate of 300 packets per second;
- one transmit antenna and three antennas per receiver;
- Azure Kinect RGB-D capture at 15 FPS;
- synchronization of Kinect and receivers before recording;
- one annotated frame paired with a CSI window of shape 1 x 3 x 3 x 30 x 20;
- amplitude and phase concatenation yielding 180 tokens of 60 values.

The wording and sentence structure must be independently written. The result
must not preserve distinctive phrasing from the source paper.

## Qualitative Figure Data

Keep the existing fixed samples:

| Dataset index | Sample name | People | Video ID | Frame |
|---:|---|---:|---|---:|
| 231 | S11_06_319 | 1 | S11_06 | 319 |
| 7218 | S52_40_322 | 2 | S52_40 | 322 |
| 4416 | S23_12_337 | 3 | S23_12 | 337 |

Each Video ID maps to its own Drive subfolder:
RESEARCH/RESFES2026/<Video ID>/output.mkv and time_list.txt. Source videos may
be downloaded to a temporary workspace directory, used to extract the required
frames, and removed after verification. Extracted frames are saved as
paper_assets/manuscript_latex/resfes2026_witidar/figures/qualitative_rgb/<sample_name>.png.

The final integer in a sample name is the synchronized frame ID. The renderer
parses time_list.txt using the existing render_presentation_video.py convention.
For the current N_<timestamp> format, frame ID N maps to video frame index N.
Extraction must fail if the requested ID is absent; it must not silently use a
neighboring frame or apply an offset.

Ground truth and model inputs come from data/wifipose/test_data.

Model assets:

| Display ID | Config | Checkpoint | Inference override |
|---|---|---|---|
| M0 | work_dirs/full_alation_20e/M0/petr_wifi.py | work_dirs/full_alation_20e/M0/latest.pth | none |
| M7 | work_dirs/full_alation_20e/M7/wi_tidir_wifi_transformer.py | work_dirs/full_alation_20e/M7/latest.pth | none |
| M9_RF2 | configs/wifi/wi_tidir_wifi_mamba2_flattened_eval_rf_2step.py | work_dirs/full_alation_20e/M9/latest.pth | already set to flow_num_steps=2 by config |

The renderer must fail clearly if a sample, RGB frame, config, or checkpoint is
missing. It must verify that all model configs resolve to the same test dataset
signature. The signature consists of the resolved dataset root, dataset mode,
and a hash of the ordered sample-name list. M0 and M7 use config snapshots saved
beside their checkpoints because those files exactly describe the trained
models. M9_RF2 uses the repository two-step evaluation config because it
inherits the trained M9 architecture and changes only inference integration to
two steps. No additional runtime mutation of flow_num_steps is applied.

## Figure 8 Layout

Use a three-row by five-column full comparison grid:

1. RGB scene;
2. Kinect ground truth;
3. M0;
4. M7;
5. M9_RF2.

Rows correspond to the 1-person, 2-person, and 3-person fixed samples. Do not put
a figure title inside the image; the title remains in the LaTeX caption.

For each row, all 3D panels share camera angle and coordinate bounds. Matched
predictions reuse the corresponding ground-truth person color. Predictions above
the fixed confidence threshold of 0.2 that remain unmatched are shown in gray
rather than hidden. This threshold must not be tuned after viewing the samples.

Prediction-to-ground-truth assignment uses Hungarian matching on mean Euclidean
3D joint distance. Assigned pairs above the manuscript evaluation threshold of
500 mm are rejected as unmatched. Matched sample MPJPE is the mean joint
distance over retained pairs. False positives and false negatives are computed
after this thresholded assignment.

Each pose panel includes a compact diagnostic footer containing:

- sample name;
- matched/ground-truth count;
- false positives;
- false negatives;
- matched sample MPJPE in millimeters.

The renderer exports a 300-DPI PNG for the manuscript and a PDF version for
inspection. Typography must remain legible at double-column width, use a
colorblind-safe palette, and avoid decorative framing.

## Manuscript Integration

Replace fig_qualitative_relabeled.png with the regenerated Figure 8 asset.
Update the Failure Cases paragraph and caption only as needed to state:

- the exact three displayed model variants;
- unmatched predictions are gray;
- footer metrics are per-sample diagnostics and not aggregate test metrics.

Do not infer broad qualitative superiority from three selected examples.

## Verification

1. For every RGB frame, log the sample name, requested frame ID, resolved video
   index, and source video. Require the frame ID to exist in the time map and
   require OpenCV seeking to return that exact video index.
2. Confirm M0, M7, and M9_RF2 load their intended configs and checkpoints.
3. Record per-panel matching diagnostics produced by inference.
4. Visually inspect the exported PNG at original size and at manuscript scale.
5. Build LaTeX using pdflatex -> bibtex -> pdflatex -> pdflatex.
6. Check for undefined citations/references, missing graphics, fatal errors, and incoherent layout.
7. Render the final PDF pages containing the new flow figure, acquisition prose, and Figure 8 for visual inspection.
8. Confirm yan2024personwifi3d exists in the bibliography and resolves without an undefined-citation warning.
9. Compare the new acquisition paragraph against the extracted source paragraph
   and flag any shared sequence of eight or more normalized words for rewrite.
10. Diff Results tables and aggregate metric statements before and after the
    change; no numeric value outside per-sample Figure 8 diagnostics may change.

## Out of Scope

- changing the three selected samples;
- retraining or tuning any model;
- changing the confidence threshold to improve appearance after seeing outputs;
- altering aggregate results, tables, or claims;
- retaining multi-gigabyte source videos in the repository.
