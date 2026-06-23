# Figure 6 Flow-Solver Redesign

## Objective

Redraw the flow-solver analysis so the small MPJPE differences and the
unmatched-person counts remain legible at manuscript width without changing
the six evaluated configurations or any reported value.

## Data

Use the existing evaluation logs as the source of truth and preserve this
ordering:

| Display label | Experiment ID | MPJPE (mm) | Unmatched persons |
|---|---|---:|---:|
| M6 Draft | `M6` | 167.573 | 54 |
| M9 Flow off | `M9_no_flow` | 167.732 | 82 |
| M9 1-step | `M9` | 168.086 | 76 |
| M9 2-step | `M9_RF2` | 165.487 | 72 |
| FW2 2-step | `T_FW2_20e_RF2` | 167.981 | 86 |
| FW2 4-step | `T_FW2_20e_RF4` | 169.454 | 99 |

## Layout

Use two aligned horizontal dot plots with shared categorical rows:

1. panel **(a)** shows overall MPJPE in millimeters;
2. panel **(b)** shows unmatched ground-truth person count.

The MPJPE axis uses a clearly labeled restricted range of 164.5--170.0 mm that
includes all six values and reveals their differences. The unmatched-person
axis spans 50--105 with modest padding around all observations. Because these
are dot plots rather than magnitude-encoding bars, neither axis needs to start
at zero. Exact values appear next to each point. The figure contains no
internal title because the LaTeX caption supplies it.

## Visual Style

- Use a white background, restrained horizontal guides, and no decorative box.
- Use neutral blue-gray for five configurations and Okabe-Ito blue for the
  selected `M9_RF2` operating point.
- Differentiate the selected point with both color and marker size/weight so
  the figure remains interpretable in grayscale.
- Use concise display labels rather than raw experiment IDs; Table IV retains
  the exact IDs and settings.
- Use sentence-case panel headings, panel labels **(a)** and **(b)**, and units
  on the axes.
- Export vector PDF and 300-DPI PNG for the current one-column manuscript at
  the `0.90\linewidth` placement used by `main.tex`.
- Do not add error bars: the source reports one deterministic checkpoint result
  per configuration and does not provide repeated-run uncertainty estimates.

## Manuscript Integration

Replace `fig_flow_solver_ablation.pdf` in place so the existing LaTeX include
continues to work. Refine the caption to state that lower values are better,
that the two panels report overall MPJPE and unmatched-person count, and that
the highlighted point is the selected two-step M9 evaluation. Do not infer
broad solver superiority beyond the tested configurations.

## Verification

1. Add focused tests for row order, values, concise labels, and output paths.
2. Regenerate the PDF and PNG from the existing logs.
3. Verify the PDF has no embedded raster-only chart and that the PNG is at
   least 300 DPI at the intended width.
4. Inspect the figure standalone and in the rebuilt manuscript.
5. Build with `pdflatex -> bibtex -> pdflatex -> pdflatex` and check for
   undefined references, missing graphics, fatal errors, and overflow.
