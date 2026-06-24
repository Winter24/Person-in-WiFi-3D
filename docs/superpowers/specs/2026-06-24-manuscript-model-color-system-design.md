# Manuscript Model Color System Design

## Objective

Use one stable, colorblind-conscious identity for every M0--M9/M9_RF2 model
where model identity is encoded in the manuscript. Preserve the current
full-paper palette so existing figures retain their visual meaning, while
eliminating cases where the same model is gray or receives a different color
in another figure or table.

## Canonical Main-Model Palette

The existing colors in `plot_full_paper_figures.py` become canonical:

| Model | Hex color | Role |
|---|---|---|
| M0 | `#D55E00` | Original baseline |
| M1 | `#8C9AA9` | Spectral PETR baseline |
| M2 | `#A7B6C2` | PETR with Mamba-1 |
| M3 | `#6B8DB5` | PETR with Mamba2-CSI |
| M4 | `#E69F00` | WiTiDAR draft with Transformer |
| M5 | `#F0C04A` | WiTiDAR draft with Mamba-1 |
| M6 | `#7CB342` | Compact Mamba2-CSI draft |
| M7 | `#56B4E9` | Transformer-flow model |
| M8 | `#CC79A7` | Mamba-1-flow model |
| M9 | `#0072B2` | One-step Mamba2-CSI flow model |
| M9_RF2 | `#009E73` | Selected two-step evaluation |

The palette is centralized in one Python module and imported by every plotting
or qualitative-rendering script that encodes model identity. The same module
generates the tracked LaTeX color-definition file, and tests verify exact
Python--LaTeX parity. No plotting script or handwritten LaTeX table may declare
its own model hex values.

For plots that need redundant encoding, the shared module also defines a stable
marker and line-style mapping. Main model IDs use distinct marker/line-style
combinations in bubble and radar plots; categorical bar charts retain printed
x-axis IDs, so bar color is supplementary rather than the only encoding.

## Derived Flow-Solver Encoding

Figure 6 contains solver settings in addition to main model IDs. Color denotes
the originating model/checkpoint family; marker shape and fill distinguish the
inference setting:

| Setting | Color source | Marker semantics |
|---|---|---|
| M6 draft | M6 green | filled circle |
| M9 flow disabled | M9 blue | open circle |
| M9 one-step | M9 blue | filled circle |
| M9 two-step / M9_RF2 | M9_RF2 green | filled diamond, larger/bold |
| FW2 two-step | supplemental neutral gray | filled square |
| FW2 four-step | supplemental neutral gray | filled triangle |

The FW2 settings are retrained supplemental controls rather than M8 variants,
so they must not reuse M8 magenta. A named supplemental neutral from the shared
style module avoids a false model-family association. Marker shape preserves
their distinction in grayscale.

## Figure Scope

Apply the canonical palette to figures where color means model identity:

- full M0--M9/M9_RF2 four-metric ablation;
- accuracy--efficiency bubble chart;
- multi-metric radar chart;
- flow-solver analysis;
- model-column headings in the qualitative renderer.

Do not recolor figures where color has another semantic meaning:

- system overview and flow-refinement diagrams use colors for modules/dataflow;
- qualitative skeleton colors identify matched persons across columns;
- unmatched qualitative predictions remain gray.

The qualitative renderer keeps heading text black and adds a short colored
rule/marker to the M0, M7, and M9_RF2 column headings. This maintains contrast
for light colors such as M7 blue. It must not recolor skeletons by model.

## Table Scope

Use a small colored square followed by a black bold model ID. Do not color the
entire row and do not render light model colors as body text, because M2 and M5
would lose contrast on white.

Apply model markers to:

- the model variant summary table;
- the full ablation results table;
- the supplementary flow table;
- M0 and M9_RF2 headers in the per-joint comparison;
- the M9_RF2 model label/header in receiver-subset diagnostics when present.

Dataset statistics and non-model categorical labels remain uncolored.

LaTeX color definitions and marker macros live in a generated dedicated input
file. The canonical Python module writes this file deterministically, including
a generated-file notice. Table values, bold best-value formatting, and row
order remain unchanged.

## Accessibility and Print Behavior

- Color is never the only encoding: IDs remain printed, and solver settings use
  distinct markers/fills.
- Table text remains black; color is confined to the adjacent square marker.
- Figures retain explicit model labels or legends.
- The selected M9_RF2 model may use a stronger outline/weight in addition to
  green.
- Grayscale previews must preserve model/solver distinction through labels,
  line width, marker shape, or fill.

## Implementation Boundaries

- No metric, model role, table ordering, caption claim, or result value changes.
- No recoloring of RGB images, pose-person identities, unmatched poses, or
  architecture-module semantics.
- Existing PDF/PNG output names and LaTeX figure references remain stable.
- The qualitative full-paper figure is not regenerated locally; only its
  server-side renderer is updated for future output.

## Verification

1. Unit-test the exact canonical mapping and all required main model IDs.
2. Test deterministic LaTeX generation and exact Python--LaTeX hex parity.
3. Test that plotting data obtains colors from the shared palette rather than
   embedded tuple values.
4. Test flow-setting colors and marker semantics, including neutral FW2
   controls.
5. Test qualitative model heading accents without changing person colors.
6. Regenerate the main ablation, bubble, radar, and flow-solver PDF/PNG assets.
7. Verify figures at manuscript scale and in grayscale.
8. Build LaTeX with `pdflatex -> bibtex -> pdflatex -> pdflatex`.
9. Check for undefined references/citations, missing graphics, fatal errors,
   overflow, and unreadable table markers.
10. Diff all tables and numeric statements to confirm values and row order are
   unchanged.
