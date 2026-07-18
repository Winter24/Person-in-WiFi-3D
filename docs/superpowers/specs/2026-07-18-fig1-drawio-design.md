# Publication-ready Figure 1 draw.io design

## Goal

Create an editable diagrams.net/draw.io XML source for the full-width Figure 1 of the RESFES 2026 manuscript. The diagram uses the user-approved publication-balanced five-panel layout and is self-contained; the implementation must not depend on an external raster reference.

## Output

- Primary source: `paper_assets/manuscript_latex/resfes2026_witidar/figures/fig_system_overview.drawio`
- Required manuscript export: `paper_assets/manuscript_latex/resfes2026_witidar/figures/fig_system_overview.pdf`, matching the existing `main.tex` include. An SVG export may be retained as an optional editing/preview artefact.
- The XML must remain uncompressed and human-readable, with `compressed="false"` on the `mxfile` element and a direct `diagram > mxGraphModel` structure.
- Every panel, processing block, free annotation, arrow, query glyph, tensor glyph, and skeleton glyph must be an independently editable `mxCell`. A processing block may carry its own editable `value`; headings and free-standing annotations must use separate text cells.
- No raster image is embedded in the source.
- The source must export cleanly from diagrams.net to PDF and SVG.

## Canvas and layout

- Landscape canvas: exactly `1800 × 840` draw.io units.
- Panel geometry: `y=40`, `height=680`, with 20-unit outer margins and 20-unit inter-panel gaps:
  - CSI: `x=20`, `width=200`
  - tokenizer: `x=240`, `width=390`
  - encoder: `x=650`, `width=280`
  - query head: `x=950`, `width=350`
  - flow: `x=1320`, `width=460`
- Final output is placed below the flow panel at approximately `x=1450`, `y=750`, `width=300`, `height=60`; bypass and refined-pose paths must remain inside the 840-unit page.
- Five left-to-right functional panels with consistent spacing:
  1. Preprocessed CSI
  2. Motion-aware spectral tokenizer
  3. Flattened Mamba-2 sequence encoder
  4. Lightweight query-pose head
  5. Conditional two-step flow refinement
- Main data flow uses horizontal arrows; computations inside panels use vertical arrows.
- Person scores first perform score-based top-k ordering/selection (`K=100` under the selected configuration). Selected draft poses and conditions enter flow refinement, while their selected scores bypass only the velocity network and are reattached unchanged to the refined output. There is no second ranking operation after refinement.

## Scientific content

### Preprocessed CSI

- Input tensor: `B × 3 × 3 × 20 × 60`.
- State the structural interpretation `9 spatial groups × 20 time steps`.
- Use a simple editable stacked-plane glyph.

### Motion-aware spectral tokenizer

- Linear projection `60 → 256` producing `Z`.
- Temporal branch: depthwise temporal Conv1D with `k = 5`, followed by GELU, producing `Z-tilde`.
- Spectral-gate branch: `11-bin |RFFT(Z)|` along the 20-step temporal axis, mean over channels, gate MLP `11 → 22 → 20`, and sigmoid producing `g ∈ R^20`.
- The gate is spatial-group-specific: show `g_s ∈ R^20, s=1,…,9` (equivalently `g ∈ R^(9 × 20)` with batch omitted).
- Apply `g_s ⊙ Z-tilde_s`, then channel projection `W_c`.
- Residually add the projected gated branch to `Z`, apply LayerNorm, and output `Z_spec ∈ R^(9 × 20 × 256)`.

### Flattened Mamba-2 sequence encoder

- The tokenizer supplies a spatial-major flattened sequence of 180 tokens. Inside each repeated encoder block, annotate the time-major routing permutation `π_time`, Mamba-2 processing, and inverse restoration `π_time^-1` before residual addition.
- Input `H^0 ∈ R^(180 × 256)`.
- Show a repeated Mamba-2 residual block three times:
  - LayerNorm
  - `π_time` route
  - Mamba-2
  - `π_time^-1` restore
  - Dropout 0.1
  - Residual add
- Apply a final LayerNorm after the three repeated blocks and output `H ∈ R^(180 × 256)`.
- List `d_state = 64`, `d_conv = 4`, `expand = 2`, and `head dimension = 64`.

### Lightweight query-pose head

- 100 learned pose queries.
- 8-head cross-attention over encoded tokens, followed by learned-query residual addition and LayerNorm: `q_i = LN(q_i^learned + Attn(q_i^learned, H, H))`.
- Query features `q_i ∈ R^256`, `i = 1,…,100`.
- Three branches:
  - person-score head → logit `ell_i`, sigmoid probability `s_i = sigmoid(ell_i)`, then score-based selection
  - two-hidden-layer draft regressor → `p-hat_i^draft ∈ R^42`
  - query condition → `c_i = q_i ∈ R^256`

### Conditional two-step flow refinement

- Before flow, show score-ranked selection of all `K=100` query indices. The selected draft and condition tensors feed the velocity network; selected scores bypass it unchanged.
- Three inputs to each velocity evaluation:
  - evolving pose state `x_t ∈ R^42`, initialized as `x_0 = p-hat_i^draft`
  - query condition `c_i ∈ R^256`
  - time embedding `e_t ∈ R^64`
- Concatenate dimensions: `42 + 256 + 64 = 362`.
- Input projection `362 → 512`.
- Three residual MLP blocks, LayerNorm, and velocity projection `512 → 42`.
- Two Euler steps with `Δt = 0.5`, shown explicitly:
  - `x_0.5 = x_0 + 0.5 v_theta(x_0, 0, c_i)`
  - `x_1 = x_0.5 + 0.5 v_theta(x_0.5, 0.5, c_i)`
- The same VelocityMLP is evaluated first on `x_0` at `t=0` and then on the updated state `x_0.5` at `t=0.5`.
- Output refined pose `p-hat_i^ref ∈ R^42`.
- Reattach the already selected, unchanged query scores to the refined poses and label the result `Score-ranked refined 3-D poses`. Do not depict a second score-ranking operation after flow.

## Visual system

- White background and professional sans-serif typography.
- Colorblind-safe functional colors:
  - CSI: `#D55E00`
  - spectral tokenizer: `#E69F00`
  - Mamba-2 encoder: `#009E73`
  - query-pose head: `#0072B2`
  - flow refinement: `#CC79A7`
- Color is redundant with panel titles and grouping; the diagram remains understandable in grayscale.
- Outer panel strokes use 2 draw.io units and remain visually distinct from internal processing boxes.
- Internal blocks use white fill and 1.5-unit strokes.
- Main data-flow arrows use 2.5-unit strokes; secondary/bypass paths use 1.75-unit strokes.
- Source typography uses Arial: 34-unit bold panel headings, 27-unit main block labels, 25-unit formulas/secondary labels, and no text below 25 units. On the 1800-unit canvas scaled to the manuscript's approximately 506-pt display width, 25-unit text is approximately 7 pt.
- Export with `crop` disabled so the exact `1800 × 840` page box is retained, then include `fig_system_overview.pdf` at the existing `0.98\linewidth` width. Verify the exported PDF at the actual compiled manuscript width; all labels must remain at least 7 pt.

## Scope

- Figure 1 depicts inference architecture only.
- Training-only Gaussian perturbation, sampled interpolation time, Hungarian positives, target pose, and flow loss remain in Figure 2 and Methods.
- Do not include internal experiment IDs, configuration paths, checkpoint names, or implementation-only class names.

## Validation

- XML parses successfully with `mxfile compressed="false"`, one `diagram`, and a direct `mxGraphModel` child.
- All `mxCell` IDs are unique; every non-root `parent`, edge `source`, and edge `target` references an existing cell.
- The following key-label manifest is validated by normalized-text matching; each item must occur exactly once unless a different count is stated:
  - five panel titles: `Preprocessed CSI`, `Motion-aware spectral tokenizer`, `Flattened Mamba-2 sequence encoder`, `Lightweight query-pose head`, and `Conditional two-step flow refinement`
  - `B × 3 × 3 × 20 × 60`
  - `9 spatial groups × 20 time steps`
  - `Linear projection 60 → 256`
  - `Depthwise temporal Conv1D, k = 5`
  - `11-bin |RFFT(Z)|`
  - `Gate MLP 11 → 22 → 20`
  - `g_s ∈ R^20, s = 1,…,9`
  - `Channel projection W_c`
  - `Z_spec ∈ R^(9 × 20 × 256)`
  - `π_time` and `π_time^-1`
  - `Mamba-2 block × 3`
  - `Final LayerNorm`
  - `H ∈ R^(180 × 256)`
  - `100 learned pose queries`
  - `8-head cross-attention`
  - `q_i = LN(q_i^learned + Attn(q_i^learned, H, H))`
  - `s_i = sigmoid(ell_i)`
  - `Score-ranked top-k selection, K = 100`
  - `p-hat_i^draft ∈ R^42`
  - `c_i = q_i ∈ R^256`
  - `42 + 256 + 64 = 362`
  - `Input projection 362 → 512`
  - `Three residual MLP blocks`
  - `Velocity projection 512 → 42`
  - `x_0.5 = x_0 + 0.5 v_theta(x_0, 0, c_i)`
  - `x_1 = x_0.5 + 0.5 v_theta(x_0.5, 0.5, c_i)`
  - `Score-ranked refined 3-D poses`
- Generic repeated words such as `LayerNorm`, `Mamba-2`, `score`, and tensor symbols are excluded from exact occurrence counting unless included in the manifest phrase above.
- The source contains no raster `image=` elements or embedded base64 data.
- Verify the page geometry is exactly `1800 × 840`, all content remains inside the page, and required source font sizes/stroke widths are used.
- Open in diagrams.net and visually inspect at full view and manuscript-scale reduction.
- Export to PDF/SVG and verify text, arrows, and panel boundaries are not clipped or overlapping.
