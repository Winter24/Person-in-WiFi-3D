# Publication-ready Figure 1 draw.io Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a scientifically accurate, fully editable diagrams.net XML source for the manuscript's full-width Figure 1 and validate its structure, data flow, labels, geometry, and publication-scale typography.

**Architecture:** A small deterministic Python builder creates a human-readable, uncompressed `mxfile` from fixed panel/block/edge definitions. A focused pytest contract parses the generated XML cell-by-cell, validates exact scientific labels and semantic connections, and checks publication styles and page bounds. The accepted XML is then opened in diagrams.net for visual QA and manually exported to the PDF already referenced by the manuscript.

**Tech Stack:** Python standard library (`xml.etree.ElementTree`), diagrams.net/draw.io `mxGraphModel`, pytest, BibTeX/LaTeX.

---

## Files and responsibilities

- Create `tools/analysis/build_fig1_drawio.py`: deterministic XML builder; contains all coordinates, styles, labels, and edge definitions.
- Create `tests/test_fig1_drawio.py`: structural, scientific-semantic, geometry, typography, color, and no-raster contract.
- Generate `paper_assets/manuscript_latex/resfes2026_witidar/figures/fig_system_overview.drawio`: primary editable output.
- Preserve `paper_assets/manuscript_latex/resfes2026_witidar/figures/fig_system_overview.pdf` until a user-confirmed final export.
- Locked specification: `docs/superpowers/specs/2026-07-18-fig1-drawio-design.md`.

## Shared test data

The test module must define the complete exact-label manifest as a set. Labels are normalized cell-by-cell; do not count substrings in one joined corpus.

```python
REQUIRED_LABELS = {
    "Preprocessed CSI",
    "Motion-aware spectral tokenizer",
    "Flattened Mamba-2 sequence encoder",
    "Lightweight query-pose head",
    "Conditional two-step flow refinement",
    "B × 3 × 3 × 20 × 60",
    "9 spatial groups × 20 time steps",
    "Linear projection 60 → 256",
    "Depthwise temporal Conv1D, k = 5",
    "GELU",
    "Z-tilde",
    "11-bin |RFFT(Z)|",
    "Mean over channels",
    "Gate MLP 11 → 22 → 20",
    "Sigmoid",
    "g_s ∈ R^20, s = 1,…,9",
    "g_s ⊙ Z-tilde_s",
    "Channel projection W_c",
    "Z_spec ∈ R^(9 × 20 × 256)",
    "Spatial-major input: 9 × 20 = 180 tokens",
    "H^0 ∈ R^(180 × 256)",
    "π_time",
    "π_time^-1",
    "Mamba-2 block × 3",
    "Dropout 0.1",
    "d_state = 64; d_conv = 4; expand = 2; head dimension = 64",
    "Final LayerNorm",
    "H ∈ R^(180 × 256)",
    "100 learned pose queries",
    "8-head cross-attention",
    "q_i ∈ R^256, i = 1,…,100",
    "q_i = LN(q_i^learned + Attn(q_i^learned, H, H))",
    "Person-score head",
    "ell_i",
    "Two-hidden-layer draft regressor",
    "Query condition",
    "s_i = sigmoid(ell_i)",
    "Score-ranked top-k selection, K = 100",
    "p-hat_i^draft ∈ R^42",
    "c_i = q_i ∈ R^256",
    "Selected draft pose",
    "Selected query condition",
    "x_t ∈ R^42",
    "x_0 = p-hat_i^draft",
    "e_t ∈ R^64",
    "42 + 256 + 64 = 362",
    "Input projection 362 → 512",
    "Three residual MLP blocks",
    "Velocity projection 512 → 42",
    "x_0.5 = x_0 + 0.5 v_theta(x_0, 0, c_i)",
    "x_1 = x_0.5 + 0.5 v_theta(x_0.5, 0.5, c_i)",
    "Shared VelocityMLP at t = 0 and t = 0.5",
    "Two Euler steps, Δt = 0.5",
    "unchanged selected scores",
    "p-hat_i^ref = x_1 ∈ R^42",
    "Score-ranked refined 3-D poses",
}
```

The builder must use these stable IDs for semantic tests:

```text
input_tensor, linear_projection, temporal_conv, temporal_gelu,
rfft_magnitude, channel_mean, gate_mlp, gate_sigmoid, gate_product,
channel_projection, tokenizer_residual, tokenizer_norm, z_spec,
spatial_major, encoder_norm, route_time, mamba2, restore_time,
encoder_dropout, encoder_residual, final_encoder_norm, encoded_h,
learned_queries, cross_attention, query_residual_norm,
score_head, score_logit, score_sigmoid, score_topk, draft_head, draft_pose,
condition_head, condition, selected_draft, selected_condition,
flow_state, time_embedding, concatenate,
flow_input_projection, flow_blocks, flow_norm, velocity_projection,
euler_step_1, euler_step_2, refined_pose, final_output
```

## Fixed internal geometry and editable glyph contract

The builder uses these absolute `(x, y, width, height)` rectangles. Minor text wrapping may not change the rectangles without updating the locked tests.

```python
CELL_LAYOUT = {
    "input_tensor": (55, 190, 130, 250),
    "linear_projection": (335, 85, 200, 50),
    "temporal_conv": (265, 160, 155, 60),
    "temporal_gelu": (290, 235, 105, 42),
    "rfft_magnitude": (450, 150, 155, 65),
    "channel_mean": (465, 230, 125, 42),
    "gate_mlp": (445, 290, 165, 50),
    "gate_sigmoid": (470, 355, 115, 42),
    "gate_product": (350, 420, 175, 45),
    "channel_projection": (350, 485, 175, 45),
    "tokenizer_residual": (350, 550, 175, 45),
    "tokenizer_norm": (350, 615, 175, 42),
    "z_spec": (300, 670, 275, 40),
    "spatial_major": (655, 670, 270, 40),
    "encoder_norm": (695, 190, 190, 45),
    "route_time": (695, 250, 190, 45),
    "mamba2": (695, 310, 190, 50),
    "restore_time": (695, 375, 190, 45),
    "encoder_dropout": (695, 435, 190, 45),
    "encoder_residual": (695, 495, 190, 45),
    "final_encoder_norm": (695, 575, 190, 45),
    "encoded_h": (680, 640, 220, 45),
    "learned_queries": (1015, 90, 220, 65),
    "cross_attention": (1015, 185, 220, 55),
    "query_residual_norm": (990, 270, 270, 70),
    "score_head": (965, 390, 100, 60),
    "draft_head": (1075, 390, 115, 60),
    "condition_head": (1200, 390, 90, 60),
    "score_logit": (965, 470, 100, 35),
    "score_sigmoid": (965, 520, 100, 45),
    "score_topk": (955, 585, 120, 75),
    "draft_pose": (1080, 480, 105, 65),
    "condition": (1195, 480, 100, 65),
    "selected_draft": (1080, 585, 105, 60),
    "selected_condition": (1195, 585, 100, 60),
    "flow_state": (1335, 100, 135, 70),
    "time_embedding": (1335, 260, 135, 60),
    "concatenate": (1495, 150, 225, 70),
    "flow_input_projection": (1515, 245, 185, 55),
    "flow_blocks": (1515, 320, 185, 60),
    "flow_norm": (1515, 400, 185, 50),
    "velocity_projection": (1515, 470, 185, 55),
    "euler_step_1": (1350, 550, 390, 55),
    "euler_step_2": (1350, 620, 390, 55),
    "refined_pose": (1450, 685, 220, 45),
    "final_output": (1450, 760, 300, 60),
}
```

All tokenizer content, including `z_spec`, remains inside its panel ending at `y=720`. `z_spec` and `spatial_major` share center line `y=690`, giving a horizontal tokenizer-to-encoder stage edge. The final output is the only deliberate below-panel process box and remains within the `1800 × 840` page.

Editable glyph requirements:

- CSI stack: exactly seven polygon/rectangle cells with IDs `csi_plane_01` through `csi_plane_07`.
- Query grid: exactly twelve square cells with IDs `query_cell_01` through `query_cell_12`.
- Token/tensor glyphs: exactly eight slim rectangle cells with IDs `token_bar_01` through `token_bar_08`.
- Draft skeleton: 14 circular joint cells `draft_joint_01..14` and 13 bone edges `draft_bone_01..13`.
- Refined skeleton: 14 circular joint cells `refined_joint_01..14` and 13 bone edges `refined_bone_01..13`.
- Tests count each prefix/range, ensure unique IDs, and confirm every glyph is a separate editable `mxCell`; no grouped raster or `image` style is allowed.

The builder derives glyph coordinates deterministically:

```python
# CSI planes
for i in range(7):
    add_vertex(..., f"csi_plane_{i + 1:02d}", "",
               52 + 10 * i, 245 - 5 * i, 82, 165, GLYPH_STYLE)

# 4 × 3 query grid
for row in range(3):
    for col in range(4):
        i = row * 4 + col + 1
        add_vertex(..., f"query_cell_{i:02d}", "",
                   1065 + 22 * col, 115 + 18 * row, 16, 14, QUERY_STYLE)

# encoded-token bars
for i in range(8):
    add_vertex(..., f"token_bar_{i + 1:02d}", "",
               705 + 14 * i, 652, 9, 28, TOKEN_STYLE)

POSE_POINTS = [
    (0, 0), (0, 20), (-18, 30), (18, 30), (-28, 55), (28, 55),
    (-38, 80), (38, 80), (-12, 70), (12, 70), (-14, 105),
    (14, 105), (-15, 140), (15, 140),
]
POSE_BONES = [
    (0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7),
    (1, 8), (1, 9), (8, 10), (9, 11), (10, 12), (11, 13),
]
```

Use `add_pose_glyph(prefix, anchor_x, anchor_y, scale, color)` with `(1130, 495, 0.34, #D81B60)` for the draft glyph and `(1715, 660, 0.38, #0072B2)` for the refined glyph. Each joint and bone receives its own stable ID.

Separate panel headings and free annotations use these fixed cells:

```python
HEADING_LAYOUT = {
    "title_csi": (25, 48, 190, 42),
    "title_tokenizer": (255, 48, 360, 42),
    "title_encoder": (660, 48, 260, 42),
    "title_head": (960, 48, 330, 42),
    "title_flow": (1330, 48, 440, 42),
}
ANNOTATION_LAYOUT = {
    "input_shape_label": (35, 105, 170, 55),
    "spatial_groups_label": (35, 455, 170, 65),
    "z_label": (410, 135, 50, 30),
    "z_tilde_label": (300, 277, 90, 30),
    "gate_shape_label": (460, 397, 145, 45),
    "gate_product_label": (360, 465, 155, 35),
    "h0_label": (670, 625, 240, 35),
    "encoder_repeat_label": (665, 150, 250, 30),
    "encoder_hparams_label": (665, 95, 250, 50),
    "query_feature_label": (980, 345, 290, 35),
    "query_equation_label": (965, 675, 320, 40),
    "flow_state_dim_label": (1330, 175, 150, 35),
    "shared_velocity_label": (1335, 485, 175, 55),
    "euler_dt_label": (1335, 525, 170, 35),
    "score_bypass_label": (1120, 770, 280, 35),
    "refined_pose_label": (1660, 690, 110, 40),
}
```

Resolve any annotation overlap during implementation by using the listed rectangles as test-locked anchors and wrapping within them; do not move them without updating the corresponding test. `query_feature_label` carries `q_i ∈ R^256, i=1,…,100`; `query_equation_label` carries the residual-normalization equation.

Required connector routes use explicit waypoints so the bounds test covers them:

```python
EDGE_POINTS = {
    "edge_zspec_to_spatial": [(630, 690), (650, 690)],
    "edge_h_to_attention": [(930, 662), (945, 662), (945, 212), (1015, 212)],
    "edge_encoder_skip": [(675, 690), (665, 690), (665, 517), (695, 517)],
    "edge_query_skip": [(1000, 122), (980, 122), (980, 305), (990, 305)],
    "edge_topk_selected_draft": [(1075, 622), (1080, 622)],
    "edge_topk_selected_condition": [(1075, 635), (1195, 635)],
    "edge_score_bypass": [(1015, 660), (1015, 790), (1450, 790)],
    "edge_refined_output": [(1560, 730), (1560, 760)],
}
```

All other internal edges are orthogonal direct connections between fixed cells and require no additional waypoint.

### Task 1: Establish dirty-worktree safety

**Files:**
- Inspect only: target builder, test, draw.io, PDF, and manuscript paths.

- [ ] **Step 1: Run scoped status preflight**

```powershell
git status --short -- `
  tools/analysis/build_fig1_drawio.py `
  tests/test_fig1_drawio.py `
  paper_assets/manuscript_latex/resfes2026_witidar/figures/fig_system_overview.drawio `
  paper_assets/manuscript_latex/resfes2026_witidar/figures/fig_system_overview.pdf `
  paper_assets/manuscript_latex/resfes2026_witidar/main.tex
```

Record the result. The PDF and `main.tex` already have user-owned/untracked changes in the current worktree. Do not overwrite, stage, or restore either file during Tasks 1–7.

- [ ] **Step 2: Confirm the two new code paths are free**

Run:

```powershell
Test-Path tools/analysis/build_fig1_drawio.py
Test-Path tests/test_fig1_drawio.py
Test-Path paper_assets/manuscript_latex/resfes2026_witidar/figures/fig_system_overview.drawio
```

Expected: `False`, `False`, `False`. If any is `True`, stop and inspect rather than overwrite.

### Task 2: Write the failing XML contract

**Files:**
- Create: `tests/test_fig1_drawio.py`

- [ ] **Step 1: Add parser and cell-by-cell normalization**

```python
from collections import Counter
from pathlib import Path
import html
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).parents[1]
DRAWIO = ROOT / "paper_assets/manuscript_latex/resfes2026_witidar/figures/fig_system_overview.drawio"


def normalize(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    return " ".join(value.split())


def load_graph():
    root = ET.parse(DRAWIO).getroot()
    assert root.tag == "mxfile"
    assert root.attrib.get("compressed") == "false"
    diagrams = root.findall("diagram")
    assert len(diagrams) == 1
    graph = diagrams[0].find("mxGraphModel")
    assert graph is not None
    cells = graph.findall("./root/mxCell")
    return root, graph, cells
```

- [ ] **Step 2: Add structure and reference tests**

Require page `1800 × 840`, unique IDs, and valid `parent`, `source`, and `target` references.

- [ ] **Step 3: Add exact-label test**

```python
def test_required_labels_are_exact_cells():
    _, _, cells = load_graph()
    counts = Counter(normalize(c.attrib.get("value", "")) for c in cells)
    for label in REQUIRED_LABELS:
        assert counts[label] == 1, (label, counts[label])
```

This avoids counting `π_time` as a substring of `π_time^-1`.

- [ ] **Step 4: Add semantic-edge contract**

Build `{edge_id: (source, target)}` and require these connections:

```python
REQUIRED_EDGES = {
    ("input_tensor", "linear_projection"),
    ("linear_projection", "temporal_conv"),
    ("temporal_conv", "temporal_gelu"),
    ("linear_projection", "rfft_magnitude"),
    ("rfft_magnitude", "channel_mean"),
    ("channel_mean", "gate_mlp"),
    ("gate_mlp", "gate_sigmoid"),
    ("temporal_gelu", "gate_product"),
    ("gate_sigmoid", "gate_product"),
    ("gate_product", "channel_projection"),
    ("linear_projection", "tokenizer_residual"),
    ("channel_projection", "tokenizer_residual"),
    ("tokenizer_residual", "tokenizer_norm"),
    ("tokenizer_norm", "z_spec"),
    ("spatial_major", "encoder_norm"),
    ("encoder_norm", "route_time"),
    ("route_time", "mamba2"),
    ("mamba2", "restore_time"),
    ("restore_time", "encoder_dropout"),
    ("encoder_dropout", "encoder_residual"),
    ("spatial_major", "encoder_residual"),
    ("encoder_residual", "final_encoder_norm"),
    ("final_encoder_norm", "encoded_h"),
    ("learned_queries", "cross_attention"),
    ("encoded_h", "cross_attention"),
    ("cross_attention", "query_residual_norm"),
    ("learned_queries", "query_residual_norm"),
    ("query_residual_norm", "score_head"),
    ("score_head", "score_logit"),
    ("score_logit", "score_sigmoid"),
    ("score_sigmoid", "score_topk"),
    ("query_residual_norm", "draft_head"),
    ("draft_head", "draft_pose"),
    ("query_residual_norm", "condition_head"),
    ("condition_head", "condition"),
    ("score_topk", "selected_draft"),
    ("draft_pose", "selected_draft"),
    ("score_topk", "selected_condition"),
    ("condition", "selected_condition"),
    ("selected_draft", "flow_state"),
    ("flow_state", "concatenate"),
    ("selected_condition", "concatenate"),
    ("time_embedding", "concatenate"),
    ("concatenate", "flow_input_projection"),
    ("flow_input_projection", "flow_blocks"),
    ("flow_blocks", "flow_norm"),
    ("flow_norm", "velocity_projection"),
    ("velocity_projection", "euler_step_1"),
    ("euler_step_1", "euler_step_2"),
    ("euler_step_2", "refined_pose"),
    ("refined_pose", "final_output"),
    ("score_topk", "final_output"),
}
```

The two edges from `score_topk` to `selected_draft` and `selected_condition` represent the selected top-k indices, not score values. Assert that the only score-value bypass is `score_topk → final_output`, that it never enters `flow_input_projection`, and that `final_output` has no outgoing ranking edge. Assert the true residual skip edges `spatial_major → encoder_residual` and `learned_queries → query_residual_norm` separately from the sequential computation edges.

Partition the contract into independently runnable tests so each implementation slice reaches green:

```text
test_tokenizer_labels_and_edges
test_encoder_labels_edges_and_residual
test_query_head_labels_selection_and_residual
test_flow_labels_edges_and_score_bypass
test_complete_manifest_once
```

The four subsystem tests use subsystem-specific subsets of `REQUIRED_LABELS` and `REQUIRED_EDGES`. Only `test_complete_manifest_once` uses the full sets and is run after Task 7.

- [ ] **Step 5: Add scientific-presence and absence assertions**

Require the encoder hyperparameter label, spatial gate, mean, sigmoid, two-hidden-layer draft regressor, shared VelocityMLP, both evolving-state Euler equations, and unchanged selected-score label. Reject training-only text and post-flow ranking terms:

```python
FORBIDDEN = {
    "Gaussian perturbation", "Hungarian positives", "flow loss",
    "target pose", "t ~ U[0,1]", "sampled interpolation time",
    "post-flow ranking", "re-ranked after refinement", "checkpoint",
    "config path", "WiTiDARHead", "VelocityMLP class",
}
```

- [ ] **Step 6: Add style/no-raster/bounds/editability tests**

Assert over the entire raw XML text that `image=` and `data:image` are absent. For every text-bearing cell require `fontFamily=Arial` and numeric `fontSize >= 25`; panel titles require `fontSize=34` and `fontStyle=1`. Validate all five functional colors and exact panel rectangles. Internal process boxes require `strokeWidth=1.5`; main edges `2.5`; both top-k index edges, both residual skip edges, and the score bypass require `strokeWidth=1.75`. The four index/skip edges must be dashed; the score bypass must be solid. Validate every `CELL_LAYOUT`, `HEADING_LAYOUT`, and `ANNOTATION_LAYOUT` rectangle, every absolute vertex, and every `EDGE_POINTS` waypoint lies within `[0,1800] × [0,840]`; relative geometries are permitted only for edge labels and must have an absolute `mxPoint` offset inside the page. Count the editable glyph ID ranges exactly: 7 CSI planes, 12 query cells, 8 token bars, 28 pose joints, and 26 pose-bone edges.

- [ ] **Step 7: Run the tests and confirm the expected red state**

```powershell
pytest tests/test_fig1_drawio.py -v
```

Expected: FAIL because the draw.io output does not exist.

- [ ] **Step 8: Commit only the test contract**

```powershell
git add -- tests/test_fig1_drawio.py
git diff --cached --name-only
git diff --cached --check
git commit -m "test: define Figure 1 drawio contract"
```

Expected staged name: only `tests/test_fig1_drawio.py`.

### Task 3: Implement the deterministic XML builder core

**Files:**
- Create: `tools/analysis/build_fig1_drawio.py`
- Generate: `paper_assets/manuscript_latex/resfes2026_witidar/figures/fig_system_overview.drawio`
- Test: `tests/test_fig1_drawio.py`

- [ ] **Step 1: Implement XML helpers**

Use `ElementTree` and these interfaces:

```python
def style(**parts) -> str: ...
def add_vertex(root, cell_id, value, x, y, width, height, cell_style, parent="1"): ...
def add_edge(root, edge_id, source, target, edge_style, points=()): ...
def add_text(root, cell_id, value, x, y, width, height, font_size=25, bold=False): ...
def build_graph() -> ET.ElementTree: ...
def main() -> int: ...
```

`main()` writes UTF-8 XML with declaration, calls `ET.indent(tree, space="  ")`, and writes only to the exact draw.io output path. Do not touch the PDF or `main.tex`.

- [ ] **Step 2: Implement shared styles and five panels**

Use exact canvas/panel geometry from the spec. Define:

```python
COLORS = {
    "csi": "#D55E00", "tokenizer": "#E69F00",
    "encoder": "#009E73", "head": "#0072B2", "flow": "#CC79A7",
}
PANEL_STYLE = "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeWidth=2;"
BOX_STYLE = "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#444444;strokeWidth=1.5;fontFamily=Arial;fontSize=27;"
MAIN_EDGE = "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeColor=#333333;strokeWidth=2.5;"
SECONDARY_EDGE = "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeColor=#666666;strokeWidth=1.75;dashed=1;dashPattern=6 4;"
BYPASS_EDGE = "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeColor=#555555;strokeWidth=1.75;"
```

Use `SECONDARY_EDGE` for the two top-k index edges and the two residual skip edges. Use `BYPASS_EDGE` only for unchanged selected scores. Tests require `strokeWidth=1.75` on all five secondary/bypass paths, `dashed=1` on index/skip paths, and no dashed style on the score-value bypass.

- [ ] **Step 3: Generate and run the structural/style slice**

```powershell
python tools/analysis/build_fig1_drawio.py
pytest tests/test_fig1_drawio.py -k "uncompressed or references or panel or typography or raster" -v
```

Expected: structural/style tests PASS; scientific-content tests remain FAIL.

### Task 4: Add CSI and spectral-tokenizer slice

**Files:** same builder/output/test files.

- [ ] **Step 1: Add exact CSI/tokenizer IDs, labels, blocks, and edges**

Place the input glyph inside the CSI panel. In the tokenizer panel use two branches after `linear_projection`, join at `gate_product`, then `channel_projection`, residual add, LayerNorm, and `z_spec`. Add a separate label `g_s ∈ R^20, s = 1,…,9`; do not depict a single global gate.

- [ ] **Step 2: Generate and run tokenizer tests**

```powershell
python tools/analysis/build_fig1_drawio.py
pytest tests/test_fig1_drawio.py::test_tokenizer_labels_and_edges -v
```

Expected: tokenizer-specific assertions PASS. Do not run the full manifest yet.

### Task 5: Add routed Mamba-2 encoder slice

**Files:** same builder/output/test files.

- [ ] **Step 1: Add encoder block and semantics**

Add `spatial_major`, then the repeated block sequence `encoder_norm → route_time → mamba2 → restore_time → encoder_dropout → encoder_residual`, plus a separate skip edge `spatial_major → encoder_residual`. Annotate `Mamba-2 block × 3`, then `final_encoder_norm → encoded_h`. Include `H^0`, `Dropout 0.1`, and the full hyperparameter label. Use one representative repeated block with `×3`, so `π_time` and `π_time^-1` each occur once.

- [ ] **Step 2: Generate and run encoder tests**

```powershell
python tools/analysis/build_fig1_drawio.py
pytest tests/test_fig1_drawio.py::test_encoder_labels_edges_and_residual -v
```

Expected: encoder semantic assertions PASS.

### Task 6: Add query head and score-selection slice

**Files:** same builder/output/test files.

- [ ] **Step 1: Add learned queries and residual-normalized cross-attention**

Add `learned_queries → cross_attention → query_residual_norm`; connect `encoded_h → cross_attention`, and add the true learned-query skip edge `learned_queries → query_residual_norm`. Include both `q_i ∈ R^256, i=1,…,100` and the complete residual-normalization equation.

- [ ] **Step 2: Add the three branches and index-selected tensors**

Add score logit/sigmoid/top-k cells, two-hidden-layer draft regressor and 42-D draft, and query condition with `c_i=q_i`. Add `selected_draft` and `selected_condition`; each receives one tensor edge from its branch and one dashed top-k-index edge from `score_topk`. This makes score ordering occur before flow without misrepresenting score values as pose/condition inputs.

- [ ] **Step 3: Generate and run query-head tests**

```powershell
python tools/analysis/build_fig1_drawio.py
pytest tests/test_fig1_drawio.py::test_query_head_labels_selection_and_residual -v
```

Expected: query and score-ordering assertions PASS.

### Task 7: Add two-step evolving-state flow and final output

**Files:** same builder/output/test files.

- [ ] **Step 1: Add flow inputs and VelocityMLP stack**

Add `flow_state` initialized by `selected_draft`, `selected_condition`, and `time_embedding` into `concatenate`, followed by `flow_input_projection → flow_blocks → flow_norm → velocity_projection`. Include `x_t ∈ R^42` and `Two Euler steps, Δt = 0.5` as separate annotations.

- [ ] **Step 2: Add the two explicit Euler cells**

Add `euler_step_1` using `x_0`, then `euler_step_2` using updated `x_0.5`. Include one `Shared VelocityMLP at t = 0 and t = 0.5` annotation. Do not duplicate the network graph.

- [ ] **Step 3: Add output and bypass semantics**

Connect `refined_pose → final_output` with a main edge. Connect `score_topk → final_output` with the one `BYPASS_EDGE` and a separate edge-label cell `unchanged selected scores`. No edge leaves `final_output`; no score edge enters the velocity network.

- [ ] **Step 4: Add all independently editable glyph cells**

Generate the exact CSI planes, query grid, token bars, and two pose skeletons from the fixed loops and point/bone arrays above. Run the glyph-count/editability test and require PASS before the full suite.

- [ ] **Step 5: Generate and run the full test contract**

```powershell
python tools/analysis/build_fig1_drawio.py
pytest tests/test_fig1_drawio.py::test_flow_labels_edges_and_score_bypass -v
pytest tests/test_fig1_drawio.py::test_editable_glyph_counts -v
pytest tests/test_fig1_drawio.py -v
```

Expected: the flow subsystem test PASS first, then all tests including `test_complete_manifest_once` PASS.

- [ ] **Step 6: Inspect and commit only builder/test/XML**

```powershell
git add -- `
  tools/analysis/build_fig1_drawio.py `
  tests/test_fig1_drawio.py `
  paper_assets/manuscript_latex/resfes2026_witidar/figures/fig_system_overview.drawio
git diff --cached --name-only
git diff --cached --check
git commit -m "fig: add editable publication-ready system overview"
```

Expected staged names: exactly the three listed files. The existing PDF and `main.tex` must not be staged.

### Task 8: Visual QA and optional PDF replacement

**Files:**
- Verify: `paper_assets/manuscript_latex/resfes2026_witidar/figures/fig_system_overview.drawio`
- Potentially overwrite only after explicit confirmation: `paper_assets/manuscript_latex/resfes2026_witidar/figures/fig_system_overview.pdf`
- Verify without editing: `paper_assets/manuscript_latex/resfes2026_witidar/main.tex`

- [ ] **Step 1: Document the diagrams.net prerequisite**

No local `drawio`/`diagrams.net` CLI is currently installed. Open `https://app.diagrams.net/` or the diagrams.net desktop application, then use **File → Open From → Device** to open `fig_system_overview.drawio`.

- [ ] **Step 2: Perform visual QA before export**

Check every object is editable; no text/connector overlap; score top-k clearly occurs before flow; the bypass skips only VelocityMLP; the two Euler equations use `x_0` then `x_0.5`; and all text remains readable at approximately 28% view scale.

- [ ] **Step 3: Re-run scoped status and stop before destructive overwrite**

```powershell
git status --short -- `
  paper_assets/manuscript_latex/resfes2026_witidar/figures/fig_system_overview.pdf `
  paper_assets/manuscript_latex/resfes2026_witidar/main.tex
```

If the PDF is modified or untracked, show the status to the user and obtain explicit confirmation before replacing it. Never overwrite `main.tex` during this task.

- [ ] **Step 4: Export after confirmation**

In diagrams.net use **File → Export as → PDF**, disable crop, set border width to zero, export the diagram page only, and save to `paper_assets/manuscript_latex/resfes2026_witidar/figures/fig_system_overview.pdf`.

- [ ] **Step 5: Export and inspect SVG**

Use **File → Export as → SVG**, keep text as editable text, disable crop, and save temporarily as `paper_assets/manuscript_latex/resfes2026_witidar/figures/fig_system_overview.svg`. Open both PDF and SVG at 100% and manuscript-scale width; verify matching geometry, unclipped text, valid arrows, and no raster `<image>` elements. Retaining the SVG is optional after inspection; do not stage it unless the user requests it.

- [ ] **Step 6: Compile with exact commands**

From `paper_assets/manuscript_latex/resfes2026_witidar` run:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: all commands exit 0; final log contains no undefined citations/references, missing assets, or overfull boxes introduced by Figure 1.

- [ ] **Step 7: Run final focused tests**

```powershell
pytest tests/test_fig1_drawio.py tests/test_render_system_overview.py tests/test_paper_evidence.py -v
```

Expected: all selected tests PASS.

- [ ] **Step 8: Stage only the confirmed export and inspect it**

```powershell
git add -- paper_assets/manuscript_latex/resfes2026_witidar/figures/fig_system_overview.pdf
git diff --cached --name-only
git diff --cached --stat
```

Commit only when the staged name list contains the intended PDF and no user-owned manuscript files.
