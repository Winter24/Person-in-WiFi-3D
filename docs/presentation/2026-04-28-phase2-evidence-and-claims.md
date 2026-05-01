# Phase 2 Evidence and Claims — ResFes 2026

This note finalizes the Phase 2 outputs for the presentation based on the final proposal [NTN_IT_CT.pdf](D:/Resfes_2026/Person-in-WiFi-3D/docs/paper/NTN_IT_CT.pdf) and the frozen benchmark snapshot in [experiment_log.csv](D:/Resfes_2026/Person-in-WiFi-3D/work_dirs/paper_M1-5/logs/experiment_log.csv).

## 1. What is confirmed from the current benchmark log

The current benchmark snapshot for `M0–M4` is internally consistent in the following ways:

- All five variants appear in the same log file.
- All five variants report the same person-count breakdown:
  - `count_1p = 2586`
  - `count_2p = 3184`
  - `count_3p = 2054`
- All five rows are explicitly marked as `no bone loss`.
- The logged configurations match the intended ablation ladder:
  - `M0`: linear adapter + Transformer + DETR regression
  - `M1`: spectral adapter + Transformer + DETR regression
  - `M2`: spectral adapter + Mamba + DETR regression
  - `M3`: spectral adapter + Transformer + draft and rectified flow decoder
  - `M4`: spectral adapter + Mamba + draft and rectified flow decoder

This means the current slide deck can legitimately treat the table as one coherent **preliminary frozen local snapshot**.

## 2. What is supported, and what is only inferred

### Directly supported by the log

- `M3` has the best MPJPE in the frozen snapshot.
- `M4` has the best FPS, lowest parameter count, and lowest peak memory in the frozen snapshot.
- `M4` is a `no bone loss` variant in the benchmark snapshot used by the proposal.

### Reasonable inference, but should be phrased carefully

- The log path suggests the benchmark was produced in an `RTX3090` run directory.
- The benchmark likely comes from one consistent local evaluation environment.
- The current results are strong enough to motivate a deployment-oriented selection of `M4`.

These are acceptable to mention in rehearsal or backup explanation, but they should not be turned into stronger claims than the evidence supports.

## 3. Locked benchmark table for the presentation

Use the numbers below as the presentation ground truth unless the benchmark is rerun and the proposal is updated.

| Model | MPJPE (mm) | FPS | Params (M) | Peak Memory (MB) |
|---|---:|---:|---:|---:|
| `M0` | 169.34 | 45.69 | 13.13 | 155.60 |
| `M1` | 164.60 | 49.36 | 13.20 | 155.86 |
| `M2` | 169.41 | 51.48 | 11.97 | 151.16 |
| `M3` | 151.99 | 138.79 | 7.06 | 38.49 |
| `M4` | 159.00 | 159.85 | 5.83 | 27.02 |

## 4. Official M3 vs M4 interpretation

This wording should be treated as fixed for the presentation:

### Short version

**M3 is the accuracy winner. M4 is the practical deployment winner.**

### Full version

`M3` achieves the lowest MPJPE in the current frozen local snapshot, so it represents the strongest accuracy-focused flow variant. `M4` accepts a modest accuracy penalty relative to `M3`, but delivers the best deployment-oriented trade-off through the highest FPS, the fewest parameters, and the lowest peak memory.

### Selection statement for the final model

**M4 is selected as the final deployment-oriented model because it provides the strongest overall accuracy-efficiency trade-off in the current benchmark snapshot.**

### What not to say

Avoid:

- `M4 is the best model.`
- `M4 wins in every metric.`
- `M4 is more accurate than M3.`

## 5. Locked research questions for slide use

Keep the slide version short and stable:

- **RQ1:** Can Spectral Tokenization and WiMamba improve CSI representation and efficiency?
- **RQ2:** Can Rectified Flow improve pose accuracy and structural fidelity?
- **RQ3:** Which model provides the best accuracy-efficiency trade-off?

## 6. Locked contributions for slide use

Use only these three contributions on the main slide deck:

1. **Draft-to-Refine Rectified Flow** for WiFi-based multi-person 3D pose estimation.
2. **Efficient CSI encoding** using Spectral Tokenization and WiMamba.
3. **Controlled M0–M4 ablation** across accuracy, FPS, parameter count, and memory usage.

## 7. Safe claims to say strongly

These claims are well supported by the current proposal and benchmark:

- WiFi sensing is attractive for privacy-sensitive indoor environments because it does not capture visual identity.
- One-shot direct regression is difficult under noisy and ambiguous WiFi CSI.
- The ablation ladder is controlled and interpretable from `M0` to `M4`.
- `M3` improves MPJPE relative to the baseline.
- `M4` offers the strongest overall deployment trade-off in the current snapshot.

## 8. Claims that must be phrased carefully

These claims should be softened or explicitly marked as preliminary:

- `These are preliminary benchmark results.`
- `The current frozen local snapshot indicates...`
- `The present results suggest...`
- `We expect repeated evaluation to further strengthen reliability.`

Avoid overclaiming with:

- `final proof`
- `state of the art`
- `universally best`
- `fully validated`

## 9. Technical claims that should not be repeated incorrectly

The presentation should **not** say the following:

- `M4 removes matching-based supervision entirely.`
- `M4 removes query-based attention entirely.`
- `M4 uses bone loss to enforce skeleton consistency.`

Instead, the safe technical wording is:

`M4 combines spectral tokenization, a WiMamba encoder, and a draft-to-refine rectified-flow head. In the current implementation, it still retains query-based decoding and matching-based supervision during training, while improving efficiency through a lighter sequence model and a streamlined refinement stage.`

## 10. Figure-linked message locks

### Figure 1

Use this figure to say:

- `This is the end-to-end system view.`
- `The key transition is from draft pose to rectified-flow refinement.`

### Figure 3

Use this figure to say:

- `This is the technical detail of the final M4 model.`
- `The model combines a spectral input adapter, a WiMamba backbone, and a query-based refinement head.`

### Figure 4

Use this figure to say:

- `M3 gives the strongest accuracy result.`
- `M4 gives the strongest speed-size-memory trade-off.`

### Figure 5

Use this figure to say:

- `These are selected qualitative cases, not a universal visual proof.`
- `The figure illustrates how the flow-based variants behave more coherently in difficult multi-person scenes.`

## 11. Recommended speaker wording

### Speaker wording for Slide 8

`The quantitative results show two key findings. First, M3 achieves the lowest MPJPE at 151.99 mm, which makes it the strongest accuracy-focused variant in the current snapshot. Second, M4 reaches 159.85 FPS with only 5.83 million parameters and 27.02 MB peak memory, which makes it the strongest deployment-oriented variant under the accuracy-efficiency trade-off.`

### Speaker wording for Slide 9

`The qualitative results are intended as selected difficult cases rather than universal proof. Even so, they are consistent with the broader quantitative trend: the flow-based variants produce more stable structures than the direct-regression baseline in crowded scenes.`

## 12. Phase 2 final outputs

At the end of Phase 2, the team should treat these items as fixed:

- The benchmark table values
- The `M3 vs M4` interpretation
- The three research questions
- The three contributions
- The safe claim list
- The caution claim list
- The corrected technical wording for `M4`
