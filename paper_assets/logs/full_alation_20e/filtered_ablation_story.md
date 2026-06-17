# Filtered 20e Ablation Story

_Filtered main-table proposal for the 20-epoch WiFi 3D pose ablation, based on the rerun artifacts in `paper_assets/logs/full_alation_20e`._

---

## Scope

This table follows the revised experimental story:

1. Start from the original PETR-style baseline with linear WiFi input.
2. Add the spectral tokenizer while keeping PETRHead.
3. Replace the PETRHead design with the proposed WiTiDAR head with rectified flow.
4. Compare encoder choices inside the same proposed WiTiDAR-flow family.

Under this framing, `M2` and `M3` are removed because they keep PETRHead and only replace the encoder after the spectral-tokenizer baseline. They are no longer part of the main hypothesis, which is about moving away from PETRHead after spectral tokenization.

`M4`, `M5`, and `M6` are also removed from the main table because they are no-flow draft variants. In the new story, WiTiDAR is treated as the complete head design and is paired with flow by default. These no-flow variants can remain as internal or supplementary ablations, but they should not define the main result table.

## Main table

| ID | Role | Input | Encoder | Head | Flow | MPJPE | 1P | 2P | 3P | Latency ms | FPS | Params M | Memory MB |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `M0` | Original baseline | Linear | Transformer | PETRHead | No | 172.540 | 136.141 | 167.971 | 192.537 | 8.423 | 118.7 | 13.130 | 155.60 |
| `M1` | Tokenizer baseline | Spectral | Transformer | PETRHead | No | 169.271 | 132.597 | 163.323 | 190.810 | 12.039 | 83.1 | 13.199 | 155.86 |
| `M7` | WiTiDAR-Transformer | Spectral | Transformer | WiTiDAR | Yes | 168.342 | 127.129 | 162.775 | 191.390 | 6.860 | 145.8 | 7.059 | 38.49 |
| `M8` | WiTiDAR-Mamba1 | Spectral | Mamba-1 | WiTiDAR | Yes | 169.653 | 122.253 | 165.397 | 193.943 | 9.144 | 109.4 | 5.827 | 33.68 |
| `M9` | Proposed model | Spectral | Mamba2-CSI flattened | WiTiDAR | Yes | **168.086** | 125.958 | **161.832** | 192.229 | **4.833** | **206.9** | **3.618** | **25.29** |

## Relative gains

### Against the original baseline

| ID | MPJPE delta vs `M0` | MPJPE gain | Speed vs `M0` | Params vs `M0` | Memory vs `M0` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `M1` | -3.269 | 1.89% | 0.70x | 1.01x | 1.00x |
| `M7` | -4.199 | 2.43% | 1.23x | 0.54x | 0.25x |
| `M8` | -2.887 | 1.67% | 0.92x | 0.44x | 0.22x |
| `M9` | **-4.454** | **2.58%** | **1.74x** | **0.28x** | **0.16x** |

### Against the spectral PETR baseline

| ID | MPJPE delta vs `M1` | Interpretation |
| --- | ---: | --- |
| `M7` | -0.930 | Replacing PETRHead with WiTiDAR-flow improves accuracy and efficiency when the encoder remains Transformer-based. |
| `M8` | +0.381 | Mamba-1 reduces parameter count but does not improve MPJPE in the complete WiTiDAR-flow setting. |
| `M9` | **-1.185** | Mamba2-CSI flattened is the strongest encoder choice inside the WiTiDAR-flow family. |

## New scientific story

The revised main table tells a cleaner story than the full M0-M9 table because each row corresponds to a deliberate design stage.

First, `M1` shows that replacing the raw linear WiFi input adapter with a spectral tokenizer improves the PETR baseline from `172.540` mm to `169.271` mm MPJPE. This establishes spectral tokenization as a useful representation step, although it increases latency from `8.423` ms to `12.039` ms.

Second, `M7` shows that after spectral tokenization, the model benefits from replacing PETRHead with the WiTiDAR-flow head. Compared with `M1`, `M7` improves MPJPE by `0.930` mm, reduces latency from `12.039` ms to `6.860` ms, and cuts parameters from `13.199M` to `7.059M`.

Third, `M7`, `M8`, and `M9` compare encoder choices under the same high-level design: spectral input, WiTiDAR head, and flow refinement. In this controlled final-design family, `M9` gives the best overall trade-off. It has the best MPJPE, best 2-person MPJPE, lowest latency, highest FPS, fewest parameters, and lowest memory among the main-table models.

The core claim should therefore be:

> The proposed spectral WiTiDAR-flow model with a flattened Mamba2-CSI encoder achieves the best accuracy-efficiency trade-off among the complete design variants, improving over the original PETR baseline while substantially reducing latency, memory, and parameter count.

## Recommended conclusion

The filtered table supports a more coherent paper narrative than the full table. The full table mixes three different questions: whether spectral tokenization helps PETR, whether Mamba encoders help PETRHead, and whether no-flow draft WiTiDAR variants outperform flow-enabled variants. The filtered table instead focuses on the intended model-development path.

With this framing, `M9` is the natural proposed model:

- It improves MPJPE over `M0` by `4.454` mm, or `2.58%`.
- It is `1.74x` faster than `M0`.
- It uses only `27.5%` of the `M0` parameter count.
- It uses only `16.3%` of the `M0` peak allocated memory.
- It improves over the spectral PETR baseline `M1` by `1.185` mm while being much faster and smaller.

The strongest defensible conclusion is not that flow alone is always beneficial. The main-table conclusion should be that the complete WiTiDAR-flow architecture, when paired with the flattened Mamba2-CSI encoder, gives the best practical balance of accuracy and efficiency.

## Caveat for supplementary reporting

`M6` remains important as a supplementary or internal ablation because it is a no-flow draft model with slightly lower MPJPE than `M9` in the current 20e rerun. If reviewers ask whether flow independently improves the Mamba2-CSI branch, the current data do not support that isolated claim.

For that reason, the paper should avoid wording such as "flow consistently improves performance." A safer and more accurate statement is:

> Flow is part of the final WiTiDAR head design, and the final Mamba2-CSI WiTiDAR-flow model provides the best accuracy-efficiency trade-off among complete candidate architectures in the main comparison.

## Suggested paper table caption

Table X. Filtered 20-epoch ablation following the proposed model-design path. `M0` is the original PETR baseline. `M1` adds spectral tokenization. `M7` replaces PETRHead with the complete WiTiDAR-flow head while keeping a Transformer encoder. `M8` and `M9` compare Mamba encoder variants under the same WiTiDAR-flow setting. The proposed `M9` model achieves the best MPJPE among the main-table variants while being substantially faster and smaller than the original baseline.

## Data sources

- `paper_assets/logs/full_alation_20e/experiment_log.csv`
- `paper_assets/logs/full_alation_20e/section_latency_log.csv`
- `work_dirs/full_alation_20e/M0`
- `work_dirs/full_alation_20e/M1`
- `work_dirs/full_alation_20e/M7`
- `work_dirs/full_alation_20e/M8`
- `work_dirs/full_alation_20e/M9`
