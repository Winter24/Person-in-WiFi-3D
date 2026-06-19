# Paper-ready Ablation and Supplementary Analysis

This document consolidates the 20-epoch ablation evidence for the Person-in-WiFi 3D extension experiments. It is written as manuscript-ready material and can be used as the basis for the main ablation section, supplementary ablation notes, and table captions. The selected model used for the primary claim is `M9_RF2`, which evaluates the trained `M9` checkpoint with two rectified-flow inference steps.

## Proposed Main Claim

The proposed spectral WiTiDAR model with a flattened Mamba2-CSI encoder and two-step rectified-flow inference achieves the best overall MPJPE among the tested 20-epoch configurations while remaining compact and fast. Compared with the original Person-in-WiFi 3D PETR baseline, the final model reduces MPJPE from 172.540 mm to 165.487 mm, improves throughput from 118.7 FPS to 209.9 FPS, and reduces the parameter count from 13.130M to 3.618M. These results support the proposed model as an accuracy-efficiency improvement over the original baseline rather than only a parameter-reduction variant.

## Main Ablation Table

| ID | Role | Input | Encoder | Head | Flow inference | MPJPE | 1P | 2P | 3P | Latency ms | FPS | Params M | Memory MB |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `M0` | Original PETR baseline | Linear | Transformer | PETRHead | No | 172.540 | 136.141 | 167.971 | 192.537 | 8.423 | 118.7 | 13.130 | 155.60 |
| `M1` | Spectral PETR baseline | Spectral | Transformer | PETRHead | No | 169.271 | 132.597 | 163.323 | 190.810 | 12.039 | 83.1 | 13.199 | 155.86 |
| `M7` | WiTiDAR with Transformer | Spectral | Transformer | WiTiDAR | 1-step | 168.342 | 127.129 | 162.775 | 191.390 | 6.860 | 145.8 | 7.059 | 38.49 |
| `M8` | WiTiDAR with Mamba-1 | Spectral | Mamba-1 | WiTiDAR | 1-step | 169.653 | 122.253 | 165.397 | 193.943 | 9.144 | 109.4 | 5.827 | 33.68 |
| `M9` | Mamba2-CSI WiTiDAR | Spectral | Mamba2-CSI flattened | WiTiDAR | 1-step | 168.086 | 125.958 | 161.832 | 192.229 | 4.833 | 206.9 | 3.618 | 25.29 |
| `M9_RF2` | Proposed final model | Spectral | Mamba2-CSI flattened | WiTiDAR | 2-step | **165.487** | **121.785** | **159.341** | **190.179** | **4.764** | **209.9** | **3.618** | **25.29** |

## Supplementary Flow Table

| ID | Checkpoint | Inference | MPJPE | Matched-only MPJPE | Missed persons | 1P | 2P | 3P | Latency ms | FPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `M6` | Draft Mamba2-CSI | No flow | 167.573 | 166.381 | **54** | 125.532 | 164.528 | 188.363 | 4.038 | 247.7 |
| `M9_no_flow` | M9 20e | Flow disabled | 167.732 | 165.919 | 82 | 122.870 | 163.026 | 191.421 | n/a | n/a |
| `M9` | M9 20e | 1-step RF | 168.086 | 166.409 | 76 | 125.958 | 161.832 | 192.229 | 4.833 | 206.9 |
| `M9_RF2` | M9 20e | 2-step RF | **165.487** | **163.886** | 72 | 121.785 | 159.341 | 190.179 | 4.764 | 209.9 |
| `T_FW2_20e_RF2` | Flow loss weight 2.0 | 2-step RF | 167.981 | 166.082 | 86 | 121.335 | 160.970 | 194.804 | 4.740 | 211.0 |
| `T_FW2_20e_RF4` | Flow loss weight 2.0 | 4-step RF | 169.454 | 167.275 | 99 | **120.505** | 161.851 | 197.854 | 5.636 | 177.4 |

## Methods Text

All ablation models were evaluated under the same Person-in-WiFi 3D evaluation protocol, using MPJPE as the primary metric and reporting results separately for one-person, two-person, and three-person scenes. The ablation design follows a controlled model-development path. `M0` is the original PETR-style baseline with a linear WiFi input adapter, Transformer encoder, and PETRHead. `M1` isolates the effect of the spectral tokenizer while keeping the PETRHead unchanged. `M7` replaces the PETRHead with the WiTiDAR head under a Transformer encoder, and `M8` and `M9` evaluate Mamba-family encoders under the same WiTiDAR-flow head. `M9_RF2` uses the same trained checkpoint as `M9` but changes the rectified-flow inference solver from one step to two steps, allowing the effect of the solver to be measured without retraining the model.

Runtime was measured with the project benchmark script using CUDA inference. The latency table reports end-to-end latency, throughput, parameter count, peak allocated memory, and section-level timing where available. This separation is important because the final model is not only selected by MPJPE; it must also preserve the efficiency advantage expected from the Mamba2-CSI encoder and the compact WiTiDAR design.

## Results Text

The original PETR baseline (`M0`) achieved 172.540 mm MPJPE, 8.423 ms latency, 118.7 FPS, and 13.130M parameters. Introducing the spectral tokenizer while retaining the PETRHead (`M1`) improved MPJPE to 169.271 mm, indicating that spectral tokenization provides a stronger representation for WiFi-based 3D pose estimation. However, the same change increased latency to 12.039 ms and reduced throughput to 83.1 FPS, showing that tokenization alone is not sufficient for a practical accuracy-efficiency improvement.

Replacing PETRHead with the WiTiDAR head improved the trade-off between accuracy and computational cost. With a Transformer encoder, the WiTiDAR model (`M7`) achieved 168.342 mm MPJPE while reducing latency to 6.860 ms and parameters to 7.059M. The Mamba2-CSI flattened encoder further improved efficiency. The original one-step rectified-flow configuration (`M9`) achieved 168.086 mm MPJPE with 4.833 ms latency, 206.9 FPS, and only 3.618M parameters. This already outperformed the original baseline by 4.454 mm MPJPE while using approximately 27.6% of its parameters.

The best result was obtained by evaluating the same `M9` checkpoint with two rectified-flow inference steps. This selected configuration (`M9_RF2`) reduced MPJPE to 165.487 mm, improving over one-step `M9` by 2.599 mm and over the original PETR baseline by 7.053 mm. The improvement was consistent across scene occupancy levels, with 1P, 2P, and 3P MPJPE values of 121.785 mm, 159.341 mm, and 190.179 mm, respectively. The solver change did not introduce a practical runtime penalty; `M9_RF2` measured 4.764 ms latency and 209.9 FPS, which is slightly faster than the measured one-step benchmark and substantially faster than the original baseline.

## Discussion Text

The ablation indicates that the main benefit comes from combining spectral WiFi tokenization, the WiTiDAR head, the flattened Mamba2-CSI encoder, and a corrected rectified-flow inference solver. The spectral tokenizer improves the input representation, but it is computationally expensive when paired with the original PETRHead. The WiTiDAR head and Mamba2-CSI encoder recover efficiency while preserving or improving accuracy. This sequence of results supports the design path from `M0` to `M9_RF2` and provides a cleaner claim than comparing only isolated modules.

The supplementary flow ablation clarifies why the selected model should be reported as `M9_RF2` rather than the originally logged one-step `M9`. The one-step rectified-flow solver behaves like a coarse Euler update and can under-integrate or overshoot the refinement trajectory. Moving from one to two inference steps improves both overall MPJPE and matched-only MPJPE, suggesting that the refined pose itself becomes more accurate rather than merely changing the detection penalty. In contrast, increasing to four steps in the retrained `T_FW2_20e` branch worsens MPJPE and reduces throughput, so the evidence supports two-step inference as the best operating point rather than a general "more steps is better" conclusion.

The `M6` draft model remains an important caveat because it has the lowest missed-person count and the fastest runtime among the reported Mamba2-CSI variants. However, its overall MPJPE is 2.086 mm worse than `M9_RF2`, and its role is better interpreted as a detection-oriented draft-head reference rather than the selected complete pipeline. Similarly, the `T_FW2_20e` experiments show that lowering the flow loss weight during full 20-epoch training does not improve the selected result. `T_FW2_20e_RF2` remains 2.494 mm worse than `M9_RF2`, and `T_FW2_20e_RF4` further degrades accuracy. Therefore, the most defensible interpretation is that inference-time correction of the rectified-flow solver is more beneficial than retraining the full pipeline with a smaller flow loss weight.

## Limitations Text

The current ablation is sufficient to support the internal model-selection claim, but it should not be overstated as a fully stabilized statistical conclusion. The reported results are based on single trained checkpoints for each configuration, so random-initialization variance has not yet been quantified. The current results also show a detection-refinement trade-off: `M6` has fewer missed persons, whereas `M9_RF2` achieves better overall MPJPE. This should be discussed transparently and, if time permits, followed by additional confidence calibration or matching diagnostics.

## Suggested Main-table Caption

Table X. Main 20-epoch ablation from the original PETR-style Person-in-WiFi 3D baseline to the selected WiTiDAR model. `M0` is the original linear-input PETR baseline. `M1` isolates spectral tokenization. `M7` introduces the WiTiDAR head with a Transformer encoder. `M8` and `M9` compare Mamba-family encoders under the same WiTiDAR-flow design. `M9_RF2` evaluates the trained `M9` checkpoint with two rectified-flow inference steps and achieves the best MPJPE while preserving low latency, high throughput, and a compact parameter count.

## Suggested Supplementary-table Caption

Table Y. Supplementary flow and training ablation for the Mamba2-CSI WiTiDAR branch. The table separates draft no-flow inference, disabled-flow inference on the trained checkpoint, one-step rectified-flow inference, two-step rectified-flow inference, and retraining with reduced flow loss weight. Two-step inference on the original `M9` checkpoint provides the best overall MPJPE, while four-step inference and reduced-flow-loss retraining do not improve the selected model.

## Data Provenance

The main numerical values in this document were taken from `experiment_log.csv` and `section_latency_log.csv` in `paper_assets/logs/full_alation_20e`. The selected model row uses `M9_RF2_eval.json` and `M9_RF2_benchmark.json`. The retrained-flow supplementary rows use `T_FW2_20e_RF2_eval.json`, `T_FW2_20e_RF2_benchmark.json`, `T_FW2_20e_RF4_eval.json`, and `T_FW2_20e_RF4_benchmark.json`. The `M9_no_flow` row is retained as an accuracy-only diagnostic because a corresponding full benchmark row was not recorded in the final log folder.
