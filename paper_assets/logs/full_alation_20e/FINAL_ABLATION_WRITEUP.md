# Final Ablation Write-up for the 20e WiFi Pose Experiments

This document provides manuscript-ready text and tables for the 20-epoch ablation results in `paper_assets/logs/full_alation_20e`. The selected candidate is `M9_RF2`, which evaluates the original 20e `M9` checkpoint with two rectified-flow inference steps.

## Main Results Table

| ID | Role | Input | Encoder | Head | Flow inference | MPJPE | 1P | 2P | 3P | Latency ms | FPS | Params M | Memory MB |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `M0` | Original PETR baseline | Linear | Transformer | PETRHead | No | 172.540 | 136.141 | 167.971 | 192.537 | 8.423 | 118.7 | 13.130 | 155.60 |
| `M1` | Spectral PETR baseline | Spectral | Transformer | PETRHead | No | 169.271 | 132.597 | 163.323 | 190.810 | 12.039 | 83.1 | 13.199 | 155.86 |
| `M7` | WiTiDAR with Transformer | Spectral | Transformer | WiTiDAR | 1-step | 168.342 | 127.129 | 162.775 | 191.390 | 6.860 | 145.8 | 7.059 | 38.49 |
| `M8` | WiTiDAR with Mamba-1 | Spectral | Mamba-1 | WiTiDAR | 1-step | 169.653 | 122.253 | 165.397 | 193.943 | 9.144 | 109.4 | 5.827 | 33.68 |
| `M9` | Mamba2-CSI WiTiDAR | Spectral | Mamba2-CSI flattened | WiTiDAR | 1-step | 168.086 | 125.958 | 161.832 | 192.229 | 4.833 | 206.9 | 3.618 | 25.29 |
| `M9_RF2` | Proposed final model | Spectral | Mamba2-CSI flattened | WiTiDAR | 2-step | **165.487** | **121.785** | **159.341** | **190.179** | **4.764** | **209.9** | **3.618** | **25.29** |

## Supplementary Flow and Training Ablation

| ID | Checkpoint | Inference | MPJPE | Matched-only MPJPE | Missed persons | 1P | 2P | 3P | Latency ms | FPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `M6` | Draft Mamba2-CSI | No flow | 167.573 | 166.381 | **54** | 125.532 | 164.528 | 188.363 | 4.038 | 247.7 |
| `M9_no_flow` | M9 20e | Flow disabled | 167.732 | 165.919 | 82 | 122.870 | 163.026 | 191.421 | n/a | n/a |
| `M9` | M9 20e | 1-step RF | 168.086 | 166.409 | 76 | 125.958 | 161.832 | 192.229 | 4.833 | 206.9 |
| `M9_RF2` | M9 20e | 2-step RF | **165.487** | **163.886** | 72 | 121.785 | 159.341 | 190.179 | 4.764 | 209.9 |
| `T_FW2_20e_RF2` | loss_flow_weight = 2.0, 20e | 2-step RF | 167.981 | 166.082 | 86 | 121.335 | 160.970 | 194.804 | 4.740 | 211.0 |
| `T_FW2_20e_RF4` | loss_flow_weight = 2.0, 20e | 4-step RF | 169.454 | 167.275 | 99 | **120.505** | 161.851 | 197.854 | 5.636 | 177.4 |

## Manuscript Results Text

The filtered 20-epoch ablation isolates the main design path from the original PETR-style WiFi pose baseline to the final WiTiDAR model. The original linear-input PETR baseline (`M0`) achieved 172.540 mm MPJPE with 8.423 ms latency and 13.130M parameters. Replacing the linear input adapter with the spectral tokenizer while keeping the PETRHead (`M1`) reduced MPJPE to 169.271 mm, confirming that spectral tokenization improves the WiFi representation. However, this change increased latency to 12.039 ms, indicating that tokenization alone does not provide the desired efficiency improvement.

Replacing PETRHead with the WiTiDAR head substantially improved the accuracy-efficiency trade-off. In the Transformer branch, the WiTiDAR-flow model (`M7`) improved MPJPE to 168.342 mm while reducing latency to 6.860 ms and parameters to 7.059M. Within the complete WiTiDAR-flow family, the Mamba2-CSI flattened encoder provided the strongest candidate. The initial one-step rectified-flow version (`M9`) achieved 168.086 mm MPJPE with only 4.833 ms latency and 3.618M parameters, making it substantially faster and smaller than the original PETR baseline.

The selected improvement comes from correcting the rectified-flow inference procedure. Evaluating the same `M9` checkpoint with two rectified-flow integration steps (`M9_RF2`) reduced MPJPE from 168.086 mm to 165.487 mm. This corresponds to a 2.599 mm improvement over the original one-step `M9` inference and a 7.053 mm improvement over the original PETR baseline. Compared with `M0`, the selected `M9_RF2` model reduces MPJPE by 4.09%, decreases latency from 8.423 ms to 4.764 ms, and reduces the parameter count from 13.130M to 3.618M. The selected model also improves the 1-person, 2-person, and 3-person subsets relative to `M9`, indicating that the two-step solver improves the pose refinement process rather than shifting performance to a single occupancy regime.

## Manuscript Discussion Text

The ablation results suggest that the main limitation of the original `M9` configuration was not the Mamba2-CSI encoder itself, but the one-step rectified-flow inference used to refine predicted poses. The one-step solver provides a coarse Euler update and can overshoot the refinement trajectory. Increasing the number of inference steps from one to two consistently improved the final Mamba2-CSI WiTiDAR model, reducing both overall MPJPE and matched-only MPJPE. In contrast, increasing the solver to four steps in the `T_FW2_20e` branch worsened MPJPE and increased latency, suggesting that more integration steps are not automatically beneficial. The results support two-step rectified-flow inference as the best accuracy-efficiency operating point for the final model.

The supplementary `T_FW2_20e` experiments show that reducing the flow loss weight during training does not improve the selected 20-epoch model over the original `M9` checkpoint. Although `T_FW2_20e_RF2` performs better than its four-step counterpart, it remains worse than `M9_RF2` by 2.494 mm MPJPE and has a higher missed-person count. This indicates that the most reliable gain comes from inference-time solver correction rather than retraining with a smaller flow loss weight. The draft no-flow model `M6` remains an important supplementary reference because it has the lowest missed-person count, but it does not match the selected `M9_RF2` MPJPE. Therefore, the most defensible claim is that the selected spectral WiTiDAR model with a flattened Mamba2-CSI encoder and two-step rectified-flow inference provides the best overall accuracy-efficiency trade-off among the tested complete architectures.

## Recommended Claim

The proposed spectral WiTiDAR model with a flattened Mamba2-CSI encoder and two-step rectified-flow inference achieves the best overall MPJPE among the tested 20-epoch configurations while remaining compact and fast. Relative to the original PETR baseline, it reduces MPJPE from 172.540 mm to 165.487 mm, increases throughput from 118.7 FPS to 209.9 FPS, and reduces the parameter count from 13.130M to 3.618M.

## Suggested Captions

Table X. Main 20-epoch ablation following the model-development path from the original PETR baseline to the selected WiTiDAR model. `M0` is the original linear-input PETR baseline, `M1` adds spectral tokenization, `M7` replaces PETRHead with the WiTiDAR-flow head under a Transformer encoder, and `M8` and `M9` compare Mamba-family encoders under the same WiTiDAR-flow design. `M9_RF2` evaluates the `M9` checkpoint with two rectified-flow inference steps and achieves the best MPJPE while retaining low latency and parameter count.

Table Y. Supplementary flow and training ablation for the Mamba2-CSI WiTiDAR branch. The results separate draft no-flow inference, one-step rectified-flow inference, two-step rectified-flow inference, and the 20-epoch retrained `loss_flow_weight=2.0` branch. The two-step solver applied to the original `M9` checkpoint gives the best overall MPJPE, whereas the four-step solver increases latency and worsens accuracy.

## Data Sources

The numbers in this write-up are taken from `experiment_log.csv`, `section_latency_log.csv`, `M9_RF2_eval.json`, `T_FW2_20e_RF2_eval.json`, and `T_FW2_20e_RF4_eval.json` in `paper_assets/logs/full_alation_20e`.
