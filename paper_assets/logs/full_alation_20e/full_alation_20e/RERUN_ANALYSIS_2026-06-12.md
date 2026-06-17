# Full Ablation 20e Rerun Analysis

Source folder: `paper_assets/logs/full_alation_20e`

Analysis time: 2026-06-12

## Data Completeness

The folder contains complete rerun artifacts for all canonical IDs `M0` to `M9`:

- 10 eval files: `M0_eval.json` to `M9_eval.json`
- 10 benchmark files: `M0_benchmark.json` to `M9_benchmark.json`
- Aggregated CSV: `experiment_log.csv`
- Section latency CSV: `section_latency_log.csv`

CSV values match the corresponding JSON values for MPJPE and latency.

## Ranking By MPJPE

| Rank | ID | MPJPE | 1P | 2P | 3P | Latency ms | FPS | Params M | Config |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | M6 | 167.573 | 125.532 | 164.528 | 188.363 | 4.038 | 247.7 | 1.813 | `wi_tidir_wifi_draft_mamba2_csi.py` |
| 2 | M9 | 168.086 | 125.958 | 161.832 | 192.229 | 4.833 | 206.9 | 3.618 | `wi_tidir_wifi_mamba2_flattened.py` |
| 3 | M7 | 168.342 | 127.129 | 162.775 | 191.390 | 6.859 | 145.8 | 7.059 | `wi_tidir_wifi_transformer.py` |
| 4 | M1 | 169.271 | 132.597 | 163.323 | 190.810 | 12.039 | 83.1 | 13.199 | `petr_wifi.py` |
| 5 | M8 | 169.653 | 122.253 | 165.397 | 193.943 | 9.144 | 109.4 | 5.827 | `wi_tidir_wifi.py` |
| 6 | M4 | 169.796 | 126.479 | 167.593 | 190.251 | 5.529 | 180.8 | 5.255 | `wi_tidir_wifi_draft_transformer.py` |
| 7 | M5 | 171.055 | 127.258 | 171.979 | 188.481 | 8.564 | 116.8 | 4.023 | `wi_tidir_wifi_draft_mamba1.py` |
| 8 | M0 | 172.540 | 136.141 | 167.971 | 192.537 | 8.423 | 118.7 | 13.130 | `petr_wifi.py` |
| 9 | M2 | 174.961 | 143.475 | 166.642 | 196.772 | 14.411 | 69.4 | 11.967 | `petr_wifi_mamba.py` |
| 10 | M3 | 175.658 | 140.116 | 169.358 | 197.084 | 9.803 | 102.0 | 10.292 | `petr_wifi_mamba2_crossscan_pos_attn.py` |

## Main Findings

1. `M6` is the best overall model in this rerun. It is best by MPJPE, latency, FPS, parameter count, memory, 3-person MPJPE, and depth MPJPE.

2. `M9` is now correctly mapped to the flattened Mamba2-CSI config. It ranks second by MPJPE and is still much faster and smaller than the original PETR baseline `M0`.

3. Rectified flow does not improve the flattened Mamba2-CSI branch in this 20e run. `M9` is worse than `M6` by `+0.513 mm`, slower by `+0.795 ms`, and has about 2x parameters.

4. Rectified flow helps the Transformer and Mamba-1 WiTiDAR branches modestly:
   - `M7 - M4 = -1.454 mm`
   - `M8 - M5 = -1.403 mm`
   - `M9 - M6 = +0.513 mm`

5. The PETR-side Mamba variants are not beneficial in this rerun:
   - `M2` is worse than `M1` by `+5.689 mm`
   - `M3` is worse than `M1` by `+6.386 mm`

6. WiTiDAR-style heads reduce missed persons strongly compared with PETR:
   - `M0`: 417 missed persons, 2.76 percent miss rate
   - `M1`: 334 missed persons, 2.21 percent miss rate
   - `M6`: 54 missed persons, 0.36 percent miss rate
   - `M9`: 76 missed persons, 0.50 percent miss rate

7. The false-positive count is very large and almost constant because the evaluator counts all dense `max_per_img=100` predictions. This metric should be treated carefully unless post-NMS/person filtering is changed.

## Deltas

| Comparison | Delta MPJPE | Interpretation |
| --- | ---: | --- |
| `M1 - M0` | -3.269 | Spectral tokenizer improves PETR baseline by 1.89 percent. |
| `M2 - M1` | +5.689 | Mamba-1 PETR encoder hurts accuracy. |
| `M3 - M1` | +6.386 | Mamba2-CSI crossscan PETR encoder hurts accuracy. |
| `M4 - M1` | +0.524 | Draft WiTiDAR Transformer is similar in accuracy but much faster. |
| `M5 - M2` | -3.906 | Draft WiTiDAR improves over PETR Mamba-1. |
| `M6 - M3` | -8.085 | Draft WiTiDAR flattened Mamba2 strongly improves over PETR Mamba2 crossscan. |
| `M7 - M4` | -1.454 | Flow helps Transformer WiTiDAR modestly. |
| `M8 - M5` | -1.403 | Flow helps Mamba-1 WiTiDAR modestly. |
| `M9 - M6` | +0.513 | Flow does not help flattened Mamba2 in this run. |
| `M9 - M0` | -4.454 | Final flattened-flow Mamba2 improves over original baseline by 2.58 percent. |
| `M6 - M0` | -4.967 | Draft flattened Mamba2 improves over original baseline by 2.88 percent. |

## Efficiency Versus M0

| ID | MPJPE Improvement vs M0 | Speed vs M0 | Params vs M0 | Memory vs M0 |
| --- | ---: | ---: | ---: | ---: |
| M6 | +2.88% | 2.09x | 0.14x | 0.12x |
| M9 | +2.58% | 1.74x | 0.28x | 0.16x |
| M7 | +2.43% | 1.23x | 0.54x | 0.25x |
| M1 | +1.89% | 0.70x | 1.01x | 1.00x |
| M8 | +1.67% | 0.92x | 0.44x | 0.22x |
| M4 | +1.59% | 1.52x | 0.40x | 0.20x |

## Paper Implications

The strongest empirical story from this rerun is not "flow is always helpful." The stronger claim is:

> A compact WiTiDAR draft head with flattened Mamba2-CSI gives the best accuracy-efficiency tradeoff on the current 20e ablation.

If the paper must position `M9` as the proposed final model, it is defensible as a flow-enabled final candidate that improves over `M0`, but it is not the best model in the current table. For a stronger paper, either:

1. Promote `M6` as the main compact model and present `M9` as a flow refinement that improves 2-person and horizontal MPJPE but slightly hurts overall MPJPE, or
2. Run additional flow ablations for Mamba2-CSI to find settings where flow consistently improves over `M6`.

Recommended next ablations:

- `M9` with lower `loss_flow_weight`: 1.0, 2.0, 5.0
- `M9` with `flow_noise_strength`: 0.0, 0.05, 0.1
- `M9` with more `flow_num_steps`: 2, 4, 8
- Repeat `M6` and `M9` across at least 3 seeds to confirm the 0.513 mm gap.
