| ID | Role | Input | Encoder | Head | Flow | MPJPE | 1P | 2P | 3P | FPS | Params M | Memory MB |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `M0` | Original baseline | Linear | Transformer | PETRHead | No | 172.554 | 136.141 | 168.012 | 192.529 | 118.7 | 13.130 | 155.60 |
| `M1` | Tokenizer ablation | Spectral | Transformer | PETRHead | No | 169.271 | 132.597 | 163.323 | 190.810 | 83.1 | 13.199 | 155.86 |
| `M2` | Encoder ablation | Spectral | Mamba-1 | PETRHead | No | 174.961 | 143.475 | 166.642 | 196.772 | 69.4 | 11.967 | 151.16 |
| `M3` | Encoder ablation | Spectral | Mamba2-CSI | PETRHead | No | 175.658 | 140.116 | 169.358 | 197.084 | 102.0 | 10.292 | 142.69 |
| `M4` | Draft-head ablation | Spectral | Transformer | WiTiDAR draft | No | 169.796 | 126.479 | 167.593 | 190.251 | 180.8 | 5.255 | 31.61 |
| `M5` | Draft-head ablation | Spectral | Mamba-1 | WiTiDAR draft | No | 171.055 | 127.258 | 171.979 | 188.481 | 116.8 | 4.023 | 26.80 |
| `M6` | Compact draft model | Spectral | Mamba2-CSI | WiTiDAR draft | No | 167.573 | 125.532 | 164.528 | 188.363 | 247.7 | 1.813 | 18.41 |
| `M7` | Flow model | Spectral | Transformer | WiTiDAR | 1-step | 168.342 | 127.129 | 162.775 | 191.390 | 145.8 | 7.059 | 38.49 |
| `M8` | Flow model | Spectral | Mamba-1 | WiTiDAR | 1-step | 169.653 | 122.253 | 165.397 | 193.943 | 109.4 | 5.827 | 33.68 |
| `M9` | Compact flow model | Spectral | Mamba2-CSI | WiTiDAR | 1-step | 168.086 | 125.958 | 161.832 | 192.229 | 206.9 | 3.618 | 25.29 |
| `M9_RF2` | Selected evaluation | Spectral | Mamba2-CSI | WiTiDAR | 2-step | 165.487 | 121.785 | 159.341 | 190.179 | 209.9 | 3.618 | 25.29 |
