| ID | Role | Input | Encoder | Head | Flow inference | MPJPE | 1P | 2P | 3P | Latency ms | FPS | Params M | Memory MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `M0` | Budget-matched PETR baseline | Linear | Transformer | PETRHead | No | 172.540 | 136.141 | 167.971 | 192.537 | 8.423 | 118.7 | 13.130 | 155.60 |
| `M1` | Spectral PETR baseline | Spectral | Transformer | PETRHead | No | 169.271 | 132.597 | 163.323 | 190.810 | 12.039 | 83.1 | 13.199 | 155.86 |
| `M2` | PETR with Mamba-1 | Spectral | Mamba-1 | PETRHead | No | 174.961 | 143.475 | 166.642 | 196.772 | 14.411 | 69.4 | 11.967 | 151.16 |
| `M3` | PETR with Mamba2-CSI | Spectral | Mamba2-CSI flattened | PETRHead | No | 175.658 | 140.116 | 169.358 | 197.084 | 9.803 | 102.0 | 10.292 | 142.69 |
| `M4` | WiTiDAR draft with Transformer | Spectral | Transformer | WiTiDAR | No | 169.796 | 126.479 | 167.593 | 190.251 | 5.529 | 180.8 | 5.255 | 31.61 |
| `M5` | WiTiDAR draft with Mamba-1 | Spectral | Mamba-1 | WiTiDAR | No | 171.055 | 127.258 | 171.979 | 188.481 | 8.564 | 116.8 | 4.023 | 26.80 |
| `M6` | WiTiDAR draft with Mamba2-CSI | Spectral | Mamba2-CSI flattened | WiTiDAR | No | 167.573 | 125.532 | 164.528 | 188.363 | 4.038 | 247.7 | 1.813 | 18.41 |
| `M7` | WiTiDAR with Transformer | Spectral | Transformer | WiTiDAR | 1-step | 168.342 | 127.129 | 162.775 | 191.390 | 6.859 | 145.8 | 7.059 | 38.49 |
| `M8` | WiTiDAR with Mamba-1 | Spectral | Mamba-1 | WiTiDAR | 1-step | 169.653 | 122.253 | 165.397 | 193.943 | 9.144 | 109.4 | 5.827 | 33.68 |
| `M9` | Mamba2-CSI WiTiDAR | Spectral | Mamba2-CSI flattened | WiTiDAR | 1-step | 168.086 | 125.958 | 161.832 | 192.229 | 4.833 | 206.9 | 3.618 | 25.29 |
| `M9_RF2` | Proposed final model | Spectral | Mamba2-CSI flattened | WiTiDAR | 2-step | 165.487 | 121.785 | 159.341 | 190.179 | 4.764 | 209.9 | 3.618 | 25.29 |
