| ID | Role | Input | Encoder | Head | Flow inference | MPJPE | 1P | 2P | 3P | Latency ms | FPS | Params M | Memory MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `M0` | Original PETR baseline | Linear | Transformer | PETRHead | No | 172.554 | 136.141 | 168.012 | 192.529 | 8.423 | 118.7 | 13.130 | 155.60 |
| `M1` | Spectral PETR baseline | Spectral | Transformer | PETRHead | No | 169.271 | 125.502 | 165.038 | 193.273 | 12.039 | 83.1 | 13.199 | 155.86 |
| `M7` | WiTiDAR with Transformer | Spectral | Transformer | WiTiDAR | 1-step | 168.342 | 127.129 | 162.775 | 191.390 | 6.859 | 145.8 | 7.059 | 38.49 |
| `M8` | WiTiDAR with Mamba-1 | Spectral | Mamba-1 | WiTiDAR | 1-step | 169.653 | 122.253 | 165.397 | 193.943 | 9.144 | 109.4 | 5.827 | 33.68 |
| `M9` | Mamba2-CSI WiTiDAR | Spectral | Mamba2-CSI flattened | WiTiDAR | 1-step | 168.088 | 125.959 | 161.833 | 192.233 | 4.833 | 206.9 | 3.618 | 25.29 |
| `M9_RF2` | Proposed final model | Spectral | Mamba2-CSI flattened | WiTiDAR | 2-step | 165.487 | 121.785 | 159.341 | 190.179 | 4.764 | 209.9 | 3.618 | 25.29 |
