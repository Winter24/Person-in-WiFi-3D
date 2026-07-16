| ID | Checkpoint | Inference | MPJPE | Matched-only MPJPE | Missed persons | 1P | 2P | 3P | Latency ms | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `M6` | Mamba-2 Draft | No refinement | 167.573 | 166.381 | 54 | 125.532 | 164.528 | 188.363 | 4.038 | 247.7 |
| `M9_no_flow` | Mamba-2 Flow (refinement bypassed) | Refinement bypassed | 167.732 | 165.919 | 82 | 122.870 | 163.026 | 191.421 | n/a | n/a |
| `M9` | Mamba-2 Flow (1 step) | 1 step | 168.086 | 166.409 | 76 | 125.958 | 161.832 | 192.229 | 4.833 | 206.9 |
| `M9_RF2` | Mamba-2 Flow (2 steps) | 2 steps | 165.487 | 163.886 | 72 | 121.785 | 159.341 | 190.179 | 4.764 | 209.9 |
