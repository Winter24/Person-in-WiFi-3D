# Spec debug extraction + plotting cho slide M1

## 1. Mục tiêu

Mục tiêu của patch này là tạo một đường ống debug có thể:

1. Trích xuất các tensor trung gian bên trong `WifiInputAdapter` ở chế độ `spectral`.
2. Không làm thay đổi hành vi train/inference mặc định của model.
3. Xuất 4 panel ảnh thật để thay cho icon minh họa trong slide `Stage 1: residual motion-aware spectral tokenization`.
4. Tạo ra ảnh "real-but-clean": là output thật từ model, nhưng đủ sạch để đưa vào slide.

Patch này phục vụ trực tiếp cho việc visual hóa `M1`, không mở rộng sang `M2-M4`.

## 2. Nguyên tắc thiết kế

### 2.1. Không phá backward compatibility

- `WifiInputAdapter.forward(x)` khi được gọi như cũ phải trả về y nguyên tensor cũ.
- Mọi logic debug chỉ kích hoạt khi gọi rõ ràng `return_debug=True`.
- Các detector, config train/test, benchmark, và pipeline hiện có không cần sửa.

### 2.2. Debug phải bám sát implementation thật

Tensor debug phải phản ánh đúng các bước trong adapter:

1. `Linear projection`
2. `Temporal smoothing`
3. `RFFT magnitude`
4. `Doppler profile`
5. `Temporal gate`
6. `Residual spectral enhancement`
7. `Residual add + LayerNorm`

### 2.3. Plotting phải phục vụ slide, không chỉ phục vụ nghiên cứu nội bộ

Script xuất ảnh phải cho ra 4 panel có thể đưa vào slide gần như trực tiếp:

1. `Time-domain signal`
2. `Frequency spectrum / RFFT magnitude`
3. `Doppler magnitude profile`
4. `Motion-aware token map`

Không yêu cầu script xuất hình cho `Transformer Encoder` hoặc `Set Prediction`, vì hai khối này trên slide chỉ là downstream unchanged stages.

## 3. Giao diện debug cho adapter

### 3.1. File được sửa

- `D:\Resfes_2026\Person-in-WiFi-3D\opera\models\utils\spectral_tokenizer.py`

### 3.2. Forward contract

Adapter sẽ hỗ trợ giao diện:

```python
output = backbone(x)
output, debug = backbone(x, return_debug=True)
```

### 3.3. Hành vi mong muốn

#### Chế độ mặc định

```python
output = backbone(x)
```

- Nếu `mode='linear'`: trả về `x_linear`
- Nếu `mode='spectral'`: trả về `x_token`

Không thay đổi so với code cũ.

#### Chế độ debug

```python
output, debug = backbone(x, return_debug=True)
```

- `output` vẫn là tensor cuối đúng như luồng mặc định
- `debug` là `dict[str, Tensor]`

### 3.4. Debug dict cần có

Khi `mode='spectral'`, `debug` phải chứa ít nhất:

```python
debug = {
    "x_lin": ...,            # (B, L, E)
    "x_grid": ...,           # (B * S, T, E)
    "x_time": ...,           # (B * S, T, E)
    "x_fft_mag": ...,        # (B * S, F, E)
    "doppler_profile": ...,  # (B * S, F)
    "gate": ...,             # (B * S, T)
    "x_enhanced_grid": ...,  # (B * S, T, E)
    "x_enhanced": ...,       # (B, L, E)
    "x_token": ...,          # (B, L, E)
}
```

Trong đó:

- `B`: batch size
- `S`: `num_spatial`, mặc định là `9`
- `T`: `seq_len`, mặc định là `20`
- `F`: số bin `RFFT`, mặc định là `11`
- `E`: `embed_dims`, mặc định là `256`
- `L = S * T`, mặc định là `180`

Khi `mode='linear'`, `debug` tối thiểu gồm:

```python
debug = {
    "x_lin": ...,
    "x_token": ...,
}
```

## 4. Quy ước panel cần xuất

### Panel 1. Time-domain signal

- Nguồn tensor: `debug["x_time"]`
- Kiểu hình: line plot 1D
- Ý nghĩa: biểu diễn tín hiệu theo thời gian sau `temporal smoothing`
- Chọn dữ liệu:
  - 1 spatial group đại diện
  - 1 feature channel đại diện

### Panel 2. Frequency spectrum / RFFT magnitude

- Nguồn tensor: `debug["x_fft_mag"]`
- Kiểu hình: line plot 1D
- Ý nghĩa: phổ tần số của cùng spatial group và feature channel ở Panel 1
- Chọn dữ liệu:
  - cùng `spatial_index`
  - cùng `feature_index`

### Panel 3. Doppler magnitude profile

- Nguồn tensor: `debug["doppler_profile"]`
- Kiểu hình: heatmap 2D
- Ý nghĩa: mức năng lượng chuyển động theo `RFFT bins` trên tất cả `spatial groups`
- Kích thước mong muốn:
  - trục ngang: `frequency bins`
  - trục dọc: `spatial groups`

### Panel 4. Motion-aware token map

- Nguồn tensor: `debug["x_token"]` theo mặc định
- Có thể đổi sang `debug["x_enhanced_grid"]` nếu muốn minh họa riêng enhanced branch
- Kiểu hình: heatmap 2D
- Ý nghĩa: biểu diễn token cuối cùng sau residual spectral refinement
- Chọn dữ liệu:
  - 1 spatial group đại diện
  - xếp trục: `feature dim x timestep`

## 5. Chiến lược chọn sample và chỉ số đại diện

### 5.1. Sample

Script phải hỗ trợ:

- chọn `split`: `train`, `val`, hoặc `test`
- chọn `sample_index`
- dùng sample thật của dataset `WifiPoseDataset`

### 5.2. Spatial group

Mặc định script sẽ chọn `spatial_index='auto'`.

Quy tắc `auto`:

1. reshape `doppler_profile` thành `(B, S, F)`
2. với sample đang vẽ, tính tổng năng lượng theo từng spatial group
3. chọn spatial group có tổng năng lượng lớn nhất

Mục tiêu: tránh trường hợp chọn nhầm group ít chuyển động, dẫn đến hình "thật nhưng xấu".

### 5.3. Feature channel

Mặc định script sẽ chọn `feature_index='auto'`.

Quy tắc `auto`:

1. lấy `x_time` của spatial group đã chọn
2. tính độ lệch chuẩn theo thời gian cho từng feature dim
3. chọn feature dim có biên độ động học lớn nhất

Mục tiêu: Panel 1 và Panel 2 có đường cong rõ nét hơn.

## 6. Script plotting

### 6.1. File mới

- `D:\Resfes_2026\Person-in-WiFi-3D\tools\analysis\export_m1_debug_panels.py`

### 6.2. Đầu vào

Script nhận ít nhất:

```bash
python tools/analysis/export_m1_debug_panels.py <config> \
  [--checkpoint <ckpt>] \
  [--split test] \
  [--sample-index 0] \
  [--device cpu] \
  [--output-dir docs/presentation/Figures/m1_debug_panels]
```

### 6.3. Luồng xử lý

1. Nạp `Config` từ file config.
2. Build full model để có thể load checkpoint đúng keys.
3. Lấy `model.backbone` và chuyển sang `eval`.
4. Tạo dataset `WifiPoseDataset` với `pipeline=[]`.
5. Lấy một sample thật từ dataset.
6. Reshape `img` thành input backbone `(B, 180, 60)`.
7. Gọi:

```python
x_token, debug = backbone(x, return_debug=True)
```

8. Chọn `spatial_index` và `feature_index`.
9. Vẽ 4 panel.
10. Lưu:
   - 4 file PNG riêng
   - 1 file PNG tổng hợp 4 panel
   - 1 file JSON metadata

### 6.4. Tên file đầu ra

Script nên sinh:

- `time_domain_signal.png`
- `frequency_spectrum.png`
- `doppler_profile.png`
- `motion_aware_token_map.png`
- `m1_debug_panels_combined.png`
- `debug_metadata.json`

Nếu có `--prefix`, thì thêm prefix vào đầu tên file.

## 7. Metadata cần lưu

File JSON cần ghi lại ít nhất:

```json
{
  "config": "...",
  "checkpoint": "...",
  "split": "test",
  "dataset_root": "...",
  "sample_index": 0,
  "sample_name": "...",
  "device": "cpu",
  "spatial_index": 4,
  "feature_index": 137,
  "token_source": "x_token",
  "tensor_shapes": {
    "x_lin": [1, 180, 256],
    "x_time": [9, 20, 256],
    "x_fft_mag": [9, 11, 256],
    "doppler_profile": [9, 11],
    "gate": [9, 20],
    "x_token": [1, 180, 256]
  }
}
```

## 8. Quy tắc vẽ hình

### 8.1. Time-domain signal

- không làm trơn dữ liệu
- trục `x`: timestep
- trục `y`: activation

### 8.2. Frequency spectrum

- vẽ `|RFFT|` của cùng feature/channel đã chọn
- trục `x`: frequency bin
- trục `y`: magnitude

### 8.3. Doppler profile

- dùng heatmap 2D
- `origin='lower'`
- thêm colorbar

### 8.4. Token map

- dùng heatmap 2D
- mặc định vẽ `x_token`
- trục `x`: timestep
- trục `y`: feature dimension
- có thể dùng robust color scaling theo percentile để dễ nhìn hơn, nhưng không thay đổi tensor gốc

## 9. Phạm vi và giới hạn

Patch này không nhằm:

- sửa core training pipeline
- thay đổi logic `M1`
- thêm explainability cho `Transformer` hoặc `Flow`
- sinh video hoặc animation

Patch này chỉ nhằm:

- debug activation của `M1`
- tạo ảnh thật để thay icon minh họa trên slide

## 10. Tiêu chí hoàn thành

Patch được xem là hoàn thành khi:

1. `WifiInputAdapter` vẫn train/infer như cũ nếu không bật debug.
2. Script có thể chạy với config spectral adapter mà không cần sửa code tay.
3. Script xuất được 4 panel ảnh thật + 1 hình tổng hợp.
4. Panel 1-4 được map rõ ràng vào slide `Stage 1`.
