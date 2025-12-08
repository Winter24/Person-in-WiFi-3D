# @title configs/wifi/petr_wifi_mamba.py
# %%writefile /content/Person-in-WiFi-3D/configs/wifi/petr_wifi_mamba.py

# -------------------------------------------------------------------------
# Tên file: configs/petr/petr_wifi_mamba.py
# Mô tả: Cấu hình thử nghiệm thay thế Transformer Encoder bằng Mamba Encoder
# -------------------------------------------------------------------------

# 1. Kế thừa toàn bộ cấu hình từ file petr_wifi gốc
# Đảm bảo file petr_wifi.py nằm cùng thư mục với file này.
_base_ = ['./petr_wifi.py']

# 2. Ghi đè (Overwrite) phần kiến trúc mô hình
model = dict(
    bbox_head=dict(
        transformer=dict(
            # Thay thế phần Encoder
            encoder=dict(
                # _delete_=True là BẮT BUỘC để xóa sạch cấu hình Encoder cũ
                # (vì Encoder cũ có tham số 'transformerlayers' mà Mamba không dùng)
                _delete_=True,

                # Gọi Class MambaEncoder chúng ta đã đăng ký ở Bước 3
                type='MambaEncoder',

                # Kích thước embedding (phải khớp với in_channels của neck và embed_dims của decoder)
                embed_dims=256,

                # Số lớp Mamba xếp chồng lên nhau.
                # Transformer cũ dùng 6 lớp. Mamba hội tụ nhanh hơn nên có thể thử 4 hoặc 6.
                # Khuyến nghị: Giữ 6 để so sánh công bằng (fair comparison).
                num_layers=6,

                # --- Các tham số riêng của Mamba ---
                # Kích thước trạng thái ẩn (SSM state dimension).
                # Tăng lên (ví dụ 32, 64) sẽ nhớ tốt hơn nhưng tốn VRAM hơn.
                d_state=16,

                # Kích thước kernel tích chập cục bộ 1D.
                # Giúp mô hình nhìn thấy thông tin lân cận trước khi đưa vào SSM.
                d_conv=4,

                # Hệ số mở rộng block.
                # Input 256 -> Chiếu lên 256*2 = 512 bên trong Mamba -> Chiếu về 256.
                expand=2,

                # Dropout để tránh overfitting
                dropout=0.1
            ),

            # Phần Decoder giữ nguyên để đảm bảo cơ chế query object không bị đổi
            # Chúng ta không cần khai báo lại Decoder nếu không sửa gì,
            # vì nó đã được kế thừa từ _base_.
        )
    )
)

# 3. Tinh chỉnh Optimizer (Tùy chọn nhưng KHUYẾN NGHỊ)
# Mamba thường cho phép batch size lớn hơn và train nhanh hơn.
# Nếu bạn tăng batch size, hãy nhớ tăng learning rate theo quy tắc tuyến tính.
# Ví dụ: Nếu file gốc dùng AdamW lr=2e-5 cho Transformer
# Mamba có thể chịu được lr cao hơn một chút hoặc giữ nguyên.

optimizer = dict(
    type='AdamW',
    lr=2e-4, # Thử tăng LR lên gấp 10 lần so với Transformer (thường là 2e-5) vì Mamba ổn định hơn
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }
    )
)

# 4. Cấu hình log để dễ theo dõi sự khác biệt
log_config = dict(
    interval=10, # Log thường xuyên hơn (mỗi 10 step) để xem loss có giảm không
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook') # Bật cái này nếu muốn xem biểu đồ
    ]
)

# Đổi thư mục lưu checkpoint để không ghi đè lên model cũ
work_dir = '/home/winter24/Person-in-WiFi-3D-repo/data/wifipose/result_mamba'
