# %%writefile /content/Person-in-WiFi-3D/configs/wifi/wi_tidir_wifi.py
# @title configs/wifi/wi_tidir_wifi.py
# Kế thừa toàn bộ thiết lập dữ liệu và pipeline từ file gốc
_base_ = ['./petr_wifi.py']

model = dict(
    bbox_head=dict(
        type='opera.WiTiDARHead', # Dùng Head mới
        num_query=100,
        embed_dims=256,
        num_keypoints=14,
        loss_flow_weight=10.0,
        # Loss Config (Có thể tune lại weight)
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),

        # Dùng L1 Loss cho giai đoạn Draft này (dễ hội tụ hơn MSE)
        loss_kpt=dict(type='mmdet.L1Loss', loss_weight=5.0),

        # Train Config (Assigner)
        train_cfg=dict(
            assigner=dict(
                type='opera.PoseHungarianAssigner',
                cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                kpt_cost=dict(type='opera.KptL1Cost', weight=5.0), # Khớp với loss L1
                oks_cost=dict(type='opera.OksCost', weight=0.0) # Tạm tắt OKS cho đơn giản
            )
        ),
        test_cfg=dict(max_per_img=100),  # <--- ĐÃ THÊM DẤU PHẨY Ở ĐÂY

        transformer=dict(
            # --- [THAY ĐỔI CỐT LÕI] ---
            # Thay thế Transformer Encoder (O(N^2)) bằng WiMamba Encoder (O(N))
            encoder=dict(
                type='WiMambaEncoder',
                embed_dims=256,      # Giữ nguyên để khớp với Spectral Tokenizer
                num_layers=4,        # 4 lớp Mamba thường mạnh ngang 6 lớp Transformer cũ
                d_state=16,          # Kích thước trạng thái ẩn SSM
                d_conv=4,            # Local Convolution
                expand=2,            # Hệ số mở rộng kênh
                dropout=0.1
            )
        )
    )
)

# --- Mixed Precision Training ---

# --- Logging & Checkpoint ---
# Lưu checkpoint mỗi epoch để tiện theo dõi
checkpoint_config = dict(interval=1, max_keep_ckpts=20)
evaluation = dict(interval=1, metric='mpjpe') # Đánh giá mỗi epoch
runner = dict(type='EpochBasedRunner', max_epochs=5)