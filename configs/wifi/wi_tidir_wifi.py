# %%writefile /content/Person-in-WiFi-3D/configs/wifi/wi_tidir_wifi.py
# @title configs/wifi/wi_tidir_wifi.py
# Kế thừa toàn bộ thiết lập dữ liệu và pipeline từ file gốc
_base_ = ['./petr_wifi.py']

model = dict(
    bbox_head=dict(
        type='opera.WiTiDARHead',
        num_query=100,
        embed_dims=256,
        num_keypoints=14,

        # --- Cấu hình siêu tham số Mamba từ Config ---
        mamba_cfg=dict(
            num_layers=4,     # Có thể thử 2, 4, 6 để xem cái nào nhanh nhất
            d_state=16,       # Có thể thử 32 nếu cần khả năng nhớ mạnh hơn
            d_conv=4,
            expand=2,
            dropout=0.1
        ),

        # --- Các Loss ---
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),

        loss_kpt=dict(type='mmdet.L1Loss', loss_weight=5.0),
        loss_bone=dict(type='BoneLengthLoss', loss_weight=2.0),

        # --- Cấu hình Bone Loss (Có thể bật/tắt dễ dàng) ---
        loss_limb=None,

        loss_flow_weight=10.0,

        train_cfg=dict(
            assigner=dict(
                type='opera.PoseHungarianAssigner',
                cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                kpt_cost=dict(type='opera.KptL1Cost', weight=5.0),
                oks_cost=dict(type='opera.OksCost', weight=0.0)
            )
        ),
        test_cfg=dict(max_per_img=100)
    ),
)
# --- Mixed Precision Training ---

# --- Logging & Checkpoint ---
# Lưu checkpoint mỗi epoch để tiện theo dõi
checkpoint_config = dict(interval=1, max_keep_ckpts=20)
evaluation = dict(interval=1, metric='mpjpe') # Đánh giá mỗi epoch
runner = dict(type='EpochBasedRunner', max_epochs=5)
