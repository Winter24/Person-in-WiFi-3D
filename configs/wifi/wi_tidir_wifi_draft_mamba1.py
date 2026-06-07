_base_ = ['./wi_tidir_wifi.py']

model = dict(
    backbone=dict(
        type='WifiInputAdapter',
        in_channels=60,
        embed_dims=256,
        mode='spectral'),
    bbox_head=dict(
        type='opera.WiTiDARHead',
        transformer_encoder=None,
        mamba_cfg=dict(
            type='WiMambaEncoder',
            embed_dims=256,
            num_layers=4,
            d_state=16,
            d_conv=4,
            expand=2,
            dropout=0.1),
        flow_refine_mode='none',
        flow_num_steps=1,
        flow_noise_strength=0.0,
        loss_flow_weight=0.0))

work_dir = './work_dirs/wi_tidir_wifi_draft_mamba1'
