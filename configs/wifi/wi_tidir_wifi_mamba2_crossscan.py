_base_ = ['./wi_tidir_wifi.py']

model = dict(
    backbone=dict(
        type='WifiInputAdapter',
        in_channels=60,
        embed_dims=256,
        mode='spectral'),
    bbox_head=dict(
        type='opera.WiTiDARHead',
        mamba_cfg=dict(
            _delete_=True,
            type='WiMamba2CSIEncoder',
            embed_dims=256,
            num_layers=3,
            d_state=64,
            d_conv=4,
            expand=2,
            headdim=64,
            dropout=0.1,
            routes=('time_major', 'serpentine'),
            fusion='learned',
            use_pos_embed=False,
            final_attn=False)))

work_dir = './work_dirs/wi_tidir_wifi_mamba2_crossscan'
