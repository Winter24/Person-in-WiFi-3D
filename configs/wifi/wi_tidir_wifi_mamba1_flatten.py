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
            type='WiMamba1FlatEncoder',
            embed_dims=256,
            num_layers=4,
            d_state=16,
            d_conv=4,
            expand=2,
            dropout=0.1)))

work_dir = './work_dirs/wi_tidir_wifi_mamba1_flatten'
