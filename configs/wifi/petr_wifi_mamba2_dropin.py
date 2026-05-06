_base_ = ['./petr_wifi.py']

model = dict(
    backbone=dict(
        type='WifiInputAdapter',
        in_channels=60,
        embed_dims=256,
        mode='spectral'),
    bbox_head=dict(
        transformer=dict(
            encoder=dict(
                _delete_=True,
                type='WiMamba2DropInEncoder',
                embed_dims=256,
                num_layers=4,
                d_state=64,
                d_conv=4,
                expand=2,
                headdim=64,
                dropout=0.1))))

work_dir = './work_dirs/petr_wifi_mamba2_dropin'
