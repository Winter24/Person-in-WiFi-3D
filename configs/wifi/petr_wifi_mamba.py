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
                type='WiMambaEncoder',
                embed_dims=256,
                num_layers=6,
                d_state=16,
                d_conv=4,
                expand=2,
                dropout=0.1))))

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

work_dir = './work_dirs/petr_wifi_mamba'
