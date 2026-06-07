_base_ = ['./petr_wifi.py']

model = dict(
    backbone=dict(
        type='WifiInputAdapter',
        in_channels=60,
        embed_dims=256,
        mode='spectral'),
    bbox_head=dict(
        type='opera.WiTiDARHead',
        num_query=100,
        embed_dims=256,
        num_keypoints=14,
        transformer_encoder=dict(
            type='mmcv.DetrTransformerEncoder',
            num_layers=6,
            transformerlayers=dict(
                type='mmcv.BaseTransformerLayer',
                attn_cfgs=dict(
                    type='mmcv.MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1),
                feedforward_channels=1024,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
        mamba_cfg=None,
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_kpt=dict(type='mmdet.L1Loss', loss_weight=5.0),
        loss_bone=None,
        flow_refine_mode='rectified_flow',
        flow_num_steps=1,
        flow_noise_strength=0.1,
        loss_flow_weight=10.0,
        train_cfg=dict(
            assigner=dict(
                type='opera.PoseHungarianAssigner',
                cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                kpt_cost=dict(type='opera.KptL1Cost', weight=5.0),
                oks_cost=dict(type='opera.OksCost', weight=0.0))),
        test_cfg=dict(max_per_img=100)),
)

checkpoint_config = dict(interval=5, max_keep_ckpts=20)
evaluation = dict(interval=5, metric='mpjpe')
runner = dict(type='EpochBasedRunner', max_epochs=20)
work_dir = './work_dirs/wi_tidir_wifi_transformer'
