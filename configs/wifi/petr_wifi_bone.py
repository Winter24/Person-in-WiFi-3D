_base_ = ['./petr_wifi.py']

model = dict(
    bbox_head=dict(
        loss_bone=dict(
            _delete_=True,
            type='BoneLengthLoss',
            loss_weight=2.0,
        ),
    ),
)
