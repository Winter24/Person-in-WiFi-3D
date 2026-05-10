_base_ = ['./wi_tidir_wifi.py']

model = dict(
    bbox_head=dict(
        flow_refine_mode='none',
        flow_num_steps=1,
        loss_flow_weight=0.0))

work_dir = './work_dirs/wi_tidir_wifi_flow_draft'
