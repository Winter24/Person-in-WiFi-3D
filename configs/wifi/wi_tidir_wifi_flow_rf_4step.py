_base_ = ['./wi_tidir_wifi.py']

model = dict(
    bbox_head=dict(
        flow_refine_mode='rectified_flow',
        flow_num_steps=4,
        flow_noise_strength=0.0,
        loss_flow_weight=10.0))

work_dir = './work_dirs/wi_tidir_wifi_flow_rf_4step'
