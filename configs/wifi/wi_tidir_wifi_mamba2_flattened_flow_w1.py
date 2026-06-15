_base_ = ['./wi_tidir_wifi_mamba2_flattened.py']

model = dict(
    bbox_head=dict(
        flow_refine_mode='rectified_flow',
        flow_num_steps=1,
        flow_noise_strength=0.1,
        loss_flow_weight=1.0))

work_dir = './work_dirs/wi_tidir_wifi_mamba2_flattened_flow_w1'
