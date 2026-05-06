_base_ = ['./petr_wifi_mamba2_crossscan_pos.py']

model = dict(
    bbox_head=dict(
        transformer=dict(
            encoder=dict(
                final_attn=True,
                num_heads=8,
                ffn_ratio=2.0))))

work_dir = './work_dirs/petr_wifi_mamba2_crossscan_pos_attn'
