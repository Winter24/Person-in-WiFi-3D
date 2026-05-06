_base_ = ['./petr_wifi_mamba2_crossscan.py']

model = dict(
    bbox_head=dict(
        transformer=dict(
            encoder=dict(
                use_pos_embed=True))))

work_dir = './work_dirs/petr_wifi_mamba2_crossscan_pos'
