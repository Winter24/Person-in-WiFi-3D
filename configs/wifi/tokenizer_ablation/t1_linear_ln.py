_base_ = ['../petr_wifi.py']

model = dict(backbone=dict(mode='linear_ln'))
seed = 42
deterministic = True
runner = dict(type='EpochBasedRunner', max_epochs=20)
work_dir = './work_dirs/spectral_tokenizer_ablation/T1'
