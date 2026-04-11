_base_ = ['./wi_tidir_wifi.py']

model = dict(
    backbone=dict(
        type='WifiInputAdapter',
        mode='linear'))

work_dir = './work_dirs/wi_tidir_wifi_linear'
