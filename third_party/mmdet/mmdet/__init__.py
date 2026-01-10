# @title third_party/mmdet/mmdet/__init__.py
# %%writefile /content/Person-in-WiFi-3D/third_party/mmdet/mmdet/__init__.py

import mmcv
from .version import __version__, short_version

def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version

mmcv_minimum_version = '1.3.17'
mmcv_maximum_version = '1.6.0'
mmcv_version = digit_version(mmcv.__version__)


__all__ = ['__version__', 'short_version']
