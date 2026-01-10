# %%writefile /content/Person-in-WiFi-3D/opera/models/dense_heads/__init__.py
# @title opera/models/dense_heads/__init__.py
# Copyright (c) Hikvision Research Institute. All rights reserved.
from .inspose_head import InsPoseHead
from .petr_head import PETRHead
from .soit_head import SOITHead
from .wi_tidar_head import WiTiDARHead

__all__ = ['InsPoseHead', 'PETRHead', 'SOITHead','WiTiDARHead']
