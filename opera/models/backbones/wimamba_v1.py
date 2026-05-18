# Copyright (c) Hikvision Research Institute. All rights reserved.
"""Compatibility wrapper for the original factorized WiMamba backbone."""

from .wimamba import FactorizedWiMambaBlock, WiMambaEncoder

__all__ = ['FactorizedWiMambaBlock', 'WiMambaEncoder']
