# Copyright (c) Hikvision Research Institute. All rights reserved.
from mmcv.cnn.bricks.transformer import TRANSFORMER_LAYER_SEQUENCE as \
    MMCV_TRANSFORMER_LAYER_SEQUENCE

from ..builder import BACKBONES
from ..utils.builder import TRANSFORMER_LAYER_SEQUENCE
from .wimamba2_csi import WiMamba2CSIEncoder


@BACKBONES.register_module()
@MMCV_TRANSFORMER_LAYER_SEQUENCE.register_module()
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class WiMamba2FlatEncoder(WiMamba2CSIEncoder):
    """Single-route flattened Mamba2 encoder for fair backbone ablation.

    This wrapper keeps a dedicated registry name for the plain L=180 Mamba2
    baseline while reusing the CSI route implementation.
    """

    def __init__(self,
                 embed_dims=256,
                 num_layers=3,
                 d_state=64,
                 d_conv=4,
                 expand=2,
                 headdim=64,
                 dropout=0.1,
                 num_spatial=9,
                 seq_len=20,
                 init_cfg=None):
        super().__init__(
            embed_dims=embed_dims,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            dropout=dropout,
            num_spatial=num_spatial,
            seq_len=seq_len,
            routes=('time_major',),
            fusion='mean',
            use_pos_embed=False,
            final_attn=False,
            init_cfg=init_cfg)
