# Copyright (c) Hikvision Research Institute. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import TRANSFORMER_LAYER_SEQUENCE as \
    MMCV_TRANSFORMER_LAYER_SEQUENCE
from mmcv.runner import BaseModule

from ..builder import BACKBONES
from ..utils.builder import TRANSFORMER_LAYER_SEQUENCE

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class FactorizedWiMambaBlock(nn.Module):
    """Factorized spatio-temporal Mamba block for WiFi CSI tokens."""

    def __init__(self,
                 dim,
                 d_state=16,
                 d_conv=4,
                 expand_t=2,
                 expand_s=1,
                 dropout=0.1):
        super().__init__()

        self.norm_t = nn.LayerNorm(dim)
        self.mamba_t = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand_t)
        self.drop_t = nn.Dropout(dropout)

        self.norm_s = nn.LayerNorm(dim)
        self.mamba_s_fwd = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand_s)
        self.mamba_s_bwd = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand_s)
        self.drop_s = nn.Dropout(dropout)

    def forward(self, x):
        """Forward with x in shape (B, S, T, C)."""
        B, S, T, C = x.shape

        residual_t = x
        x_t = self.norm_t(x)
        x_t = x_t.reshape(B * S, T, C)
        if not x_t.is_contiguous():
            x_t = x_t.contiguous()
        x_t = self.mamba_t(x_t)
        x_t = self.drop_t(x_t).reshape(B, S, T, C)
        x = residual_t + x_t

        residual_s = x
        x_s = self.norm_s(x)
        x_s = x_s.transpose(1, 2).contiguous().view(B * T, S, C)

        out_fwd = self.mamba_s_fwd(x_s)

        x_s_rev = torch.flip(x_s, dims=[1]).contiguous()
        out_bwd = self.mamba_s_bwd(x_s_rev)
        out_bwd = torch.flip(out_bwd, dims=[1]).contiguous()

        out_s = out_fwd + out_bwd
        out_s = self.drop_s(out_s)
        out_s = out_s.view(B, T, S, C).transpose(1, 2).contiguous()
        x = residual_s + out_s

        return x


@BACKBONES.register_module()
@MMCV_TRANSFORMER_LAYER_SEQUENCE.register_module()
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class WiMambaEncoder(BaseModule):
    """Factorized spatio-temporal WiMamba encoder."""

    def __init__(self,
                 embed_dims=256,
                 num_layers=4,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 dropout=0.1,
                 num_spatial=9,
                 seq_len=20,
                 init_cfg=None):
        super().__init__(init_cfg)

        if Mamba is None:
            raise ImportError(
                '\n[Error] Mamba not found.\n'
                'Please install mamba-ssm to use WiMambaEncoder.\n'
                'Command: pip install causal-conv1d>=1.2.0 mamba-ssm'
            )

        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_spatial = num_spatial
        self.seq_len = seq_len

        self.layers = nn.ModuleList([
            FactorizedWiMambaBlock(
                dim=embed_dims,
                d_state=d_state,
                d_conv=d_conv,
                expand_t=expand,
                expand_s=1,
                dropout=dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dims)

    def forward(self,
                x=None,
                query_pos=None,
                key=None,
                value=None,
                key_padding_mask=None,
                query=None,
                **kwargs):
        if x is None:
            x = query
        if x is None:
            raise ValueError('WiMambaEncoder requires either "x" or "query".')

        if query_pos is not None:
            x = x + query_pos

        if x.dim() != 3:
            raise ValueError(f'WiMambaEncoder expects a 3D tensor, got {x.dim()}D.')

        if x.shape[0] == (self.num_spatial * self.seq_len):
            x = x.permute(1, 0, 2).contiguous()
        elif not x.is_contiguous():
            x = x.contiguous()

        B, L, C = x.shape
        if L != self.num_spatial * self.seq_len:
            raise RuntimeError(
                f"Expected sequence length {self.num_spatial * self.seq_len}, got {L}")

        x = x.reshape(B, self.num_spatial, self.seq_len, C)

        for layer in self.layers:
            x = layer(x)

        x = x.reshape(B, L, C)
        x = self.final_norm(x)
        x = x.permute(1, 0, 2).contiguous()
        return x
