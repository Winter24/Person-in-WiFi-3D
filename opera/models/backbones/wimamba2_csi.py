# Copyright (c) Hikvision Research Institute. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import TRANSFORMER_LAYER_SEQUENCE as \
    MMCV_TRANSFORMER_LAYER_SEQUENCE
from mmcv.runner import BaseModule

from ..builder import BACKBONES
from ..utils.builder import TRANSFORMER_LAYER_SEQUENCE

try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None


def _require_mamba2():
    if Mamba2 is None:
        raise ImportError(
            '\n[Error] Mamba2 not found.\n'
            'Please install a recent mamba-ssm build to use WiMamba2 encoders.\n'
            'Suggested command: pip install -U causal-conv1d mamba-ssm\n'
            'Mamba3 requires a source install and is intentionally not used '
            'in this Mamba2 ablation ladder.'
        )
    return Mamba2


def _build_mamba2(dim, d_state, d_conv, expand, headdim):
    mamba2_cls = _require_mamba2()
    return mamba2_cls(
        d_model=dim,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        headdim=headdim)


def _run_mamba2_with_padding(mamba, x, pad_multiple=8):
    """Run Mamba2 with sequence padding for fused causal-conv stride rules."""
    seq_len = x.shape[1]
    pad_len = (-seq_len) % pad_multiple
    if pad_len:
        pad_shape = list(x.shape)
        pad_shape[1] = pad_len
        x = torch.cat([x, x.new_zeros(pad_shape)], dim=1)
    if not x.is_contiguous():
        x = x.contiguous()
    x = mamba(x)
    return x[:, :seq_len, :]


def _route_indices(route, num_spatial, seq_len):
    if route == 'antenna_major':
        order = [ant * seq_len + t
                 for ant in range(num_spatial)
                 for t in range(seq_len)]
    elif route == 'time_major':
        order = [ant * seq_len + t
                 for t in range(seq_len)
                 for ant in range(num_spatial)]
    elif route == 'serpentine':
        order = []
        for t in range(seq_len):
            ants = range(num_spatial)
            if t % 2 == 1:
                ants = reversed(range(num_spatial))
            order.extend([ant * seq_len + t for ant in ants])
    else:
        raise ValueError(f'Unsupported CSI Mamba2 route: {route}')

    order = torch.tensor(order, dtype=torch.long)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), dtype=torch.long)
    return order, inverse


class FactorizedWiMamba2Block(nn.Module):
    """Drop-in Mamba2 version of the current factorized WiMamba block."""

    def __init__(self,
                 dim,
                 d_state=64,
                 d_conv=4,
                 expand_t=2,
                 expand_s=1,
                 headdim=64,
                 dropout=0.1):
        super().__init__()

        self.norm_t = nn.LayerNorm(dim)
        self.mamba_t = _build_mamba2(
            dim, d_state=d_state, d_conv=d_conv,
            expand=expand_t, headdim=headdim)
        self.drop_t = nn.Dropout(dropout)

        self.norm_s = nn.LayerNorm(dim)
        self.mamba_s_fwd = _build_mamba2(
            dim, d_state=d_state, d_conv=d_conv,
            expand=expand_s, headdim=headdim)
        self.mamba_s_bwd = _build_mamba2(
            dim, d_state=d_state, d_conv=d_conv,
            expand=expand_s, headdim=headdim)
        self.drop_s = nn.Dropout(dropout)

    def forward(self, x):
        """Forward with x in shape (B, S, T, C)."""
        B, S, T, C = x.shape

        residual_t = x
        x_t = self.norm_t(x).reshape(B * S, T, C)
        x_t = _run_mamba2_with_padding(self.mamba_t, x_t)
        x_t = self.drop_t(x_t).reshape(B, S, T, C)
        x = residual_t + x_t

        residual_s = x
        x_s = self.norm_s(x)
        x_s = x_s.transpose(1, 2).contiguous().view(B * T, S, C)

        out_fwd = _run_mamba2_with_padding(self.mamba_s_fwd, x_s)
        x_s_rev = torch.flip(x_s, dims=[1]).contiguous()
        out_bwd = _run_mamba2_with_padding(self.mamba_s_bwd, x_s_rev)
        out_bwd = torch.flip(out_bwd, dims=[1]).contiguous()

        out_s = self.drop_s(out_fwd + out_bwd)
        out_s = out_s.view(B, T, S, C).transpose(1, 2).contiguous()
        return residual_s + out_s


class CSISeparablePositionEmbedding(nn.Module):
    """Learnable antenna/time positional prior for CSI tokens."""

    def __init__(self, embed_dims, num_spatial=9, seq_len=20, dropout=0.0):
        super().__init__()
        self.ant_embed = nn.Parameter(
            torch.zeros(1, num_spatial, 1, embed_dims))
        self.time_embed = nn.Parameter(
            torch.zeros(1, 1, seq_len, embed_dims))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.ant_embed, std=0.02)
        nn.init.normal_(self.time_embed, std=0.02)

    def forward(self, x):
        return self.dropout(x + self.ant_embed + self.time_embed)


class FlattenedCSIMamba2Block(nn.Module):
    """Mamba2 block over full CSI length with optional route fusion."""

    def __init__(self,
                 dim,
                 routes=('time_major',),
                 fusion='mean',
                 num_spatial=9,
                 seq_len=20,
                 d_state=64,
                 d_conv=4,
                 expand=2,
                 headdim=64,
                 dropout=0.1):
        super().__init__()
        if isinstance(routes, str):
            routes = (routes,)
        if fusion not in ('mean', 'learned'):
            raise ValueError(f'Unsupported route fusion: {fusion}')

        self.routes = tuple(routes)
        self.fusion = fusion
        self.num_routes = len(self.routes)
        self.norm = nn.LayerNorm(dim)
        self.mamba = _build_mamba2(
            dim, d_state=d_state, d_conv=d_conv,
            expand=expand, headdim=headdim)
        self.dropout = nn.Dropout(dropout)

        orders = []
        inverses = []
        for route in self.routes:
            order, inverse = _route_indices(route, num_spatial, seq_len)
            orders.append(order)
            inverses.append(inverse)
        self.register_buffer(
            'route_orders', torch.stack(orders, dim=0), persistent=False)
        self.register_buffer(
            'route_inverses', torch.stack(inverses, dim=0), persistent=False)

        if self.fusion == 'learned':
            self.route_logits = nn.Parameter(torch.zeros(self.num_routes))
        else:
            self.route_logits = None

    def _fuse_routes(self, restored):
        if self.fusion == 'learned':
            weights = torch.softmax(self.route_logits, dim=0)
            weights = weights.view(self.num_routes, 1, 1, 1)
            return (restored * weights).sum(dim=0)
        return restored.mean(dim=0)

    def forward(self, x):
        """Forward with x in shape (B, L, C)."""
        B, L, C = x.shape
        residual = x
        x = self.norm(x)

        routed = [
            x.index_select(1, self.route_orders[i])
            for i in range(self.num_routes)
        ]
        routed = torch.cat(routed, dim=0)
        routed = _run_mamba2_with_padding(self.mamba, routed)
        routed = routed.view(self.num_routes, B, L, C)

        restored = [
            routed[i].index_select(1, self.route_inverses[i])
            for i in range(self.num_routes)
        ]
        restored = torch.stack(restored, dim=0)
        fused = self._fuse_routes(restored)
        return residual + self.dropout(fused)


class FinalAttentionBlock(nn.Module):
    """One lightweight global reasoning block after Mamba2 mixing."""

    def __init__(self,
                 dim,
                 num_heads=8,
                 ffn_ratio=2.0,
                 dropout=0.1):
        super().__init__()
        hidden_dim = int(dim * ffn_ratio)
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.drop_attn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))

    def forward(self, x):
        residual = x
        x_norm = self.norm_attn(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm, need_weights=False)
        x = residual + self.drop_attn(attn_out)
        return x + self.ffn(self.norm_ffn(x))


@BACKBONES.register_module()
@MMCV_TRANSFORMER_LAYER_SEQUENCE.register_module()
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class WiMamba2DropInEncoder(BaseModule):
    """Mamba2 drop-in ablation for the existing factorized WiMamba layout."""

    def __init__(self,
                 embed_dims=256,
                 num_layers=4,
                 d_state=64,
                 d_conv=4,
                 expand=2,
                 headdim=64,
                 dropout=0.1,
                 num_spatial=9,
                 seq_len=20,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_spatial = num_spatial
        self.seq_len = seq_len

        self.layers = nn.ModuleList([
            FactorizedWiMamba2Block(
                dim=embed_dims,
                d_state=d_state,
                d_conv=d_conv,
                expand_t=expand,
                expand_s=1,
                headdim=headdim,
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
            raise ValueError(
                'WiMamba2DropInEncoder requires either "x" or "query".')

        if query_pos is not None:
            x = x + query_pos
        if x.dim() != 3:
            raise ValueError(
                f'WiMamba2DropInEncoder expects a 3D tensor, got {x.dim()}D.')

        expected_len = self.num_spatial * self.seq_len
        if x.shape[0] == expected_len:
            x = x.permute(1, 0, 2).contiguous()
        elif not x.is_contiguous():
            x = x.contiguous()

        B, L, C = x.shape
        if L != expected_len:
            raise RuntimeError(f'Expected sequence length {expected_len}, got {L}')

        x = x.reshape(B, self.num_spatial, self.seq_len, C)
        for layer in self.layers:
            x = layer(x)

        x = x.reshape(B, L, C)
        x = self.final_norm(x)
        return x.permute(1, 0, 2).contiguous()


@BACKBONES.register_module()
@MMCV_TRANSFORMER_LAYER_SEQUENCE.register_module()
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class WiMamba2CSIEncoder(BaseModule):
    """Structured flattened Mamba2 encoder for WiFi CSI tokens."""

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
                 routes=('time_major',),
                 fusion='mean',
                 use_pos_embed=False,
                 final_attn=False,
                 num_heads=8,
                 ffn_ratio=2.0,
                 init_cfg=None):
        super().__init__(init_cfg)
        if isinstance(routes, str):
            routes = (routes,)

        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_spatial = num_spatial
        self.seq_len = seq_len
        self.routes = tuple(routes)
        self.use_pos_embed = use_pos_embed
        self.final_attn_enabled = final_attn

        if use_pos_embed:
            self.pos_embed = CSISeparablePositionEmbedding(
                embed_dims=embed_dims,
                num_spatial=num_spatial,
                seq_len=seq_len,
                dropout=dropout)
        else:
            self.pos_embed = None

        self.layers = nn.ModuleList([
            FlattenedCSIMamba2Block(
                dim=embed_dims,
                routes=self.routes,
                fusion=fusion,
                num_spatial=num_spatial,
                seq_len=seq_len,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                dropout=dropout)
            for _ in range(num_layers)
        ])

        if final_attn:
            self.final_attn = FinalAttentionBlock(
                dim=embed_dims,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout)
        else:
            self.final_attn = None
        self.final_norm = nn.LayerNorm(embed_dims)

    def _to_batch_first(self, x):
        expected_len = self.num_spatial * self.seq_len
        if x.shape[0] == expected_len:
            return x.permute(1, 0, 2).contiguous()
        if not x.is_contiguous():
            return x.contiguous()
        return x

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
            raise ValueError('WiMamba2CSIEncoder requires either "x" or "query".')

        if query_pos is not None:
            x = x + query_pos
        if x.dim() != 3:
            raise ValueError(
                f'WiMamba2CSIEncoder expects a 3D tensor, got {x.dim()}D.')

        x = self._to_batch_first(x)
        B, L, C = x.shape
        expected_len = self.num_spatial * self.seq_len
        if L != expected_len:
            raise RuntimeError(f'Expected sequence length {expected_len}, got {L}')

        if self.pos_embed is not None:
            x_grid = x.reshape(B, self.num_spatial, self.seq_len, C)
            x_grid = self.pos_embed(x_grid)
            x = x_grid.reshape(B, L, C)

        for layer in self.layers:
            x = layer(x)

        if self.final_attn is not None:
            x = self.final_attn(x)

        x = self.final_norm(x)
        return x.permute(1, 0, 2).contiguous()
