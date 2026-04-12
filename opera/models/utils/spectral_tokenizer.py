import torch
import torch.nn as nn
import torch.fft
from mmdet.models.builder import BACKBONES as MMDET_BACKBONES

from ..builder import BACKBONES as OPERA_BACKBONES


@MMDET_BACKBONES.register_module()
@OPERA_BACKBONES.register_module()
class WifiInputAdapter(nn.Module):
    """Wi-Fi input adapter with switchable baseline and spectral modes.

    [CRITICAL ASSUMPTION]
    This module expects x with shape (B, 180, C), where 180 comes from
    flattening (Rx=3, Tx=3, Time=20) in C-contiguous order. Every 20
    contiguous tokens must represent the temporal sequence of one
    (Rx, Tx) antenna link. If upstream layout changes, the spectral
    branch will silently become mathematically invalid.
    """

    def __init__(self,
                 in_channels,
                 embed_dims,
                 mode='spectral',
                 num_spatial=9,
                 seq_len=20,
                 kernel_size=5):
        super().__init__()
        if mode not in ('linear', 'spectral'):
            raise ValueError(f"Unsupported mode: {mode}")

        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.mode = mode
        self.num_spatial = num_spatial
        self.seq_len = seq_len
        self.head = nn.Linear(in_channels, embed_dims)

        if self.mode == 'spectral':
            self.spatial_norm = nn.LayerNorm(num_spatial)
            self.spatial_mixer = nn.Sequential(
                nn.Linear(num_spatial, num_spatial * 2),
                nn.GELU(),
                nn.Linear(num_spatial * 2, num_spatial))
            self.time_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=embed_dims,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=1)
            self.time_act = nn.GELU()

            self.freq_proj = nn.Linear(in_channels, embed_dims)
            self.complex_weight = nn.Parameter(
                torch.empty(
                    1, self.seq_len // 2 + 1, embed_dims, 2,
                    dtype=torch.float32))
            self.fusion = nn.Linear(embed_dims * 2, embed_dims)
            self.norm = nn.LayerNorm(embed_dims)

        self._init_weights()

    @property
    def linear_proj(self):
        """Backward-compatible alias for code/checkpoints using linear_proj."""
        return self.head

    def _init_weights(self):
        if self.mode != 'spectral':
            return

        nn.init.xavier_uniform_(self.spatial_mixer[0].weight)
        nn.init.constant_(self.spatial_mixer[0].bias, 0)
        nn.init.constant_(self.spatial_mixer[2].weight, 0)
        nn.init.constant_(self.spatial_mixer[2].bias, 0)

        nn.init.kaiming_normal_(
            self.time_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.time_conv.bias is not None:
            nn.init.constant_(self.time_conv.bias, 0)

        nn.init.xavier_uniform_(self.freq_proj.weight)
        if self.freq_proj.bias is not None:
            nn.init.constant_(self.freq_proj.bias, 0)

        nn.init.xavier_uniform_(self.fusion.weight)
        if self.fusion.bias is not None:
            nn.init.constant_(self.fusion.bias, 0)

        # Start from an identity frequency filter to avoid injecting
        # artificial spectral noise in the first optimization steps.
        nn.init.constant_(self.complex_weight[..., 0], 1.0)
        nn.init.constant_(self.complex_weight[..., 1], 0.0)

    def forward(self, x):
        # x shape: (B, 180, C)
        B, L, C = x.shape
        x_linear = self.head(x)

        if self.mode == 'linear':
            return x_linear

        if L != self.num_spatial * self.seq_len:
            raise RuntimeError(
                f"Expected sequence length {self.num_spatial * self.seq_len}, "
                f"got {L}")

        # Mix antenna-link information at each time step before temporal FFT.
        x_grid = x.reshape(B, self.num_spatial, self.seq_len, C)
        x_spatial = x_grid.permute(0, 2, 3, 1)
        residual = x_spatial
        x_spatial = self.spatial_norm(x_spatial)
        x_spatial = self.spatial_mixer(x_spatial)
        x_spatial = residual + x_spatial
        x_grid = x_spatial.permute(0, 3, 1, 2).contiguous()

        # Recover the temporal axis so temporal Conv/FFT operate on the
        # real time dimension rather than on the flattened token axis.
        x_temp = x_grid.reshape(B * self.num_spatial, self.seq_len, C)

        x_permute = x_temp.permute(0, 2, 1)
        x_time = self.time_act(self.time_conv(x_permute))
        x_time = x_time.permute(0, 2, 1)

        x_freq_feat = self.freq_proj(x_temp)
        original_dtype = x_freq_feat.dtype
        x_fft_1d = torch.fft.rfft(
            x_freq_feat.to(torch.float32), dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        if weight.shape[1] != x_fft_1d.shape[1]:
            weight = weight[:, :x_fft_1d.shape[1], :]
        x_fft_1d = x_fft_1d * weight.to(x_fft_1d.device)

        x_freq_32 = torch.fft.irfft(
            x_fft_1d, n=self.seq_len, dim=1, norm='ortho')
        x_freq = x_freq_32.to(original_dtype)

        x_combined = torch.cat([x_time, x_freq], dim=-1)
        x_out = self.fusion(x_combined)
        x_out = x_out.reshape(B, self.num_spatial, self.seq_len, self.embed_dims)
        x_out = x_out.reshape(B, L, self.embed_dims)
        return self.norm(x_out + x_linear)


@MMDET_BACKBONES.register_module()
@OPERA_BACKBONES.register_module()
class SpectralTokenizer(WifiInputAdapter):
    """Backward-compatible alias for the improved spectral adapter."""

    def __init__(self,
                 in_channels,
                 embed_dims,
                 seq_len=20,
                 num_spatial=9,
                 kernel_size=5):
        super().__init__(
            in_channels=in_channels,
            embed_dims=embed_dims,
            mode='spectral',
            num_spatial=num_spatial,
            seq_len=seq_len,
            kernel_size=kernel_size)
