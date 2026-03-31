import torch
import torch.nn as nn
import torch.fft
from mmdet.models.builder import BACKBONES as MMDET_BACKBONES

from ..builder import BACKBONES as OPERA_BACKBONES


@MMDET_BACKBONES.register_module()
@OPERA_BACKBONES.register_module()
class WifiInputAdapter(nn.Module):
    """WiFi input adapter with switchable baseline and improved modes.

    `linear` reproduces the original Person-in-WiFi 3D projection:
    a single Linear layer mapping raw CSI features to the transformer space.

    `spectral` enables the improved tokenizer that mixes local temporal
    filtering and global frequency-domain cues before fusion.
    """

    def __init__(self,
                 in_channels,
                 embed_dims,
                 mode='spectral',
                 seq_len=180,
                 kernel_size=5):
        super().__init__()
        if mode not in ('linear', 'spectral'):
            raise ValueError(
                f"Unsupported WifiInputAdapter mode: {mode}. "
                "Expected 'linear' or 'spectral'.")

        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.mode = mode
        self.linear_proj = nn.Linear(in_channels, embed_dims)

        if self.mode == 'spectral':
            self.time_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=embed_dims,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=1)
            self.time_act = nn.GELU()

            self.freq_proj = nn.Linear(in_channels, embed_dims)
            freq_seq_len = seq_len
            freq_dim_len = embed_dims // 2 + 1
            self.complex_weight = nn.Parameter(
                torch.randn(
                    1, freq_seq_len, freq_dim_len, 2,
                    dtype=torch.float32) * 0.02)
            self.fusion = nn.Linear(embed_dims * 2, embed_dims)
            self.norm = nn.LayerNorm(embed_dims)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        if self.linear_proj.bias is not None:
            nn.init.constant_(self.linear_proj.bias, 0)

        if self.mode != 'spectral':
            return

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

    def forward(self, x):
        # x shape: (batch, length, in_channels)
        if self.mode == 'linear':
            return self.linear_proj(x)

        x_permute = x.permute(0, 2, 1)
        x_time = self.time_conv(x_permute)
        x_time = self.time_act(x_time)
        x_time = x_time.permute(0, 2, 1)

        x_freq_feat = self.freq_proj(x)
        original_dtype = x_freq_feat.dtype
        x_freq_feat_32 = x_freq_feat.to(torch.float32)

        x_fft_2d = torch.fft.rfft2(x_freq_feat_32, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        if weight.shape[1] != x_fft_2d.shape[1]:
            weight = weight[:, :x_fft_2d.shape[1], :]
        x_fft_2d = x_fft_2d * weight.to(x_fft_2d.device)

        x_freq_32 = torch.fft.irfft2(
            x_fft_2d,
            s=x_freq_feat.shape[1:],
            dim=(1, 2),
            norm='ortho')
        x_freq = x_freq_32.to(original_dtype)

        x_combined = torch.cat([x_time, x_freq], dim=-1)
        x_out = self.fusion(x_combined)
        return self.norm(x_out + x_time)


@MMDET_BACKBONES.register_module()
@OPERA_BACKBONES.register_module()
class SpectralTokenizer(WifiInputAdapter):
    """Backward-compatible alias for the improved spectral adapter."""

    def __init__(self, in_channels, embed_dims, seq_len=180, kernel_size=5):
        super().__init__(
            in_channels=in_channels,
            embed_dims=embed_dims,
            mode='spectral',
            seq_len=seq_len,
            kernel_size=kernel_size)
