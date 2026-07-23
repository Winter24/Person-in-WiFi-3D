import torch
import torch.nn as nn
from mmdet.models.builder import BACKBONES as MMDET_BACKBONES

from ..builder import BACKBONES as OPERA_BACKBONES


TOKENIZER_MODES = (
    'linear',
    'linear_ln',
    'temporal_residual',
    'spectral_gate_residual',
    'spectral',
)


@MMDET_BACKBONES.register_module()
@OPERA_BACKBONES.register_module()
class WifiInputAdapter(nn.Module):
    """Spectrally conditioned temporal residual tokenizer.

    ``spectral`` preserves the established checkpoint structure while the
    other modes isolate normalization, temporal convolution, and spectral
    gating for controlled component ablations.
    """

    def __init__(self,
                 in_channels,
                 embed_dims,
                 mode='spectral',
                 num_spatial=9,
                 seq_len=20,
                 kernel_size=5):
        super().__init__()
        if mode not in TOKENIZER_MODES:
            raise ValueError(
                f'Unsupported mode: {mode}. Expected one of {TOKENIZER_MODES}.')

        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.mode = mode
        self.num_spatial = num_spatial
        self.seq_len = seq_len
        self.head = nn.Linear(in_channels, embed_dims)

        self.uses_temporal_branch = mode in ('temporal_residual', 'spectral')
        self.uses_spectral_gate = mode in ('spectral_gate_residual', 'spectral')
        self.uses_residual_correction = mode in (
            'temporal_residual', 'spectral_gate_residual', 'spectral')

        if self.uses_temporal_branch:
            self.time_conv = nn.Conv1d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=embed_dims)
            self.time_act = nn.GELU()

        if self.uses_spectral_gate:
            fft_bins = self.seq_len // 2 + 1
            self.freq_gate = nn.Sequential(
                nn.Linear(fft_bins, fft_bins * 2),
                nn.GELU(),
                nn.Linear(fft_bins * 2, self.seq_len),
            )

        if self.uses_residual_correction:
            self.channel_proj = nn.Linear(embed_dims, embed_dims)

        if mode != 'linear':
            self.norm = nn.LayerNorm(embed_dims)

        self._init_weights()

    @property
    def linear_proj(self):
        return self.head

    def _init_weights(self):
        if self.uses_temporal_branch:
            nn.init.kaiming_normal_(
                self.time_conv.weight, mode='fan_out', nonlinearity='relu')
            if self.time_conv.bias is not None:
                nn.init.constant_(self.time_conv.bias, 0)

        if self.uses_spectral_gate:
            nn.init.xavier_uniform_(self.freq_gate[0].weight)
            nn.init.constant_(self.freq_gate[0].bias, 0)
            nn.init.xavier_uniform_(self.freq_gate[2].weight)
            nn.init.constant_(self.freq_gate[2].bias, 0)

        if self.uses_residual_correction:
            nn.init.constant_(self.channel_proj.weight, 0.0)
            nn.init.constant_(self.channel_proj.bias, 0.0)

    def _validate_grouped_length(self, length):
        expected = self.num_spatial * self.seq_len
        if length != expected:
            raise ValueError(
                f'Grouped tokenizer modes require L = num_spatial * seq_len '
                f'= {self.num_spatial} * {self.seq_len} = {expected}, got {length}.')

    def forward(self, x, return_debug=False):
        batch_size, length, _ = x.shape
        x_linear = self.head(x)

        if self.mode == 'linear':
            if return_debug:
                return x_linear, {
                    'x_linear': x_linear,
                    'x_token': x_linear,
                }
            return x_linear

        if self.mode == 'linear_ln':
            x_token = self.norm(x_linear)
            if return_debug:
                return x_token, {
                    'x_linear': x_linear,
                    'x_token': x_token,
                }
            return x_token

        self._validate_grouped_length(length)
        x_grid = x_linear.view(
            batch_size * self.num_spatial, self.seq_len, self.embed_dims)

        if self.uses_temporal_branch:
            temporal_channels = x_grid.permute(0, 2, 1).contiguous()
            temporal_channels = self.time_act(self.time_conv(temporal_channels))
            residual_branch = temporal_channels.permute(0, 2, 1).contiguous()
        else:
            temporal_channels = None
            residual_branch = x_grid

        x_fft_magnitude = None
        spectral_descriptor = None
        temporal_gate = None
        if self.uses_spectral_gate:
            x_fft = torch.fft.rfft(x_grid, dim=1, norm='ortho')
            x_fft_magnitude = torch.abs(x_fft)
            spectral_descriptor = x_fft_magnitude.mean(dim=-1)
            temporal_gate = torch.sigmoid(self.freq_gate(spectral_descriptor))
            residual_branch = residual_branch * temporal_gate.unsqueeze(-1)

        residual_correction_grid = self.channel_proj(residual_branch)
        residual_correction = residual_correction_grid.view(
            batch_size, length, self.embed_dims)
        x_token = self.norm(x_linear + residual_correction)

        if return_debug:
            debug = {
                'x_linear': x_linear,
                'x_grid': x_grid,
                'residual_branch': residual_branch,
                'residual_correction_grid': residual_correction_grid,
                'residual_correction': residual_correction,
                'x_token': x_token,
            }
            if temporal_channels is not None:
                debug['temporal_features'] = temporal_channels.permute(
                    0, 2, 1).contiguous()
            if spectral_descriptor is not None:
                debug.update({
                    'x_fft_magnitude': x_fft_magnitude,
                    'spectral_descriptor': spectral_descriptor,
                    'temporal_gate': temporal_gate,
                })
            return x_token, debug

        return x_token


@MMDET_BACKBONES.register_module()
@OPERA_BACKBONES.register_module()
class SpectralTokenizer(WifiInputAdapter):
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
