# opera/models/utils/spectral_tokenizer.py
import torch
import torch.nn as nn
import torch.fft
from mmdet.models.builder import BACKBONES as MMDET_BACKBONES

from ..builder import BACKBONES as OPERA_BACKBONES


@MMDET_BACKBONES.register_module()
@OPERA_BACKBONES.register_module()
class SpectralTokenizer(nn.Module):
    """
    SpectraPose Tokenizer v3: Local-Global Fusion.
    - Branch 1: Temporal Conv1D (Local Features & Noise Smoothing).
    - Branch 2: 2D FFT with Gating (Global Doppler & Spatial Features).
    """
    def __init__(self, in_channels, embed_dims, seq_len=180, kernel_size=5):
        super().__init__()
        self.embed_dims = embed_dims
        
        self.time_conv = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=embed_dims, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2,
            groups=1 
        )
        self.time_act = nn.GELU() 
        
        self.freq_proj = nn.Linear(in_channels, embed_dims)
        freq_seq_len = seq_len 
        freq_dim_len = embed_dims // 2 + 1
        
        self.complex_weight = nn.Parameter(
            torch.randn(1, freq_seq_len, freq_dim_len, 2, dtype=torch.float32) * 0.02
        )

        # --- 3. Fusion Layer ---
        self.fusion = nn.Linear(embed_dims * 2, embed_dims)
        self.norm = nn.LayerNorm(embed_dims)
        self.act = nn.GELU()

        self._init_weights()

    def _init_weights(self):
        # Init cho Conv1D (Kaiming He init tốt cho Conv)
        nn.init.kaiming_normal_(self.time_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.time_conv.bias is not None:
            nn.init.constant_(self.time_conv.bias, 0)
            
        nn.init.xavier_uniform_(self.freq_proj.weight)
        nn.init.xavier_uniform_(self.fusion.weight)

    def forward(self, x):
        # x shape: (Batch, Length, In_Channels)
        B, L, C = x.shape
        
        # --- Branch 1: Time Domain (Local Context) ---
        # Conv1D cần input (B, C, L) -> Cần permute
        x_permute = x.permute(0, 2, 1) # (B, C, L)
        x_time = self.time_conv(x_permute) # -> (B, Embed_Dims, L)
        x_time = self.time_act(x_time)
        x_time = x_time.permute(0, 2, 1) # Quay lại (B, L, Embed_Dims)
        
        # --- Branch 2: 2D Frequency Domain (Global Context) ---
        x_freq_feat = self.freq_proj(x)
        
        # Ép kiểu float32
        original_dtype = x_freq_feat.dtype 
        x_freq_feat_32 = x_freq_feat.to(torch.float32)
        
        # 2D FFT
        x_fft_2d = torch.fft.rfft2(x_freq_feat_32, dim=(1, 2), norm='ortho')
        
        # Spectral Gating
        weight = torch.view_as_complex(self.complex_weight)
        if weight.shape[1] != x_fft_2d.shape[1]:
             weight = weight[:, :x_fft_2d.shape[1], :] 
        x_fft_2d = x_fft_2d * weight.to(x_fft_2d.device)
        
        # Inverse 2D FFT
        x_freq_32 = torch.fft.irfft2(
            x_fft_2d, 
            s=x_freq_feat.shape[1:], 
            dim=(1, 2), 
            norm='ortho'
        )
        x_freq = x_freq_32.to(original_dtype)

        # --- Branch 3: Fusion ---
        x_combined = torch.cat([x_time, x_freq], dim=-1)
        x_out = self.fusion(x_combined)
        
        return self.norm(x_out + x_time)
