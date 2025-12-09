# %%writefile /content/Person-in-WiFi-3D/opera/models/utils/spectral_tokenizer.py
# @title opera/models/utils/spectral_tokenizer.py
import torch
import torch.nn as nn
import torch.fft

class SpectralTokenizer(nn.Module):
    def __init__(self, in_channels, embed_dims):
        super().__init__()
        self.embed_dims = embed_dims

        self.time_proj = nn.Linear(in_channels, embed_dims)
        self.freq_proj = nn.Linear(in_channels, embed_dims)
        self.fusion = nn.Linear(embed_dims * 2, embed_dims)
        self.norm = nn.LayerNorm(embed_dims)
        self.act = nn.GELU()

        self._init_weights()

    def _init_weights(self):
        for m in [self.time_proj, self.freq_proj, self.fusion]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (Batch, Length, Channels)

        # --- Nhánh 1: Time Domain ---
        x_time = self.time_proj(x)

        # --- Nhánh 2: Frequency Domain ---
        x_freq_feat = self.freq_proj(x)

        # [FIX CRITICAL BUG]
        # Lưu lại kiểu dữ liệu gốc (float16 hoặc float32)
        original_dtype = x_freq_feat.dtype

        # Ép sang float32 để tính FFT.
        # Lý do 1: Tránh lỗi cuFFT FP16 yêu cầu power-of-2.
        # Lý do 2: FFT nhạy cảm với độ chính xác, float32 tốt hơn cho tín hiệu sóng.
        x_freq_feat_32 = x_freq_feat.to(torch.float32)

        # FFT (Real-to-Complex) trên float32
        x_fft = torch.fft.rfft(x_freq_feat_32, dim=1, norm='ortho')

        # iFFT (Complex-to-Real) quay lại miền thời gian trên float32
        x_freq_32 = torch.fft.irfft(x_fft, n=x.shape[1], dim=1, norm='ortho')

        # Ép ngược lại kiểu dữ liệu gốc (float16) để khớp với mạng
        x_freq = x_freq_32.to(original_dtype)

        # --- Nhánh 3: Fusion ---
        x_combined = torch.cat([x_time, x_freq], dim=-1)
        x_out = self.fusion(x_combined)

        # Residual connection + Norm
        return self.norm(x_out + x_time)