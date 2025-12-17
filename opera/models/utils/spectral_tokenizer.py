# @title opera/models/utils/spectral_tokenizer.py
import torch
import torch.nn as nn
import torch.fft

class SpectralTokenizer(nn.Module):
    """
    2D Physics-Aware Spectral Tokenizer with Learnable Gating.
    Thực hiện biến đổi Fourier 2D và lọc nhiễu thích nghi trong miền tần số.
    """
    def __init__(self, in_channels, embed_dims, seq_len=180):
        super().__init__()
        self.embed_dims = embed_dims

        # 1. Nhánh Thời gian (Time-domain)
        self.time_proj = nn.Linear(in_channels, embed_dims)

        # 2. Nhánh Tần số (Frequency-domain)
        self.freq_proj = nn.Linear(in_channels, embed_dims)

        # [NEW] Learnable Spectral Filter (Bộ lọc phổ có thể học)
        # Kích thước tần số sau rfft2 (Real-to-Complex)
        # Chiều Length giữ nguyên (L), Chiều Dim giảm còn (D//2 + 1)
        # Ta khởi tạo trọng số phức (Complex Parameter)
        freq_seq_len = seq_len
        freq_dim_len = embed_dims // 2 + 1

        # Tạo trọng số dạng số phức: (1, L, D//2+1)
        # Broadcasting cho Batch size
        self.complex_weight = nn.Parameter(
            torch.randn(1, freq_seq_len, freq_dim_len, 2, dtype=torch.float32) * 0.02
        )

        # 3. Fusion Layer
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
        # x shape: (Batch, Length, In_Channels)
        B, L, C = x.shape

        # --- Branch 1: Time Domain ---
        x_time = self.time_proj(x) # -> (B, L, D)

        # --- Branch 2: 2D Frequency Domain ---
        # B1: Project input
        x_freq_feat = self.freq_proj(x) # -> (B, L, D)

        # [QUAN TRỌNG]: Ép kiểu float32
        original_dtype = x_freq_feat.dtype
        x_freq_feat_32 = x_freq_feat.to(torch.float32)

        # B2: 2D FFT (Real-to-Complex)
        # Output shape: (B, L, D//2 + 1) - Số phức
        x_fft_2d = torch.fft.rfft2(x_freq_feat_32, dim=(1, 2), norm='ortho')

        # [NEW] B3: Spectral Gating (Lọc phổ)
        # Nhân element-wise với trọng số đã học
        # Chuyển weight về dạng số phức để nhân
        weight = torch.view_as_complex(self.complex_weight)

        # Nếu độ dài chuỗi thay đổi (inference realtime), ta cần nội suy weight
        # Nhưng ở đây ta giả sử L=180 cố định cho training ổn định
        if weight.shape[1] != x_fft_2d.shape[1]:
             # Fallback an toàn: chỉ dùng weight cho chiều Dim, broadcast chiều Time
             # (Nếu bạn muốn code flexible hoàn toàn)
             weight = weight[:, :x_fft_2d.shape[1], :]

        x_fft_2d = x_fft_2d * weight.to(x_fft_2d.device)

        # B4: Inverse 2D FFT
        x_freq_32 = torch.fft.irfft2(
            x_fft_2d,
            s=x_freq_feat.shape[1:],
            dim=(1, 2),
            norm='ortho'
        )

        # Ép ngược lại kiểu dữ liệu gốc
        x_freq = x_freq_32.to(original_dtype)

        # --- Branch 3: Fusion ---
        x_combined = torch.cat([x_time, x_freq], dim=-1)
        x_out = self.fusion(x_combined)

        return self.norm(x_out + x_time)