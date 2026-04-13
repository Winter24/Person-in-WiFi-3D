import torch
import torch.nn as nn
import torch.fft
from mmdet.models.builder import BACKBONES as MMDET_BACKBONES

from ..builder import BACKBONES as OPERA_BACKBONES


@MMDET_BACKBONES.register_module()
@OPERA_BACKBONES.register_module()
class WifiInputAdapter(nn.Module):
    """Doppler-Guided Wi-Fi Input Adapter.
    
    Tận dụng biến đổi FFT để trích xuất Hồ sơ chuyển động (Doppler Profile).
    Dùng hồ sơ này để tạo ra Attention Mask (Gating) loại bỏ nhiễu tĩnh (Multipath),
    mà KHÔNG làm thay đổi cấu trúc pha (Phase) không gian của tín hiệu gốc.
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
            # Depthwise Conv1d: Làm mượt tín hiệu thời gian mà KHÔNG trộn lẫn các kênh
            self.time_conv = nn.Conv1d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=embed_dims) # Depthwise
            self.time_act = nn.GELU()

            # Mạng sinh Gate từ Doppler Profile (11 bins của RFFT từ 20 timesteps)
            fft_bins = self.seq_len // 2 + 1
            self.freq_gate = nn.Sequential(
                nn.Linear(fft_bins, fft_bins * 2),
                nn.GELU(),
                nn.Linear(fft_bins * 2, self.seq_len) # Trả về 20 trọng số cho 20 timestep
            )

            # Lớp trộn kênh cuối cùng trước khi cộng vào gốc
            self.channel_proj = nn.Linear(embed_dims, embed_dims)
            self.norm = nn.LayerNorm(embed_dims)

        self._init_weights()

    @property
    def linear_proj(self):
        return self.head

    def _init_weights(self):
        if self.mode != 'spectral':
            return

        nn.init.kaiming_normal_(self.time_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.time_conv.bias is not None:
            nn.init.constant_(self.time_conv.bias, 0)

        nn.init.xavier_uniform_(self.freq_gate[0].weight)
        nn.init.constant_(self.freq_gate[0].bias, 0)
        nn.init.xavier_uniform_(self.freq_gate[2].weight)
        nn.init.constant_(self.freq_gate[2].bias, 0)

        # 🚀 ZERO-INIT: Bắt buộc lớp chiếu cuối cùng bằng 0 để xuất phát từ Baseline
        nn.init.constant_(self.channel_proj.weight, 0.0)
        nn.init.constant_(self.channel_proj.bias, 0.0)

    def forward(self, x):
        # x shape: (B, 180, C)
        B, L, C = x.shape
        
        # Đặc trưng gốc bảo toàn 100% pha Không gian (Spatial Phase / AoA)
        x_linear = self.head(x)

        if self.mode == 'linear':
            return x_linear

        # Định hình lại: (B * 9, 20, Embed_Dims)
        x_grid = x_linear.view(B * self.num_spatial, self.seq_len, self.embed_dims)

        # 1. TEMPORAL SMOOTHING
        # (B*9, Embed_Dims, 20)
        x_time = x_grid.permute(0, 2, 1).contiguous()
        x_time = self.time_act(self.time_conv(x_time))

        # 2. DOPPLER MOTION PROFILE (FFT)
        # Thực hiện RFFT dọc theo chiều thời gian (dim=1) của x_grid
        # Kết quả: (B*9, 11, Embed_Dims)
        x_fft = torch.fft.rfft(x_grid, dim=1, norm='ortho')
        
        # Lấy biên độ (Magnitude) -> Chính là cường độ chuyển động
        x_fft_mag = torch.abs(x_fft)
        
        # Gộp dọc theo kênh để lấy Hồ sơ chuyển động tổng quát cho từng ăng-ten
        doppler_profile = x_fft_mag.mean(dim=-1) # -> (B*9, 11)

        # 3. FREQUENCY GATING (Doppler-Guided Attention)
        # Đưa Doppler Profile qua MLP để sinh ra trọng số cho 20 timestep
        time_gate = self.freq_gate(doppler_profile) # -> (B*9, 20)
        time_gate = torch.sigmoid(time_gate).unsqueeze(1) # -> (B*9, 1, 20)

        # 4. MODULATION & FUSION
        # Nhân đặc trưng thời gian với Gating Mask (Lọc nhiễu tĩnh)
        x_enhanced = x_time * time_gate
        
        # Đưa về lại (B*9, 20, Embed_Dims)
        x_enhanced = x_enhanced.permute(0, 2, 1).contiguous()
        
        # Trộn kênh (Channel Mixing)
        x_enhanced = self.channel_proj(x_enhanced)
        
        # Trả về shape gốc (B, 180, Embed_Dims)
        x_enhanced = x_enhanced.view(B, L, self.embed_dims)

        # Residual Connection
        return self.norm(x_linear + x_enhanced)


@MMDET_BACKBONES.register_module()
@OPERA_BACKBONES.register_module()
class SpectralTokenizer(WifiInputAdapter):
    def __init__(self, in_channels, embed_dims, seq_len=20, num_spatial=9, kernel_size=5):
        super().__init__(
            in_channels=in_channels, embed_dims=embed_dims, mode='spectral',
            num_spatial=num_spatial, seq_len=seq_len, kernel_size=kernel_size)