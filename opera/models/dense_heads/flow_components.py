# opera/models/dense_heads/flow_components.py
import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    """
    Mã hóa thời gian t (diffusion timestep) thành vector đặc trưng.
    Sử dụng hàm Sin/Cos tần số khác nhau (giống Positional Encoding của Transformer).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: tensor (B,) chứa giá trị thời gian [0, 1]
        device = t.device
        half_dim = self.dim // 2

        # Tạo các tần số từ 1 đến 10000 theo thang log
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # (B, 1) * (1, half_dim) -> (B, half_dim)
        emb = t[:, None] * emb[None, :]

        # Ghép sin và cos -> (B, dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResBlock(nn.Module):
    """
    Khối Residual MLP cơ bản: x -> Norm -> Linear -> SiLU -> Linear -> + -> x
    Giúp huấn luyện mạng sâu hơn mà không bị mất gradient.
    """
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), # SiLU (Swish) hoạt động rất tốt trong các mô hình Generative/Flow
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.net(x)

class VelocityMLP(nn.Module):
    """
    Mạng dự đoán vận tốc (Velocity Field Predictor).
    Nhiệm vụ: v = Model(x_t, t, condition)
    """
    def __init__(self,
                 input_dim=42,      # 14 keypoints * 3 coords
                 cond_dim=256,      # WiFi feature dim
                 time_dim=64,       # Time embedding dim
                 hidden_dim=512,    # Hidden layer size
                 num_layers=3,      # Số lượng ResBlock
                 dropout=0.1):
        super().__init__()

        # 1. Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # 2. Input Projection
        # Tổng hợp: Pose (42) + Condition (256) + Time (64) = 362
        self.input_size = input_dim + cond_dim + time_dim

        # Nâng số chiều lên hidden_dim để xử lý
        self.input_proj = nn.Linear(self.input_size, hidden_dim)

        # 3. Main Backbone (Stacked ResBlocks)
        self.blocks = nn.Sequential(*[
            ResBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)
        ])

        # 4. Output Projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        # Lớp cuối cùng dự đoán vận tốc
        # Init weight=0 để bắt đầu training với vận tốc ~0 (ổn định hơn)
        self.final_linear = nn.Linear(hidden_dim, input_dim)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def forward(self, x, t, condition):
        """
        Args:
            x: (B, N, input_dim) - Pose hiện tại (có thể là pose 14*3 flattened)
            t: (B,) - Thời gian
            condition: (B, N, cond_dim) - Đặc trưng WiFi (đã align với N queries)

        Returns:
            v: (B, N, input_dim) - Vector vận tốc
        """
        B, N, _ = x.shape

        # --- Xử lý Time ---
        t_emb = self.time_mlp(t) # (B, time_dim)
        # Broadcast t cho tất cả N queries: (B, 1, time_dim) -> (B, N, time_dim)
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)

        # --- Concatenate ---
        # Ghép tất cả thông tin lại theo chiều feature (dim=-1)
        # x: (B, N, 42)
        # condition: (B, N, 256)
        # t_emb: (B, N, 64)
        inp = torch.cat([x, condition, t_emb], dim=-1)

        # --- Forward Pass ---
        h = self.input_proj(inp) # -> (B, N, hidden_dim)
        h = self.blocks(h)       # -> (B, N, hidden_dim)
        h = self.output_norm(h)

        v = self.final_linear(h) # -> (B, N, input_dim)

        return v
