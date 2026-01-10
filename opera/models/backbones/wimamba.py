# @title opera/models/backbones/wimamba.py
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from ..builder import BACKBONES

# Import thư viện Mamba
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

class WiMambaBlock(nn.Module):
    """
    Khối xử lý cơ bản của WiMamba.
    Cấu trúc: Norm -> Mamba (SSM) -> Residual Connection.
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        # Selective State Space Model
        self.mamba = Mamba(
            d_model=dim,      # Dimension of model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,    # Local convolution width
            expand=expand,    # Block expansion factor
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input x: (Batch, Length, Dim)
        # Output x: (Batch, Length, Dim)
        output = self.mamba(self.norm(x))
        return x + self.dropout(output)

@BACKBONES.register_module()
class WiMambaEncoder(BaseModule):
    """
    WiMamba Encoder: Thay thế Transformer Encoder O(N^2) bằng Mamba O(N).

    Args:
        embed_dims (int): Kích thước embedding (VD: 256).
        num_layers (int): Số lượng Mamba Block chồng lên nhau.
        d_state (int): Kích thước trạng thái ẩn của SSM (thường là 16 hoặc 32).
        d_conv (int): Kích thước kernel của local convolution trong Mamba.
        expand (int): Hệ số mở rộng kênh trong Mamba.
        dropout (float): Tỷ lệ dropout.
    """
    def __init__(self,
                 embed_dims=256,
                 num_layers=4,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 dropout=0.1,
                 init_cfg=None):
        super().__init__(init_cfg)

        # --- [FIX LỖI ATTRIBUTE ERROR] ---
        # Lưu attribute này để lớp cha (Transformer) có thể đọc được
        self.embed_dims = embed_dims
        # ---------------------------------

        if Mamba is None:
            raise ImportError(
                'Please install mamba-ssm to use WiMambaEncoder.\n'
                'Run: pip install causal-conv1d>=1.2.0 mamba-ssm'
            )

        self.layers = nn.ModuleList([
            WiMambaBlock(
                dim=embed_dims,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Norm cuối cùng để ổn định feature trước khi vào Decoder
        self.final_norm = nn.LayerNorm(embed_dims)

    def forward(self, x, **kwargs):
        """
        Lưu ý quan trọng về Shape:
        - PETRTransformer (gốc từ DETR) truyền vào x có dạng: (Length, Batch, Embed_Dims).
        - Mamba yêu cầu input dạng: (Batch, Length, Embed_Dims).

        Vì vậy, ta cần permute đầu vào và đầu ra.
        """
        # 1. Chuyển đổi Shape: (L, B, D) -> (B, L, D)
        x = x.permute(1, 0, 2)

        # 2. Đi qua các lớp Mamba
        for layer in self.layers:
            x = layer(x)

        # 3. Final Norm
        x = self.final_norm(x)

        # 4. Chuyển đổi ngược lại Shape cho Decoder: (B, L, D) -> (L, B, D)
        x = x.permute(1, 0, 2)

        return x