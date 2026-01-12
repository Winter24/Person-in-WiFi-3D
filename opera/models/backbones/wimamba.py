# Copyright (c) Hikvision Research Institute. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from ..builder import BACKBONES

# Cố gắng import Mamba, nếu chưa cài đặt sẽ báo lỗi rõ ràng trong __init__
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

class WiMambaBlock(nn.Module):
    """
    WiMamba Block: Khối cơ bản của kiến trúc.
    Luồng xử lý: Input -> Norm -> Mamba (SSM) -> Dropout -> Residual Add
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        # LayerNorm trước khi vào Mamba (Pre-norm stabilization)
        self.norm = nn.LayerNorm(dim)

        # Selective State Space Model
        self.mamba = Mamba(
            d_model=dim,      # Model dimension (D)
            d_state=d_state,  # SSM state expansion factor (N)
            d_conv=d_conv,    # Local convolution width
            expand=expand,    # Block expansion factor (E)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor có shape (Batch, Length, Dim)
        """
        # [QUAN TRỌNG] Mamba backend yêu cầu memory layout liên tục.
        # Sau các phép toán permute/view bên ngoài, tensor có thể bị rời rạc.
        residual = x
        x = self.norm(x)

        if not x.is_contiguous():
            x = x.contiguous()

        out = self.mamba(x)
        out = self.dropout(out)

        return residual + out

@BACKBONES.register_module()
class WiMambaEncoder(BaseModule):
    """
    WiMamba Encoder: Thay thế Transformer Encoder O(N^2) bằng State Space Model O(N).

    Đặc điểm:
    - Tốc độ suy luận tuyến tính theo độ dài chuỗi (Linear Complexity).
    - Bộ nhớ tiêu thụ thấp hơn Transformer.
    - Phù hợp với dữ liệu chuỗi thời gian liên tục như WiFi CSI.

    Args:
        embed_dims (int): Kích thước embedding.
        num_layers (int): Số lượng Mamba Block.
        d_state (int): Kích thước trạng thái ẩn SSM (thường là 16).
        d_conv (int): Kích thước kernel Conv1d cục bộ (thường là 4).
        expand (int): Hệ số mở rộng kênh (thường là 2).
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

        if Mamba is None:
            raise ImportError(
                '\n[Error] Mamba not found.\n'
                'Please install mamba-ssm to use WiMambaEncoder.\n'
                'Command: pip install causal-conv1d>=1.2.0 mamba-ssm'
            )

        self.embed_dims = embed_dims
        self.num_layers = num_layers

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

    def forward(self, x, query_pos=None, **kwargs):
        """
        Args:
            x (Tensor): Input feature.
                        Thông thường từ PETRTransformer sẽ là (Length, Batch, Embed_Dims).
            query_pos (Tensor, optional): Positional Encoding.
                                          Shape (Length, Batch, Embed_Dims).
        """

        # 1. Tích hợp Positional Encoding (nếu có)
        # Mặc dù Mamba là mô hình chuỗi (RNN-like), việc cộng PE giúp nó
        # nhận biết vị trí tuyệt đối tốt hơn trong không gian 3D.
        if query_pos is not None:
            x = x + query_pos

        # 2. Kiểm tra và chuyển đổi chiều dữ liệu
        # DETR/PETR dùng: (Length, Batch, Dim)
        # Mamba yêu cầu:  (Batch, Length, Dim)

        # Kiểm tra nếu đang ở dạng (L, B, D) -> Chuyển thành (B, L, D)
        if x.dim() == 3:
            # Giả sử chiều 1 là batch size nếu nó khớp với các config batch size
            # Tuy nhiên, an toàn nhất là permute (1, 0, 2) vì transformer input chuẩn là L, B, D
            x = x.permute(1, 0, 2)

        # [QUAN TRỌNG] Đảm bảo contiguous sau khi permute
        if not x.is_contiguous():
            x = x.contiguous()

        # 3. Forward qua các lớp Mamba
        for layer in self.layers:
            x = layer(x)

        # 4. Final Norm
        x = self.final_norm(x)

        # 5. Chuyển đổi ngược lại về định dạng Transformer cho Decoder
        # (Batch, Length, Dim) -> (Length, Batch, Dim)
        x = x.permute(1, 0, 2)

        if not x.is_contiguous():
            x = x.contiguous()

        return x