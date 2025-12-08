# @title opera/models/utils/mamba_encoder.py
# %%writefile /content/Person-in-WiFi-3D/opera/models/utils/mamba_encoder.py

import torch
import torch.nn as nn
from mmcv.runner.base_module import BaseModule
from .builder import TRANSFORMER_LAYER_SEQUENCE

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MambaEncoder(BaseModule):
    """
    Mamba Encoder thay thế cho Transformer Encoder.
    Sử dụng State Space Models để mô hình hóa chuỗi CSI dài với chi phí tuyến tính.

    Args:
        embed_dims (int): Kích thước embedding (d_model).
        num_layers (int): Số lớp Mamba xếp chồng lên nhau.
        d_state (int): Kích thước trạng thái ẩn (SSM state dimension). Mặc định 16.
        d_conv (int): Kích thước kernel của lớp Conv1d cục bộ. Mặc định 4.
        expand (int): Hệ số mở rộng khối (Block expansion factor). Mặc định 2.
    """

    def __init__(self,
                 embed_dims=256,
                 num_layers=6,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 dropout=0.1,
                 init_cfg=None):
        super().__init__(init_cfg)

        if Mamba is None:
            raise ImportError('Please install mamba-ssm to use MambaEncoder.')

        self.embed_dims = embed_dims
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            # Khối Mamba
            self.layers.append(
                Mamba(
                    d_model=embed_dims,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                )
            )
            # LayerNorm sau mỗi khối (Pre-norm hoặc Post-norm đều được, ở đây dùng Post-process)
            self.norms.append(nn.LayerNorm(embed_dims))

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key=None, value=None, query_pos=None, key_padding_mask=None, **kwargs):
        """
        Args:
            query (Tensor): Input feature có dạng [Seq_Len, Batch_Size, Embed_Dim].
                            (Trong PETR, đây là feat_flatten).
            key, value: Không sử dụng trong Mamba (vì nó tự mô hình hóa chuỗi).
            query_pos (Tensor): Positional encoding [Seq_Len, Batch_Size, Embed_Dim].
            key_padding_mask: [Batch_Size, Seq_Len].
        """

        # 1. Xử lý Positional Encoding
        # Mamba có khả năng nội tại để hiểu vị trí, nhưng với PETR,
        # cộng query_pos vào query giúp giữ thông tin không gian tốt hơn.
        if query_pos is not None:
            x = query + query_pos
        else:
            x = query

        # 2. Chuyển đổi chiều (Permute)
        # PETR/MMCV dùng: [Len, Batch, Dim]
        # Mamba yêu cầu:  [Batch, Len, Dim]
        x = x.permute(1, 0, 2)

        # 3. Đi qua các lớp Mamba
        for i in range(self.num_layers):
            residual = x

            # Mamba forward
            out = self.layers[i](x)

            # Dropout
            out = self.dropout(out)

            # Residual Connection + Norm
            x = self.norms[i](out + residual)

        # 4. Chuyển đổi chiều ngược lại cho Decoder
        # [Batch, Len, Dim] -> [Len, Batch, Dim]
        x = x.permute(1, 0, 2)

        return x