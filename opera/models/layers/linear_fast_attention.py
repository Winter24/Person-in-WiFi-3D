import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner import BaseModule


@ATTENTION.register_module()
class LinearFastAttention(BaseModule):
    """Implementation of Linear Attention.

    This module is designed as a drop-in replacement for `MultiheadAttention`
    in Transformer-based architectures like DETR or PETR. It follows the API
    of `MultiScaleDeformableAttention` for maximum compatibility.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        dropout (float): A Dropout layer on `identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim) or (n, batch, embed_dim). Default to False.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        
        self.head_dim = embed_dims // num_heads
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # Projection layers for query, key, value, and output
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.q_proj, distribution='uniform', bias=0.)
        xavier_init(self.k_proj, distribution='uniform', bias=0.)
        xavier_init(self.v_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                # Gracefully accept and ignore Deformable Attention's specific args
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """
        Forward Function of LinearFastAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims) or (bs, num_query, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)` or `(bs, num_key, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)` or `(bs, num_key, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `key`, with
                shape [bs, num_key]. True values indicate padded positions.

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims] or [bs, num_query, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if self.batch_first:
            # (bs, n, c) -> (n, bs, c)
            query = query.permute(1, 0, 2)
            key = key.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        seq_len_q, bsz, _ = query.shape
        seq_len_kv, _, _ = key.shape

        # 1. Project Q, K, V and reshape for multi-head
        q = self.q_proj(query).view(seq_len_q, bsz, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(seq_len_kv, bsz, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(seq_len_kv, bsz, self.num_heads, self.head_dim)

        # 2. Apply kernel function to Q and K
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # 3. Apply padding mask to keys before computation
        if key_padding_mask is not None:
            # key_padding_mask is (bsz, seq_len_kv)
            # k is (seq_len_kv, bsz, num_heads, head_dim)
            # We need to broadcast mask to fill k
            # (bsz, seq_len_kv) -> (seq_len_kv, bsz, 1, 1)
            mask = key_padding_mask.T.unsqueeze(-1).unsqueeze(-1)
            k = k.masked_fill(mask, 0.0)

        # 4. Permute for efficient matrix multiplication
        q = q.permute(1, 2, 0, 3)  # (bsz, num_heads, seq_len_q, head_dim)
        k = k.permute(1, 2, 3, 0)  # (bsz, num_heads, head_dim, seq_len_kv)
        v = v.permute(1, 2, 0, 3)  # (bsz, num_heads, seq_len_kv, head_dim)

        # 5. Core of Linear Attention: Q @ (K.T @ V)
        # (bsz, num_heads, head_dim, seq_len_kv) @ (bsz, num_heads, seq_len_kv, head_dim)
        # -> (bsz, num_heads, head_dim, head_dim)
        kv_context = torch.matmul(k, v)

        # (bsz, num_heads, seq_len_q, head_dim) @ (bsz, num_heads, head_dim, head_dim)
        # -> (bsz, num_heads, seq_len_q, head_dim)
        output = torch.matmul(q, kv_context)

        # 6. Normalization
        # (bsz, num_heads, head_dim, seq_len_kv) -> sum over seq_len_kv
        # -> (bsz, num_heads, head_dim, 1)
        k_sum = k.sum(dim=-1, keepdim=True)
        
        # (bsz, num_heads, seq_len_q, head_dim) @ (bsz, num_heads, head_dim, 1)
        # -> (bsz, num_heads, seq_len_q, 1)
        normalizer = torch.matmul(q, k_sum).clamp(min=1e-6)
        
        output = output / normalizer

        # 7. Reshape and project output
        # (bsz, num_heads, seq_len_q, head_dim) -> (seq_len_q, bsz, num_heads, head_dim)
        # -> (seq_len_q, bsz, embed_dims)
        output = output.permute(2, 0, 1, 3).contiguous().view(seq_len_q, bsz, self.embed_dims)
        output = self.output_proj(output)

        if self.batch_first:
            # (n, bs, c) -> (bs, n, c)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity