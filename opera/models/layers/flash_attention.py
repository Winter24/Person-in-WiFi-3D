# Copyright (c) OpenMMLab. All rights reserved.
# Custom module by User, adapted for Flash Attention.
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner import BaseModule

# Check for PyTorch 2.0+ availability for scaled_dot_product_attention
IS_PYTORCH_2 = hasattr(F, 'scaled_dot_product_attention')
if not IS_PYTORCH_2:
    warnings.warn(
        'PyTorch version is lower than 2.0. FlashAttention backend is not available. '
        'This module will not be registered.')

# Only register the module if the backend is available
if IS_PYTORCH_2:
    @ATTENTION.register_module()
    class FlashAttentionMMCV(BaseModule):
        """A wrapper for the built-in PyTorch 2.0 scaled_dot_product_attention.

        This module is designed as a drop-in replacement for `MultiheadAttention`
        and follows the API of `MultiScaleDeformableAttention` for maximum
        compatibility within the MMDetection/MMCV framework.

        Args:
            embed_dims (int): The embedding dimension of Attention.
            num_heads (int): Parallel attention heads.
            dropout (float): A Dropout layer on `identity`. Default: 0.0.
            batch_first (bool): Whether the input is (batch, n, c). Default: False.
            init_cfg (dict): The Config for initialization. Default: None.
        """

        def __init__(self,
                     embed_dims,
                     num_heads,
                     dropout=0.0,
                     batch_first=False,
                     init_cfg=None):
            super().__init__(init_cfg)
            if not IS_PYTORCH_2:
                raise ImportError(
                    'FlashAttentionMMCV requires PyTorch 2.0 or higher.')

            if embed_dims % num_heads != 0:
                raise ValueError(f'embed_dims must be divisible by num_heads, '
                                 f'but got {embed_dims} and {num_heads}')

            self.embed_dims = embed_dims
            self.num_heads = num_heads
            self.dropout = dropout
            self.batch_first = batch_first

            # Use a single projection layer for Q, K, V for efficiency
            self.qkv_proj = nn.Linear(embed_dims, embed_dims * 3)
            self.output_proj = nn.Linear(embed_dims, embed_dims)

            self.init_weights()

        def init_weights(self):
            """Default initialization for Parameters of Module."""
            xavier_init(self.qkv_proj, distribution='uniform', bias=0.)
            xavier_init(self.output_proj, distribution='uniform', bias=0.)

        def forward(self,
                    query,
                    key=None,
                    value=None,
                    identity=None,
                    query_pos=None,
                    key_padding_mask=None,
                    # Gracefully accept and ignore unused args
                    reference_points=None,
                    spatial_shapes=None,
                    level_start_index=None,
                    **kwargs):
            
            if key is None:
                key = query
            if value is None:
                value = query
            if identity is None:
                identity = query
            if query_pos is not None:
                query = query + query_pos
            
            # For self-attention, query, key, and value are the same.
            # We can process them together.
            if key is query and value is query:
                # Project Q, K, V together
                q, k, v = self.qkv_proj(query).chunk(3, dim=-1)
            else:
                # This path is for cross-attention, less common for FlashAttention usage
                # but supported for completeness.
                q = nn.Linear(self.embed_dims, self.embed_dims, device=query.device)(query)
                k = nn.Linear(self.embed_dims, self.embed_dims, device=key.device)(key)
                v = nn.Linear(self.embed_dims, self.embed_dims, device=value.device)(value)


            if self.batch_first:
                # (bs, n, c) -> (n, bs, c)
                query, key, value = [x.permute(1, 0, 2) for x in (q, k, v)]

            seq_len_q, bsz, _ = query.shape
            
            # Reshape for scaled_dot_product_attention: (bs, num_heads, seq_len, head_dim)
            query = query.view(seq_len_q, bsz, self.num_heads, -1).permute(1, 2, 0, 3)
            key = key.view(-1, bsz, self.num_heads, -1).permute(1, 2, 0, 3)
            value = value.view(-1, bsz, self.num_heads, -1).permute(1, 2, 0, 3)

            # The core call to the optimized backend
            # key_padding_mask should be a boolean mask where True indicates padding
            output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, # For self-attention, key_padding_mask is often not needed, but can be passed here if available
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False)

            # Reshape back to (seq_len, bs, embed_dims)
            output = output.permute(2, 0, 1, 3).contiguous().view(seq_len_q, bsz, self.embed_dims)
            output = self.output_proj(output)

            if self.batch_first:
                # (n, bs, c) -> (bs, n, c)
                output = output.permute(1, 0, 2)

            return output + identity