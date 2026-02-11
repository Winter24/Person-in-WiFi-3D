import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import build_attention, build_feedforward_network
from mmcv.cnn.bricks import build_norm_layer

# =============================================================================
# 1. WiFiFNetLayer (Dùng cho Encoder - Thay thế Attention bằng Fourier)
# =============================================================================
@TRANSFORMER_LAYER.register_module()
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class WiFiFNetLayer(BaseModule):
    def __init__(self,
                 attn_cfgs=None, 
                 feedforward_channels=1024,
                 ffn_dropout=0.0,
                 operation_order=('self_attn', 'norm', 'ffn', 'norm'), 
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 **kwargs):
        
        super(WiFiFNetLayer, self).__init__(init_cfg)
        
        # Để tương thích với DetrTransformerEncoder
        self.operation_order = operation_order
        self.pre_norm = operation_order[0] == 'norm'

        # FNet embed_dims
        self.embed_dims = 256 
        if attn_cfgs is not None and isinstance(attn_cfgs, dict):
             self.embed_dims = attn_cfgs.get('embed_dims', 256)
        
        # Feed Forward Network (FFN)
        ffn_cfg = dict(
            type='FFN',
            embed_dims=self.embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=ffn_dropout,
            act_cfg=act_cfg)
        self.ffn = build_feedforward_network(ffn_cfg)
        
        # Norm Layers
        _, self.norm1 = build_norm_layer(norm_cfg, self.embed_dims)
        _, self.norm2 = build_norm_layer(norm_cfg, self.embed_dims)

    def forward(self, query, key=None, value=None, query_pos=None, key_pos=None, **kwargs):
        
        identity = query
        
        # --- Khối 1: Fourier Mixing (Global Context) ---
        # 1. Permute [Length, Batch, Dim] -> [Batch, Length, Dim]
        x = query.permute(1, 0, 2) 
        
        # 2. 2D FFT (trên chiều Thời gian và Feature)
        x = torch.fft.fftn(x, dim=(1, 2))
        
        # 3. Lấy phần thực
        x = x.real
        
        # 4. Permute lại -> [Length, Batch, Dim]
        x = x.permute(1, 0, 2)
        
        # Add & Norm
        if self.pre_norm:
            query = self.norm1(query + x)
        else:
            query = self.norm1(query + x) 
            
        # --- Khối 2: Feed Forward ---
        identity = query
        x = self.ffn(query)
        
        # Add & Norm
        query = self.norm2(query + x)
        
        return query

# =============================================================================
# 2. WiFiGraphLayer (Dùng cho Refine Decoder - Sửa lỗi bằng Skeleton GCN)
# =============================================================================
@TRANSFORMER_LAYER.register_module()
class WiFiGraphLayer(BaseModule):
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=('self_attn', 'norm', 'graph', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 num_keypoints=14, 
                 init_cfg=None,
                 **kwargs):
        
        super(WiFiGraphLayer, self).__init__(init_cfg)
        
        self.operation_order = operation_order
        self.pre_norm = operation_order[0] == 'norm'
        
        # 1. Attention
        if not isinstance(attn_cfgs, list):
            attn_cfgs = [attn_cfgs]
        
        self.attentions = nn.ModuleList()
        for cfg in attn_cfgs:
            self.attentions.append(build_attention(cfg))
            
        self.embed_dims = self.attentions[0].embed_dims
        
        # 2. FFN (Sử dụng ModuleList để hỗ trợ nhiều FFN trong pipeline)
        ffn_cfg = dict(
            type='FFN',
            embed_dims=self.embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=ffn_dropout,
            act_cfg=act_cfg)
        
        num_ffns = operation_order.count('ffn')
        self.ffns = nn.ModuleList()
        for _ in range(num_ffns):
            self.ffns.append(build_feedforward_network(ffn_cfg))

        # 3. Norms
        num_norms = operation_order.count('norm')
        self.norms = nn.ModuleList()
        for _ in range(num_norms):
            _, norm = build_norm_layer(norm_cfg, self.embed_dims)
            self.norms.append(norm)

        # 4. Graph Construction (14 Keypoints - CrowdPose/WifiPose Standard)
        # Mapping: 0:L-Sho, 1:R-Sho, 2:L-Elb, 3:R-Elb, 4:L-Wri, 5:R-Wri, 
        #          6:L-Hip, 7:R-Hip, 8:L-Kne, 9:R-Kne, 10:L-Ank, 11:R-Ank, 12:Head, 13:Neck
        self.skeleton = [
            # Thân trên
            (13, 12), (13, 0), (13, 1), (0, 1),
            # Tay
            (0, 2), (2, 4), (1, 3), (3, 5),
            # Thân dưới
            (0, 6), (1, 7), (13, 6), (13, 7), (6, 7),
            # Chân
            (6, 8), (8, 10), (7, 9), (9, 11)
        ]
        
        self.adj = torch.eye(num_keypoints, requires_grad=False)
        for i, j in self.skeleton:
            if i < num_keypoints and j < num_keypoints:
                self.adj[i, j] = 1.0
                self.adj[j, i] = 1.0
        
        
        # Normalized Laplacian
        d = self.adj.sum(1)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[d_inv_sqrt == float('inf')] = 0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj_norm = torch.mm(torch.mm(d_mat_inv_sqrt, self.adj), d_mat_inv_sqrt)
        self.register_buffer('adj_norm', adj_norm)
        
        
        # GCN Linear Weight (Zero Init để không phá vỡ pretrained features ban đầu)
        self.gcn_weight = nn.Linear(self.embed_dims, self.embed_dims)
        nn.init.zeros_(self.gcn_weight.weight)
        nn.init.zeros_(self.gcn_weight.bias)

    def forward(self, query, key=None, value=None, query_pos=None, key_pos=None, 
                attn_masks=None, query_key_padding_mask=None, key_padding_mask=None, **kwargs):
        
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        
        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query, temp_key, temp_value, identity if self.pre_norm else None,
                    query_pos=query_pos, key_pos=query_pos, attn_mask=attn_masks, 
                    key_padding_mask=query_key_padding_mask, **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query, key, value, identity if self.pre_norm else None,
                    query_pos=query_pos, key_pos=key_pos, attn_mask=attn_masks, 
                    key_padding_mask=key_padding_mask, **kwargs)
                attn_index += 1
                identity = query
                
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'ffn':
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1
                
            elif layer == 'graph':
                # Soft-GCN Mixing
                # 1. Linear Transform: XW
                gcn_feat = self.gcn_weight(query) # [len, bs, dim]
                
                # 2. Graph Convolution: A(XW) via einsum
                # 'lk,kbd->lbd': (Joints, Joints) x (Joints, Batch, Dim) -> (Joints, Batch, Dim)
                gcn_feat = torch.einsum('lk,kbd->lbd', self.adj_norm, gcn_feat)
                
                # 3. Activation Function (GELU mượt hơn ReLU)
                gcn_feat = torch.nn.functional.gelu(gcn_feat)

                if self.pre_norm:
                    residual = query
                else:
                    residual = identity
                
                # 4. Residual Connection
                query = gcn_feat + residual
                identity = query

        return query