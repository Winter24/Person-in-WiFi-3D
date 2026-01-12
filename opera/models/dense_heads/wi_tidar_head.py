# @title opera/models/dense_heads/wi_tidar_head.py
# %%writefile /content/Person-in-WiFi-3D/opera/models/dense_heads/wi_tidar_head.py
# Copyright (c) Hikvision Research Institute. All rights reserved.
import torch
import torch.nn as nn
import json
import os
from mmcv.runner import BaseModule, force_fp32
from mmdet.core import multi_apply
from ..builder import HEADS, build_loss
from opera.core.bbox import build_assigner, build_sampler

# Import các module cho Flow Matching (Kiểm tra xem file đã tồn tại chưa)
try:
    from opera.models.utils.rectified_flow import RectifiedFlowWrapper
    from opera.models.dense_heads.flow_components import VelocityMLP
except ImportError:
    RectifiedFlowWrapper = None
    VelocityMLP = None
    print("Warning: Could not import Flow modules. Refinement branch will be disabled.")

@HEADS.register_module()
class WiTiDARHead(BaseModule):
    """
    WiTiDAR Head: Đầu dự đoán lai ghép cho bài toán Wifi-Pose 3D.

    Kiến trúc gồm 2 giai đoạn:
    1. Draft Stage: Dự đoán tư thế thô từ Object Queries.
    2. Refine Stage: Sử dụng Rectified Flow để tinh chỉnh tư thế dựa trên đặc trưng ngữ cảnh.
    """

    def __init__(self,
                 num_query=100,
                 embed_dims=256,
                 num_keypoints=14,
                 num_classes=1,
                 in_channels=2048, # Giữ tham số này để tương thích config, dù không dùng trực tiếp
                 sync_cls_avg_factor=True,
                 loss_cls=dict(
                     type='mmdet.FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
                 loss_kpt=dict(type='mmdet.L1Loss', loss_weight=5.0),
                 # [UPGRADE D] Loss ràng buộc độ dài xương
                 loss_bone=dict(type='BoneLengthLoss', loss_weight=10.0),
                 # Trọng số cho Flow Matching Loss
                 loss_flow_weight=10.0,
                 train_cfg=None,
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        super(WiTiDARHead, self).__init__(init_cfg)
        self.num_query = num_query
        self.embed_dims = embed_dims
        self.num_keypoints = num_keypoints
        self.num_classes = num_classes
        self.loss_flow_weight = loss_flow_weight
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # --- 1. Xây dựng Assigner & Sampler ---
        if train_cfg:
            self.assigner = build_assigner(train_cfg['assigner'])
            # Trong DETR-like models, sampler thường là PseudoSampler
            sampler_cfg = dict(type='mmdet.PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        # --- 2. Xây dựng các hàm Loss ---
        self.loss_cls = build_loss(loss_cls)
        self.loss_kpt = build_loss(loss_kpt)
        self.loss_bone = build_loss(loss_bone)

        # --- 3. Tải thống kê xương (Quan trọng cho Bone Loss) ---
        # Sử dụng register_buffer để dữ liệu tự động chuyển sang GPU/CPU theo model
        self._load_bone_statistics()

        # --- 4. Kiến trúc Mạng ---

        # A. Decoder Attention (Tiny Decoder)
        # Chuyển đổi Memory (từ Mamba) -> Object Queries
        # batch_first=True vì Mamba encoder output dạng (B, L, D)
        self.decoder_attn = nn.MultiheadAttention(embed_dims, num_heads=8, batch_first=True)
        self.decoder_norm = nn.LayerNorm(embed_dims)

        # Learnable Queries (tương tự DETR)
        self.query_embedding = nn.Embedding(num_query, embed_dims)

        # B. Draft Branch (Regressor & Classifier)
        self.cls_head = nn.Linear(embed_dims, num_classes) # WifiPose thường chỉ có 1 class (person)
        self.draft_regressor = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 3 * num_keypoints) # Output: (x, y, z) * 14
        )

        # C. Refine Branch (Rectified Flow)
        if VelocityMLP is not None:
            # Mạng dự đoán vận tốc: Input (Pose + Time + Condition) -> Output (Velocity)
            velocity_net = VelocityMLP(
                input_dim=3 * num_keypoints, # 42
                cond_dim=embed_dims,         # 256 (Feature từ query)
                time_dim=64,
                hidden_dim=512,
                num_layers=3
            )
            self.flow_model = RectifiedFlowWrapper(velocity_net)
        else:
            self.flow_model = None

    def _load_bone_statistics(self):
        """Tải file thống kê độ dài xương để dùng cho BoneLoss."""
        bone_stats_path = 'gt_bone_stats.json'
        if os.path.exists(bone_stats_path):
            try:
                with open(bone_stats_path, 'r') as f:
                    stats = json.load(f)
                # Đăng ký buffer: không phải tham số học được, nhưng cần lưu trong state_dict
                self.register_buffer('gt_bone_lengths_mean', torch.tensor(stats['mean']))
                print(f"[WiTiDARHead] Đã tải thống kê xương thành công từ {bone_stats_path}")
            except Exception as e:
                print(f"[WiTiDARHead] Lỗi khi đọc file xương: {e}. Bone Loss sẽ bị vô hiệu hóa.")
                self.register_buffer('gt_bone_lengths_mean', torch.zeros(15))
        else:
            print(f"[WiTiDARHead] Cảnh báo: Không tìm thấy '{bone_stats_path}'. Hãy chạy script phân tích trước.")
            # Tạo tensor 0 để tránh lỗi runtime
            self.register_buffer('gt_bone_lengths_mean', torch.zeros(15))

    def init_weights(self):
        """Khởi tạo trọng số mạng."""
        # Init cho regressor
        for m in self.draft_regressor:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        # Init cho classification head (Focal loss bias trick)
        # Giúp training ổn định lúc đầu bằng cách dự đoán xác suất thấp cho foreground
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_head.bias, bias_value)

        # Init cho Queries
        nn.init.normal_(self.query_embedding.weight, std=0.01)

    def forward(self, memory, **kwargs):
        """
        Forward pass.
        Args:
            memory (Tensor): Output từ Backbone/Encoder (WiMamba).
                             Shape: (Batch, Length, Embed_Dims).
        """
        batch_size = memory.size(0)

        # 1. Chuẩn bị Queries: (100, D) -> (B, 100, D)
        query = self.query_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # 2. Cross-Attention: Queries truy vấn thông tin từ Memory
        # [QUAN TRỌNG] Xử lý Mixed Precision (FP16)
        # MultiheadAttention của PyTorch đôi khi không ổn định với FP16 thuần
        if memory.dtype == torch.float16:
            query_fp32 = query.float()
            memory_fp32 = memory.float()
            # query (Target), memory (Source)
            query_feat, _ = self.decoder_attn(query_fp32, memory_fp32, memory_fp32)
            query_feat = query_feat.half() # Ép lại về FP16
        else:
            query_feat, _ = self.decoder_attn(query, memory, memory)

        # Residual connection + Norm
        query_feat = self.decoder_norm(query + query_feat)

        # 3. Dự đoán (Draft Stage)
        cls_scores = self.cls_head(query_feat)          # (B, N, 1)
        draft_preds = self.draft_regressor(query_feat)  # (B, N, 42)

        # Trả về query_feat để dùng làm điều kiện (condition) cho Refine Stage
        return cls_scores, draft_preds, query_feat

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels=None,
                      gt_keypoints=None, gt_areas=None, gt_bboxes_ignore=None,
                      **kwargs):
        """Forward function for training mode."""
        # Nếu x là list/tuple (từ FPN/Backbone), lấy phần tử cuối (hoặc duy nhất)
        if isinstance(x, (list, tuple)):
            feat = x[-1]
        else:
            feat = x

        # Forward qua mạng
        outs = self(feat)

        # Tính toán Loss
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_keypoints, gt_areas, img_metas)
        losses = self.loss(*loss_inputs)
        return losses

    @force_fp32(apply_to=('cls_scores', 'draft_preds'))
    def loss(self, cls_scores, draft_preds, query_feat,
             gt_bboxes, gt_labels, gt_keypoints, gt_areas, img_metas):
        """Tính toán tổng hợp các loại Loss."""
        num_imgs = cls_scores.size(0)

        # Reshape draft_preds từ (B, N, 42) -> (B, N, 14, 3) để khớp định dạng của Assigner
        kpt_preds_reshaped = draft_preds.reshape(num_imgs, self.num_query, self.num_keypoints, 3)

        # Chuẩn bị list ground truths
        gt_labels_list = [gt_labels[i] for i in range(num_imgs)]
        gt_keypoints_list = [gt_keypoints[i] for i in range(num_imgs)]
        gt_areas_list = [gt_areas[i] for i in range(num_imgs)]

        losses_cls = []
        losses_kpt = []
        losses_bone = []
        losses_flow = []

        for i in range(num_imgs):
            # --- 1. Hungarian Matching ---
            # Tìm cặp (Prediction <-> Ground Truth) tối ưu
            assign_result = self.assigner.assign(
                cls_scores[i],
                kpt_preds_reshaped[i],
                gt_labels_list[i],
                gt_keypoints_list[i],
                gt_areas_list[i],
                img_metas[i]
            )
            # Sample để lấy ra các chỉ số positive/negative
            sampling_result = self.sampler.sample(
                assign_result,
                kpt_preds_reshaped[i],
                gt_keypoints_list[i]
            )

            pos_inds = sampling_result.pos_inds
            # Tránh chia cho 0
            num_pos = max(len(pos_inds), 1)

            # --- 2. Classification Loss ---
            # Tạo target labels (mặc định là background)
            labels_target = gt_labels[i].new_full((self.num_query,), self.num_classes, dtype=torch.long)
            # Gán label thật cho các query dương tính
            labels_target[pos_inds] = gt_labels_list[i][sampling_result.pos_assigned_gt_inds]

            losses_cls.append(self.loss_cls(cls_scores[i], labels_target, avg_factor=num_pos))

            # --- 3. Regression Losses (Chỉ tính cho Positive Queries) ---
            if len(pos_inds) > 0:
                pos_gt_kpts = gt_keypoints_list[i][sampling_result.pos_assigned_gt_inds]
                pos_kpt_pred = kpt_preds_reshaped[i][pos_inds]

                # A. Keypoint Coordinate Loss (L1) - Draft Stage
                # Weight = 1.0 cho tất cả khớp (có thể thay đổi nếu có visibility flag)
                kpt_weight = torch.ones_like(pos_kpt_pred)
                losses_kpt.append(self.loss_kpt(
                    pos_kpt_pred, pos_gt_kpts, kpt_weight, avg_factor=num_pos
                ))

                # B. Bone Length Loss - Anatomy Constraint
                # Tính sai số độ dài xương ngay trên bản phác thảo
                losses_bone.append(self.loss_bone(
                    pos_kpt_pred, self.gt_bone_lengths_mean
                ))

                # C. Flow Matching Loss - Refine Stage
                if self.flow_model is not None:
                    # Flatten về (N_pos, 42)
                    pos_gt_flat = pos_gt_kpts.reshape(-1, self.num_keypoints * 3)

                    # Detach gradient của draft:
                    # Flow chỉ học cách sửa sai, không ảnh hưởng ngược lại quá trình tạo draft
                    pos_draft_flat = draft_preds[i][pos_inds].detach()

                    # Lấy feature điều kiện tương ứng với query dương tính
                    pos_cond = query_feat[i][pos_inds]

                    flow_loss = self.flow_model.get_train_loss(
                        x_1=pos_gt_flat,        # Đích đến (GT)
                        condition=pos_cond,     # Ngữ cảnh
                        x_0=pos_draft_flat      # Điểm bắt đầu (Draft)
                    )
                    losses_flow.append(flow_loss * self.loss_flow_weight)
            else:
                # Nếu không có mẫu dương nào, loss = 0 (giữ đồ thị tính toán)
                losses_kpt.append(cls_scores[i].sum() * 0)
                losses_bone.append(cls_scores[i].sum() * 0)
                losses_flow.append(cls_scores[i].sum() * 0)

        return dict(
            loss_cls=sum(losses_cls) / num_imgs,
            loss_kpt=sum(losses_kpt) / num_imgs,
            loss_bone=sum(losses_bone) / num_imgs,
            loss_flow=sum(losses_flow) / num_imgs
        )

    def simple_test(self, feats, img_metas, rescale=False):
        """Inference function."""
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]

        # Forward
        outs = self(feats)

        # Post-process (Decoding)
        results = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results

    def get_bboxes(self, cls_scores, draft_preds, query_feat, img_metas, rescale=False):
        """Chuyển đổi raw output thành kết quả cuối cùng (bboxes, kpts)."""
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            draft_pred = draft_preds[img_id]
            cond = query_feat[img_id]

            # 1. Lọc theo điểm số phân loại
            scores = cls_score.sigmoid().view(-1)
            # Lấy max_per_img từ config test (mặc định 10 người/ảnh)
            max_per_img = self.test_cfg.get('max_per_img', 10)
            scores, indexes = scores.topk(max_per_img)

            # Lấy các ứng viên tốt nhất
            top_draft = draft_pred[indexes]
            top_cond = cond[indexes]

            # 2. Refinement (Nếu có Flow Model)
            if self.flow_model is not None:
                # Dùng ODE Solver (Euler) để di chuyển từ Draft -> Refined Pose
                # num_steps=1 để chạy nhanh (one-step generation)
                refined_pred = self.flow_model.sample(
                    x_init=top_draft,
                    condition=top_cond,
                    num_steps=1
                )
            else:
                refined_pred = top_draft

            # 3. Định dạng lại kết quả
            det_kpts = refined_pred.reshape(-1, self.num_keypoints, 3)

            # Tạo dummy bbox (vì bài toán này tập trung vào pose)
            # Định dạng bbox: [x1, y1, x2, y2, score]
            det_bboxes = torch.zeros((len(scores), 5), device=scores.device)
            det_bboxes[:, 4] = scores # Cột cuối là score

            # Label mặc định là 0 (person)
            det_labels = torch.zeros((len(scores),), dtype=torch.long, device=scores.device)

            result_list.append((det_bboxes, det_labels, det_kpts))

        return result_list