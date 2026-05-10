# opera/models/dense_heads/wi_tidar_head.py
import json
import os

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, force_fp32

from opera.core.bbox import build_assigner, build_sampler
from ..builder import HEADS, build_loss
from ..utils import build_transformer_layer_sequence

try:
    from opera.models.utils.rectified_flow import RectifiedFlowWrapper
    from opera.models.dense_heads.flow_components import VelocityMLP
except ImportError:
    RectifiedFlowWrapper = None
    VelocityMLP = None

try:
    from opera.models.backbones.wimamba import WiMambaEncoder
except ImportError:
    WiMambaEncoder = None


@HEADS.register_module()
class WiTiDARHead(BaseModule):
    def __init__(self,
                 num_query=100,
                 embed_dims=256,
                 num_keypoints=14,
                 num_classes=1,
                 in_channels=2048,
                 sync_cls_avg_factor=True,
                 with_kpt_refine=True,
                 as_two_stage=True,
                 transformer=None,
                 transformer_encoder=None,
                 mamba_cfg=dict(
                     num_layers=4,
                     d_state=16,
                     d_conv=4,
                     expand=2,
                     dropout=0.1),
                 loss_cls=dict(
                     type='mmdet.FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
                 loss_kpt=dict(type='mmdet.L1Loss', loss_weight=5.0),
                 loss_bone=dict(type='BoneLengthLoss', loss_weight=2.0),
                 loss_flow_weight=10.0,
                 flow_refine_mode='rectified_flow',
                 flow_num_steps=1,
                 flow_noise_strength=0.1,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)

        self.num_query = num_query
        self.embed_dims = embed_dims
        self.num_keypoints = num_keypoints
        self.num_classes = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_flow_weight = loss_flow_weight
        if flow_refine_mode not in ('none', 'rectified_flow'):
            raise ValueError(f'Unsupported flow_refine_mode: {flow_refine_mode}')
        self.flow_refine_mode = flow_refine_mode
        self.flow_num_steps = flow_num_steps

        if train_cfg:
            self.assigner = build_assigner(train_cfg['assigner'])
            sampler_cfg = dict(type='mmdet.PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.loss_cls = build_loss(loss_cls)
        self.loss_kpt = build_loss(loss_kpt)
        if loss_bone is not None:
            self.loss_bone = build_loss(loss_bone)
        else:
            self.loss_bone = None

        bone_stats, has_bone_stats = self._load_bone_statistics()
        self.register_buffer('gt_bone_lengths_mean', bone_stats)
        self.has_bone_stats = has_bone_stats

        if transformer_encoder is not None and mamba_cfg is not None:
            raise ValueError('Specify only one of transformer_encoder or mamba_cfg for WiTiDARHead.')

        if transformer_encoder is not None:
            self.encoder = build_transformer_layer_sequence(transformer_encoder)
            self.encoder_type = 'transformer'
        elif WiMambaEncoder is not None and mamba_cfg is not None:
            mamba_cfg_copy = mamba_cfg.copy()
            mamba_cfg_copy.pop('type', None)
            self.encoder = WiMambaEncoder(embed_dims=embed_dims, **mamba_cfg_copy)
            self.encoder_type = 'mamba'
        else:
            print('WARNING: WiMambaEncoder not found or mamba_cfg is None. Using Identity mapping.')
            self.encoder = nn.Identity()
            self.encoder_type = 'identity'

        self.decoder_attn = nn.MultiheadAttention(embed_dims, num_heads=8, batch_first=True)
        self.query_embedding = nn.Embedding(num_query, embed_dims)
        self.decoder_norm = nn.LayerNorm(embed_dims)

        self.draft_regressor = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 3 * num_keypoints))
        self.cls_head = nn.Linear(embed_dims, 1)

        self.flow_model = None
        if flow_refine_mode == 'rectified_flow' and VelocityMLP is not None:
            velocity_net = VelocityMLP(
                input_dim=3 * num_keypoints,
                cond_dim=embed_dims,
                time_dim=64,
                hidden_dim=512,
                num_layers=3)
            self.flow_model = RectifiedFlowWrapper(
                velocity_net,
                noise_strength=flow_noise_strength)

    def _load_bone_statistics(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
        bone_stats_path = os.path.join(project_root, 'gt_bone_stats.json')
        try:
            with open(bone_stats_path, 'r', encoding='utf-8') as f:
                bone_stats = json.load(f)
            print(f'[WiTiDARHead] Loaded gt_bone_stats.json from {bone_stats_path}')
            return torch.tensor(bone_stats['mean'], dtype=torch.float32), True
        except FileNotFoundError:
            print(f'[WiTiDARHead] WARNING: {bone_stats_path} not found. BoneLengthLoss disabled.')
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            print(f'[WiTiDARHead] WARNING: Failed to parse gt_bone_stats.json ({exc}). BoneLengthLoss will be disabled.')
        return torch.zeros(15, dtype=torch.float32), False

    def init_weights(self):
        for m in self.draft_regressor:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_head.bias, bias_value)
        nn.init.normal_(self.query_embedding.weight, std=0.01)

    def forward(self, feat, **kwargs):
        batch_size = feat.size(0)

        if self.encoder_type == 'mamba':
            feat = feat.permute(1, 0, 2)
            memory = self.encoder(feat)
            memory = memory.permute(1, 0, 2)
        elif self.encoder_type == 'transformer':
            feat = feat.permute(1, 0, 2)
            memory = self.encoder(query=feat, key=None, value=None)
            memory = memory.permute(1, 0, 2)
        else:
            memory = feat

        query = self.query_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)

        if memory.dtype == torch.float16:
            query = query.float()
            memory = memory.float()
            query_feat, _ = self.decoder_attn(query, memory, memory)
            query_feat = query_feat.half()
            memory = memory.half()
        else:
            query_feat, _ = self.decoder_attn(query, memory, memory)

        query_feat = self.decoder_norm(query + query_feat)

        cls_scores = self.cls_head(query_feat)
        draft_preds = self.draft_regressor(query_feat)
        return cls_scores, draft_preds, query_feat

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels=None,
                      gt_keypoints=None, gt_areas=None, gt_bboxes_ignore=None,
                      **kwargs):
        if isinstance(x, (list, tuple)):
            feat = x[-1]
        else:
            feat = x

        outs = self(feat)
        loss_inputs = outs + (
            gt_bboxes, gt_labels, gt_keypoints, gt_areas, img_metas,
            gt_bboxes_ignore)
        losses = self.loss(*loss_inputs)
        return losses

    @force_fp32(apply_to=('cls_scores', 'draft_preds'))
    def loss(self, cls_scores, draft_preds, query_feat, gt_bboxes, gt_labels,
             gt_keypoints, gt_areas, img_metas, gt_bboxes_ignore=None):
        num_imgs = cls_scores.size(0)
        kpt_preds_reshaped = draft_preds.reshape(
            num_imgs, self.num_query, self.num_keypoints, 3)

        gt_bboxes_list = [gt_bboxes[i] for i in range(num_imgs)]
        gt_labels_list = [gt_labels[i] for i in range(num_imgs)]
        gt_keypoints_list = [gt_keypoints[i] for i in range(num_imgs)]
        gt_areas_list = [gt_areas[i] for i in range(num_imgs)]

        losses_cls, losses_kpt, losses_bone, losses_flow = [], [], [], []

        for i in range(num_imgs):
            cls_score = cls_scores[i]
            kpt_pred = kpt_preds_reshaped[i]
            draft_pred_flat = draft_preds[i]
            cond = query_feat[i]

            gt_label = gt_labels_list[i]
            gt_kpt = gt_keypoints_list[i]
            gt_area = gt_areas_list[i]
            img_meta = img_metas[i]

            assign_result = self.assigner.assign(
                cls_score, kpt_pred, gt_label, gt_kpt, gt_area, img_meta)
            sampling_result = self.sampler.sample(assign_result, kpt_pred, gt_kpt)
            pos_inds = sampling_result.pos_inds

            labels = gt_label.new_full(
                (self.num_query,), self.num_classes, dtype=torch.long)
            labels[pos_inds] = gt_label[sampling_result.pos_assigned_gt_inds]

            num_pos = max(len(pos_inds), 1)
            loss_cls = self.loss_cls(cls_score, labels, avg_factor=num_pos)
            losses_cls.append(loss_cls)

            if len(pos_inds) > 0:
                pos_gt_kpts = gt_kpt[sampling_result.pos_assigned_gt_inds]
                pos_kpt_pred = kpt_pred[pos_inds]

                loss_kpt = self.loss_kpt(
                    pos_kpt_pred,
                    pos_gt_kpts,
                    torch.ones_like(pos_kpt_pred),
                    avg_factor=num_pos)
                losses_kpt.append(loss_kpt)

                if self.loss_bone is not None:
                    if self.has_bone_stats:
                        loss_bone = self.loss_bone(
                            pos_kpt_pred, self.gt_bone_lengths_mean)
                        losses_bone.append(loss_bone)
                    else:
                        losses_bone.append(cls_score.sum() * 0)
                else:
                    losses_bone.append(cls_score.sum() * 0)

                if self.flow_refine_mode == 'rectified_flow' and self.flow_model is not None:
                    pos_gt_flat = pos_gt_kpts.reshape(-1, 42)
                    pos_draft_flat = draft_pred_flat[pos_inds].detach()
                    pos_cond = cond[pos_inds]
                    loss_flow = self.flow_model.get_train_loss(
                        x_1=pos_gt_flat,
                        condition=pos_cond,
                        x_0=pos_draft_flat)
                    losses_flow.append(loss_flow * self.loss_flow_weight)
                else:
                    losses_flow.append(cls_score.sum() * 0)
            else:
                losses_kpt.append(cls_score.sum() * 0)
                losses_bone.append(cls_score.sum() * 0)
                losses_flow.append(cls_score.sum() * 0)

        loss_dict = dict(
            loss_cls=sum(losses_cls) / num_imgs,
            loss_kpt=sum(losses_kpt) / num_imgs,
            loss_flow=sum(losses_flow) / num_imgs)
        if self.loss_bone is not None:
            loss_dict['loss_bone'] = sum(losses_bone) / num_imgs
        return loss_dict

    def simple_test(self, feats, img_metas, rescale=False):
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        outs = self(feats)
        return self.get_bboxes(*outs, img_metas, rescale=rescale)

    def get_bboxes(self, cls_scores, draft_preds, query_feat, img_metas,
                   rescale=False):
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            draft_pred = draft_preds[img_id]
            cond = query_feat[img_id]

            scores = cls_score.sigmoid().view(-1)
            max_per_img = self.test_cfg.get('max_per_img', 10)
            scores, indexes = scores.topk(max_per_img)

            top_draft = draft_pred[indexes]
            top_cond = cond[indexes]

            if self.flow_refine_mode == 'rectified_flow' and self.flow_model is not None:
                refined_pred = self.flow_model.sample(
                    x_init=top_draft,
                    condition=top_cond,
                    num_steps=self.flow_num_steps)
            else:
                refined_pred = top_draft

            det_kpts = refined_pred.reshape(-1, self.num_keypoints, 3)
            det_bboxes = torch.zeros((len(scores), 5), device=scores.device)
            det_bboxes[:, 4] = scores
            det_labels = torch.zeros(
                (len(scores),), dtype=torch.long, device=scores.device)

            result_list.append((det_bboxes, det_labels, det_kpts))

        return result_list
