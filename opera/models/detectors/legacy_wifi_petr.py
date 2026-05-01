# Copyright (c) Hikvision Research Institute. All rights reserved.
"""Legacy WiFi PETR detector used by the original linear-input baseline.

Older WiFi checkpoints declared a ResNet/ChannelMapper backbone in the config,
but the PETR detector forward path bypassed that backbone and projected the CSI
tokens directly with a top-level ``head = Linear(60, 256)``.  Keep that behavior
isolated here so old M0 checkpoints can be loaded without changing the current
PETR implementation used by the new M0-M4 models.
"""

import warnings

import numpy as np
import torch
from mmcv.cnn import Linear
from mmdet.core import bbox_mapping_back, multiclass_nms
from mmdet.models.detectors.detr import DETR
from mmdet.models.detectors.single_stage import SingleStageDetector

from opera.core.keypoint import bbox_kpt2result, kpt_mapping_back
from ..builder import DETECTORS


@DETECTORS.register_module()
class LegacyWifiPETR(DETR):
    """PETR compatibility wrapper for old WiFi linear-projection checkpoints."""

    def __init__(self, *args, **kwargs):
        # Match the legacy constructor exactly: build the configured backbone,
        # neck, and bbox_head via SingleStageDetector, then add the top-level
        # CSI projection layer that the old forward path actually used.
        super(DETR, self).__init__(*args, **kwargs)
        self.head = Linear(60, 256)

    def extract_feat(self, img):
        """Project CSI tokens with the legacy top-level linear head.

        Expected input shape after the WiFi dataset pipeline:
        ``(B, 3, 3, 20, 60)``.  The old model flattens the first three spatial
        axes to 180 tokens and applies ``Linear(60, 256)``.
        """
        if img.dim() != 5:
            raise RuntimeError(
                'LegacyWifiPETR expects 5D WiFi CSI input '
                f'(B, 3, 3, 20, 60), got shape {tuple(img.shape)}')
        batch_size, _, _, _, channel = img.shape
        if channel != 60:
            raise RuntimeError(
                'LegacyWifiPETR expects the last CSI dimension to be 60, '
                f'got {channel}.')
        x = img.reshape(batch_size, -1, channel)
        return self.head(x)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_keypoints,
                      gt_areas,
                      gt_bboxes_ignore=None):
        super(SingleStageDetector, self).forward_train(img, img_metas)
        feat = self.extract_feat(img)
        losses = self.bbox_head.forward_train(feat, img_metas, gt_bboxes,
                                              gt_labels, gt_keypoints,
                                              gt_areas, gt_bboxes_ignore)
        return losses

    def forward_test(self, imgs, img_metas, **kwargs):
        """Normalize WiFi inference inputs without relying on image TTA paths."""
        if isinstance(imgs, list):
            assert len(imgs) == 1, 'WiFi inference only supports one test-time augmentation.'
            img = imgs[0]
        else:
            img = imgs

        if hasattr(img_metas, 'data'):
            img_metas = img_metas.data[0]
        elif isinstance(img_metas, list) and img_metas and hasattr(img_metas[0], 'data'):
            img_metas = img_metas[0].data[0]
        elif isinstance(img_metas, tuple):
            img_metas = list(img_metas)

        if isinstance(img_metas, list) and len(img_metas) == 1 and isinstance(img_metas[0], list):
            img_metas = img_metas[0]
        elif isinstance(img_metas, dict):
            img_metas = [img_metas]

        for img_meta in img_metas:
            img_meta['batch_input_shape'] = tuple(img.size()[-2:])

        if 'proposals' in kwargs and isinstance(kwargs['proposals'], list):
            kwargs['proposals'] = kwargs['proposals'][0]
        return self.simple_test(img, img_metas, **kwargs)

    def forward_dummy(self, img):
        """Used for computing network FLOPs."""
        warnings.warn('Warning! MultiheadAttention in DETR does not support '
                      'FLOPs computation. Do not use these results in papers.')
        batch_size = img.shape[0]
        dummy_img_metas = [
            dict(
                batch_input_shape=tuple(img.size()[-2:]),
                img_shape=tuple(img.size()[-2:]) + (3,),
                scale_factor=np.array([1., 1., 1., 1.]))
            for _ in range(batch_size)
        ]
        feat = self.extract_feat(img)
        return self.bbox_head(feat, img_metas=dummy_img_metas)

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation."""
        batch_size = len(img_metas)
        assert batch_size == 1, 'Currently only batch_size 1 for inference ' \
            f'mode is supported. Found batch_size {batch_size}.'

        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)

        return [
            bbox_kpt2result(det_bboxes, det_labels, det_kpts,
                            self.bbox_head.num_classes)
            for det_bboxes, det_labels, det_kpts in results_list
        ]

    def merge_aug_results(self, aug_bboxes, aug_kpts, aug_scores, img_metas):
        recovered_bboxes = []
        recovered_kpts = []
        for bboxes, kpts, img_info in zip(aug_bboxes, aug_kpts, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                       flip_direction)
            kpts = kpt_mapping_back(kpts, img_shape, scale_factor, flip,
                                    flip_direction)
            recovered_bboxes.append(bboxes)
            recovered_kpts.append(kpts)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        kpts = torch.cat(recovered_kpts, dim=0)
        if aug_scores is None:
            return bboxes, kpts
        scores = torch.cat(aug_scores, dim=0)
        return bboxes, kpts, scores

    def aug_test(self, imgs, img_metas, rescale=False):
        feats = self.extract_feats(imgs)
        aug_bboxes = []
        aug_scores = []
        aug_kpts = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x, img_meta)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=False)

            for det_bboxes, det_labels, det_kpts in bbox_list:
                aug_bboxes.append(det_bboxes[:, :4])
                aug_scores.append(det_bboxes[:, 4])
                aug_kpts.append(det_kpts[..., :2])

        merged_bboxes, merged_kpts, merged_scores = self.merge_aug_results(
            aug_bboxes, aug_kpts, aug_scores, img_metas)

        merged_scores = merged_scores.unsqueeze(1)
        padding = merged_scores.new_zeros(merged_scores.shape[0], 1)
        merged_scores = torch.cat([merged_scores, padding], dim=-1)
        det_bboxes, det_labels, keep_inds = multiclass_nms(
            merged_bboxes,
            merged_scores,
            self.test_cfg.score_thr,
            self.test_cfg.nms,
            self.test_cfg.max_per_img,
            return_inds=True)
        det_kpts = merged_kpts[keep_inds]
        det_kpts = torch.cat(
            (det_kpts, det_kpts.new_ones(det_kpts[..., :1].shape)), dim=2)

        return [
            bbox_kpt2result(det_bboxes, det_labels, det_kpts,
                            self.bbox_head.num_classes)
        ]
