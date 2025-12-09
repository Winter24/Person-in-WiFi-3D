# %%writefile /content/Person-in-WiFi-3D/opera/core/bbox/match_costs/match_cost.py
# @title opera/core/bbox/match_costs/match_cost.py

import torch
import numpy as np
import torch.nn.functional as F
from .builder import MATCH_COST


@MATCH_COST.register_module()
class KptL1Cost(object):
    """KptL1Cost."""
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, kpt_pred, gt_keypoints, valid_kpt_flag):
        kpt_cost = []
        for i in range(len(gt_keypoints)):
            kpt_pred_tmp = kpt_pred.clone()
            valid_flag = valid_kpt_flag[i] > 0
            valid_flag_expand = valid_flag.unsqueeze(0).unsqueeze(
                -1).expand_as(kpt_pred_tmp)
            kpt_pred_tmp[~valid_flag_expand] = 0

            # --- FIX: Ép kiểu float() trước khi tính cdist ---
            cost = torch.cdist(
                kpt_pred_tmp.reshape(kpt_pred_tmp.shape[0], -1).float(),
                gt_keypoints[i].reshape(-1).unsqueeze(0).float(),
                p=1)
            # -----------------------------------------------

            avg_factor = torch.clamp(valid_flag.float().sum() * 2, 1.0)
            cost = cost / avg_factor
            kpt_cost.append(cost)
        kpt_cost = torch.cat(kpt_cost, dim=1)
        return kpt_cost * self.weight

@MATCH_COST.register_module()
class KptMSECost(object):
    """KptMSECost."""
    def __init__(self, weight=1.0, align=None):
        self.weight = weight
        self.align = align

    def __call__(self, kpt_pred, gt_keypoints, valid_kpt_flag):
        kpt_cost = []
        for i in range(len(gt_keypoints)):
            kpt_pred_tmp = kpt_pred.clone()
            valid_flag = valid_kpt_flag[i] > 0
            valid_flag_expand = valid_flag.unsqueeze(0).unsqueeze(
                -1).expand_as(kpt_pred_tmp)
            kpt_pred_tmp[~valid_flag_expand] = 0

            # --- FIX: Ép kiểu float() trước khi tính cdist ---
            cost = torch.cdist(
                kpt_pred_tmp.reshape(kpt_pred_tmp.shape[0], -1).float(),
                gt_keypoints[i].reshape(-1).unsqueeze(0).float(),
                p=2)
            # -----------------------------------------------

            avg_factor = torch.clamp(valid_flag.float().sum() * 2, 1.0)
            cost = cost / avg_factor
            kpt_cost.append(cost)
        kpt_cost = torch.cat(kpt_cost, dim=1)
        return kpt_cost * self.weight


@MATCH_COST.register_module()
class OksCost(object):
    # Class này giữ nguyên, không cần sửa vì nó tính thủ công
    def __init__(self, num_keypoints=17, weight=1.0):
        self.weight = weight
        if num_keypoints == 17:
            self.sigmas = np.array([
                .26,
                .25, .25,
                .35, .35,
                .79, .79,
                .72, .72,
                .62, .62,
                1.07, 1.07,
                .87, .87,
                .89, .89], dtype=np.float32) / 10.0
        elif num_keypoints == 14:
            self.sigmas = np.array([
                1.0, 1.0,
                1.0, 1.0,
                1.0, 1.0,
                1.0, 1.0,
                1.0, 1.0,
                1.0, 1.0,
                1.0, 1.0], dtype=np.float32) / 10.0
        else:
            raise ValueError(f'Unsupported keypoints number {num_keypoints}')

    def __call__(self, kpt_pred, gt_keypoints, valid_kpt_flag, gt_areas):
        sigmas = torch.from_numpy(self.sigmas).to(kpt_pred.device)
        variances = (sigmas * 2)**2

        oks_cost = []
        assert len(gt_keypoints) == len(gt_areas)
        for i in range(len(gt_keypoints)):
            # Chắc chắn tính toán ở float32 để tránh lỗi
            kpt_pred_i = kpt_pred.float()
            gt_keypoints_i = gt_keypoints[i].float()

            squared_distance = \
                (kpt_pred_i[:, :, 0] - gt_keypoints_i[:, 0].unsqueeze(0)) ** 2 + \
                (kpt_pred_i[:, :, 1] - gt_keypoints_i[:, 1].unsqueeze(0)) ** 2
            vis_flag = (valid_kpt_flag[i] > 0).int()
            vis_ind = vis_flag.nonzero(as_tuple=False)[:, 0]
            num_vis_kpt = vis_ind.shape[0]
            assert num_vis_kpt > 0
            area = gt_areas[i]

            squared_distance0 = squared_distance / (area * variances * 2)
            squared_distance0 = squared_distance0[:, vis_ind]
            squared_distance1 = torch.exp(-squared_distance0).sum(
                dim=1, keepdim=True)
            oks = squared_distance1 / num_vis_kpt
            oks_cost.append(-oks)
        oks_cost = torch.cat(oks_cost, dim=1)
        return oks_cost * self.weight