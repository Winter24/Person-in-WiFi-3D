# opera/models/losses/bone_loss.py (PHIÊN BẢN CUỐI CÙNG - XỬ LÝ BẤT ĐỐI XỨNG)

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from mmdet.models.losses.utils import weighted_loss

# Ánh xạ đã xác minh
# 0: Head, 1: Neck, 2: R_Shoulder, 3: L_Shoulder, 4: R_Elbow, 5: L_Elbow, 6: R_Hip, 
# 7: L_Wrist, 8: R_Wrist, 9: R_Knee, 10: R_Ankle, 11: L_Knee, 12: L_Hip, 13: L_Ankle

# Định nghĩa các xương sẽ được sử dụng trong loss
# Chúng ta sẽ sử dụng tất cả các xương chi và xương ngang
TARGET_BONES = [
    (3, 2),    # LShoulder <-> RShoulder
    (12, 6),   # LHip <-> RHip
    (3, 5),    # LShoulder -> LElbow (Tay Trái)
    (5, 7),    # LElbow -> LWrist
    (2, 4),    # RShoulder -> RElbow (Tay Phải)
    (4, 8),    # RElbow -> RWrist
    (12, 11),  # LHip -> LKnee (Chân Trái)
    (11, 13),  # LKnee -> LAnkle
    (6, 9),    # RHip -> RKnee (Chân Phải)
    (9, 10),   # RKnee -> RAnkle
]

# Định nghĩa các cặp xương đối xứng để lấy trung bình
# (chỉ số trong TARGET_BONES)
SYMMETRIC_PAIRS = {
    2: 4,  # L_UpperArm <-> R_UpperArm
    3: 5,  # L_LowerArm <-> R_LowerArm
    6: 8,  # L_Thigh <-> R_Thigh
    7: 9   # L_Calf <-> R_Calf
}

# Định nghĩa tất cả các xương để map với file JSON
BONES_FULL_BODY = [
    (0, 1), (1, 3), (1, 2), (3, 2), (3, 12), (2, 6), (12, 6),
    (3, 5), (5, 7), (2, 4), (4, 8), (12, 11), (11, 13), (6, 9), (9, 10)
]

@weighted_loss
def bone_length_loss_l1(pred_lengths, target_lengths):
    loss = F.l1_loss(pred_lengths, target_lengths, reduction='none')
    return loss

@LOSSES.register_module()
class BoneLengthLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):
        super(BoneLengthLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.target_bones_tensor = torch.tensor(TARGET_BONES, dtype=torch.long)
        self.json_indices = self._create_json_mapping()

    def _create_json_mapping(self):
        """Tạo mapping từ TARGET_BONES tới chỉ số trong file JSON."""
        mapping = []
        for target_bone in TARGET_BONES:
            for i, full_bone in enumerate(BONES_FULL_BODY):
                if (target_bone[0] == full_bone[0] and target_bone[1] == full_bone[1]) or \
                   (target_bone[0] == full_bone[1] and target_bone[1] == full_bone[0]):
                    mapping.append(i)
                    break
        return torch.tensor(mapping, dtype=torch.long)

    def _create_symmetrized_targets(self, full_target_lengths_mean):
        """Tạo ra một bộ target lengths đã được đối xứng hóa."""
        reliable_targets = full_target_lengths_mean[self.json_indices].clone()
        for idx_left, idx_right in SYMMETRIC_PAIRS.items():
            avg_len = (reliable_targets[idx_left] + reliable_targets[idx_right]) / 2.0
            reliable_targets[idx_left] = avg_len
            reliable_targets[idx_right] = avg_len
        return reliable_targets

    def forward(self,
                pred_kpts,
                full_target_lengths_mean,
                weight=None, avg_factor=None, reduction_override=None, **kwargs):
        
        reduction = (reduction_override if reduction_override else self.reduction)
        
        if self.target_bones_tensor.device != pred_kpts.device:
            self.target_bones_tensor = self.target_bones_tensor.to(pred_kpts.device)
        
        num_pos_samples = pred_kpts.size(0)
        if num_pos_samples == 0:
            return pred_kpts.sum() * 0

        # Tạo target lengths đã được đối xứng hóa
        symmetrized_target_lengths = self._create_symmetrized_targets(full_target_lengths_mean.to(pred_kpts.device))

        p1_coords = pred_kpts[:, self.target_bones_tensor[:, 0], :3]
        p2_coords = pred_kpts[:, self.target_bones_tensor[:, 1], :3]
        pred_lengths = torch.norm(p1_coords - p2_coords, p=2, dim=-1)
        
        target_lengths_expanded = symmetrized_target_lengths.expand_as(pred_lengths)

        loss = self.loss_weight * bone_length_loss_l1(
            pred_lengths,
            target_lengths_expanded,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor)
        
        return loss