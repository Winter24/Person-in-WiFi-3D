# opera/models/losses/bone_loss.py (PHIÊN BẢN CUỐI CÙNG)

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from mmdet.models.losses.utils import weighted_loss

# =================================================================
# == ĐỊNH NGHĨA XƯƠNG (PHIÊN BẢN ĐÃ XÁC MINH) ==
# =================================================================
# Ánh xạ đã xác minh:
# 0: Head, 1: Neck, 2: R_Shoulder, 3: L_Shoulder, 4: R_Elbow, 5: L_Elbow, 
# 6: R_Hip, 7: L_Wrist, 8: R_Wrist, 9: R_Knee, 10: R_Ankle, 
# 11: L_Knee, 12: L_Hip, 13: L_Ankle

# Đây là định nghĩa đầy đủ, dùng để ánh xạ với file JSON
BONES_FULL_BODY = [
    (0, 1), (1, 3), (1, 2), (3, 2), (3, 12), (2, 6), (12, 6),
    (3, 5), (5, 7), (2, 4), (4, 8), (12, 11), (11, 13), (6, 9), (9, 10)
]

# Đây là định nghĩa các xương đáng tin cậy mà chúng ta sẽ dùng để tính loss
RELIABLE_BONES = [
    (3, 2),    # L_Shoulder <-> R_Shoulder
    (12, 6),   # L_Hip <-> R_Hip
    (3, 5),    # L_Shoulder -> L_Elbow
    (5, 7),    # L_Elbow -> L_Wrist
    (2, 4),    # R_Shoulder -> R_Elbow
    (4, 8),    # R_Elbow -> R_Wrist
    (12, 11),  # L_Hip -> L_Knee
    (11, 13),  # L_Knee -> L_Ankle
    (6, 9),    # R_Hip -> R_Knee
    (9, 10),   # R_Knee -> R_Ankle
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
        
        # Tạo tensor cho các xương đáng tin cậy
        self.reliable_bones_tensor = torch.tensor(RELIABLE_BONES, dtype=torch.long)
        
        # Tạo mapping để lấy đúng giá trị mean từ file JSON
        self.json_indices = []
        for reliable_bone in RELIABLE_BONES:
            # Tìm chỉ số của xương này trong danh sách đầy đủ
            for i, full_bone in enumerate(BONES_FULL_BODY):
                # Kiểm tra cả hai chiều (p1, p2) và (p2, p1)
                if (reliable_bone[0] == full_bone[0] and reliable_bone[1] == full_bone[1]) or \
                   (reliable_bone[0] == full_bone[1] and reliable_bone[1] == full_bone[0]):
                    self.json_indices.append(i)
                    break
        self.json_indices = torch.tensor(self.json_indices, dtype=torch.long)

    def forward(self,
                pred_kpts,             # shape (num_pos_samples, 14, 3)
                full_target_lengths_mean, # shape (len(BONES_FULL_BODY),)
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        
        # Chuyển các tensor lên đúng device
        if self.reliable_bones_tensor.device != pred_kpts.device:
            self.reliable_bones_tensor = self.reliable_bones_tensor.to(pred_kpts.device)
            self.json_indices = self.json_indices.to(pred_kpts.device)

        num_pos_samples = pred_kpts.size(0)
        if num_pos_samples == 0:
            return pred_kpts.sum() * 0

        # Lấy giá trị mean tương ứng với các xương đáng tin cậy
        reliable_target_lengths_mean = full_target_lengths_mean[self.json_indices]
        
        # Tính chiều dài xương từ dự đoán
        p1_coords = pred_kpts[:, self.reliable_bones_tensor[:, 0], :3]
        p2_coords = pred_kpts[:, self.reliable_bones_tensor[:, 1], :3]
        pred_lengths = torch.norm(p1_coords - p2_coords, p=2, dim=-1)
        
        target_lengths_expanded = reliable_target_lengths_mean.expand_as(pred_lengths)

        loss = self.loss_weight * bone_length_loss_l1(
            pred_lengths,
            target_lengths_expanded,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor)
        
        return loss