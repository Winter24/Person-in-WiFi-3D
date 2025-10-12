# opera/models/losses/bone_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from mmdet.models.losses.utils import weighted_loss

# Định nghĩa lại cấu trúc xương ở đây để module này độc lập
# Đảm bảo nó phải khớp 100% với script phân tích!
BONES_WIFI_14KPT = [
    (13, 12), (0, 13), (1, 13), (0, 1), (6, 13), (7, 13), (6, 7),
    (0, 2), (2, 4), (1, 3), (3, 5), (6, 8), (8, 10), (7, 9), (9, 11)
]

# Đây là hàm loss cơ bản, sẽ được gọi bởi lớp loss bên dưới
@weighted_loss
def bone_length_loss(pred_lengths, target_lengths):
    """
    Hàm tính L1 loss giữa chiều dài xương dự đoán và mục tiêu.
    Args:
        pred_lengths (Tensor): Chiều dài xương dự đoán, shape (N, num_bones).
        target_lengths (Tensor): Chiều dài xương mục tiêu, shape (N, num_bones).
    Returns:
        Tensor: Loss tensor.
    """
    loss = F.l1_loss(pred_lengths, target_lengths, reduction='none')
    return loss

@LOSSES.register_module()
class BoneLengthLoss(nn.Module):
    """
    Loss phạt sự chênh lệch giữa chiều dài xương dự đoán và chiều dài trung bình.
    Args:
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):
        super(BoneLengthLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.bones = torch.tensor(BONES_WIFI_14KPT, dtype=torch.long)

    def forward(self,
                pred_kpts,             # shape (num_pos_samples, num_keypoints, 3)
                target_lengths_mean,   # shape (num_bones,)
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """
        Forward function.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        # Chuyển BONES tensor lên cùng device với pred_kpts nếu cần
        if self.bones.device != pred_kpts.device:
            self.bones = self.bones.to(pred_kpts.device)

        num_pos_samples = pred_kpts.size(0)
        if num_pos_samples == 0:
            return pred_kpts.sum() * 0 # Trả về loss 0 nếu không có mẫu dương

        # Lấy tọa độ của các điểm đầu và cuối của xương
        p1_coords = pred_kpts[:, self.bones[:, 0], :3]  # shape (num_pos, num_bones, 3)
        p2_coords = pred_kpts[:, self.bones[:, 1], :3]  # shape (num_pos, num_bones, 3)

        # Tính chiều dài xương dự đoán
        pred_lengths = torch.norm(p1_coords - p2_coords, p=2, dim=-1) # shape (num_pos, num_bones)

        # Expand target_lengths_mean để khớp shape với pred_lengths
        target_lengths_expanded = target_lengths_mean.expand_as(pred_lengths)

        # Tính loss bằng hàm bone_length_loss đã được trang trí (decorated)
        loss = self.loss_weight * bone_length_loss(
            pred_lengths,
            target_lengths_expanded,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor)
        
        return loss