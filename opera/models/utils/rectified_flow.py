# @title opera/models/utils/rectified_flow.py
# %%writefile /content/Person-in-WiFi-3D/opera/models/utils/rectified_flow.py
import torch
import torch.nn as nn
import numpy as np

class RectifiedFlowWrapper(nn.Module):
    """
    Rectified Flow Wrapper for Pose Refinement.

    Paper Reference: "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (ICLR 2023)

    Nhiệm vụ:
    1. Training: Tạo đường dẫn nội suy thẳng (Straight Path Interpolation) giữa X0 (Nhiễu/Draft) và X1 (GT).
                 Huấn luyện mạng velocity_net để dự đoán vector chỉ hướng (X1 - X0).
    2. Inference: Dùng Euler method để di chuyển từ X0 đến X1 theo hướng velocity_net dự đoán.
    """
    def __init__(self, velocity_net, sigma_min=1e-5):
        """
        Args:
            velocity_net (nn.Module): Mạng dự đoán vận tốc v(x, t, cond).
            sigma_min (float): Giá trị nhỏ để tránh lỗi chia cho 0 hoặc log(0) (nếu cần).
        """
        super().__init__()
        self.velocity_net = velocity_net
        self.sigma_min = sigma_min

    def get_train_loss(self, x_1, condition, x_0=None):
        """
        Tính Loss Flow Matching.

        Args:
            x_1 (Tensor): Ground Truth Pose (Đích đến). Shape: (B, N, D).
            condition (Tensor): WiFi Features (Điều kiện). Shape: (B, N, Cond_Dim).
            x_0 (Tensor, optional): Điểm bắt đầu.
                - Nếu None: Sample từ N(0, I) -> Standard Generation.
                - Nếu Tensor (Draft Pose): -> Refinement / Speculative Decoding.

        Returns:
            loss (Tensor): Scalar loss.
        """
        if x_1.dim() == 2:
            # Trường hợp (Total_Pos_Samples, Dim) -> Coi như (B, 1, D)
            x_1 = x_1.unsqueeze(1)
            condition = condition.unsqueeze(1)
            if x_0 is not None:
                x_0 = x_0.unsqueeze(1)

        B, N, D = x_1.shape
        device = x_1.device
        dtype = x_1.dtype

        # 1. Xác định điểm bắt đầu x_0
        if x_0 is None:
            # Trường hợp sinh từ nhiễu trắng
            x_0 = torch.randn_like(x_1)
        else:
            # Trường hợp Refinement (Speculative)
            # Quan trọng: Thêm nhiễu Gaussian nhẹ vào Draft Pose để làm giàu phân phối training.
            # Nếu không thêm nhiễu, mô hình có thể bị overfitting vào lỗi cụ thể của Draft Head
            # và không học được trường vector tổng quát.
            # Sigma=0.1 tương đương biên độ sai số khoảng 10-20mm (nếu data đã normalize).
            noise_strength = 0.1
            x_0 = x_0 + torch.randn_like(x_0) * noise_strength

        # 2. Sample thời gian t ~ Uniform[0, 1]
        # Shape (B, 1, 1) để broadcast cho phép nhân với (B, N, D)
        t = torch.rand(B, 1, 1, device=device, dtype=dtype)

        # 3. Tạo mẫu nội suy (Interpolated Sample) x_t
        # Đường thẳng nối x_0 và x_1: X_t = t * X_1 + (1 - t) * X_0
        # Tại t=0 -> X_0. Tại t=1 -> X_1.
        x_t = t * x_1 + (1 - t) * x_0

        # 4. Tính Vận tốc Mục tiêu (Target Velocity)
        # Đạo hàm của X_t theo t là: d(X_t)/dt = X_1 - X_0
        target_v = x_1 - x_0

        # 5. Dự đoán Vận tốc
        # Mạng nhận vào: (Pose nhiễu x_t, Thời gian t, Điều kiện WiFi)
        # Lưu ý: t cần flatten thành (B,) để đưa vào Time Embedding
        pred_v = self.velocity_net(x_t, t.view(B), condition)

        # 6. Tính Loss (Conditional Flow Matching Loss)
        # L_CFM = ||v_theta(x_t, t) - (x_1 - x_0)||^2
        # Dùng MSE Loss (L2)
        loss = torch.mean((pred_v - target_v) ** 2)

        return loss

    @torch.no_grad()
    def sample(self, x_init, condition, num_steps=1):
        """
        Sinh mẫu (Inference) bằng phương pháp Euler.
        """
        # [FIX] Xử lý shape linh hoạt (2D -> 3D)
        is_2d = False
        if x_init.dim() == 2:
            is_2d = True
            x_init = x_init.unsqueeze(1) # (N_queries, 1, 42)
            condition = condition.unsqueeze(1)

        B = x_init.shape[0]
        device = x_init.device
        dtype = x_init.dtype

        x = x_init
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_value = i / num_steps
            t = torch.full((B,), t_value, device=device, dtype=dtype)

            # Dự đoán vận tốc
            v = self.velocity_net(x, t, condition)

            # Cập nhật
            x = x + v * dt

        # [FIX] Trả về shape gốc nếu đầu vào là 2D
        if is_2d:
            x = x.squeeze(1)

        return x