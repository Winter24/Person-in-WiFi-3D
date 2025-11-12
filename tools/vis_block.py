import torch
import mmcv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from mmcv import Config
from opera.models import build_model
from mmcv.runner import load_checkpoint

# --- 1. CẤU HÌNH ---
# TODO: Người dùng cần chỉnh sửa hai dòng này
CONFIG_FILE = '/home/winter24/Person-in-WiFi-3D-repo/configs/wifi/petr_wifi.py'
CHECKPOINT_FILE = '/home/winter24/Person-in-WiFi-3D-repo/data/wifipose/result/epoch_5.pth'

# --- 2. HÀM TRỰC QUAN HÓA ---
def visualize_poses_3d(poses, ax, title, num_keypoints=14):
    """
    Vẽ các bộ xương 3D từ tensor tọa độ.
    poses: Tensor có shape (num_persons, num_keypoints * 3)
    """
    ax.clear()
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 2)
    ax.view_init(elev=20, azim=-70) # Đặt góc nhìn cố định

    # Các cặp keypoint để nối thành xương (dựa trên bộ dữ liệu COCO)
    # Bạn có thể cần điều chỉnh lại cho phù hợp với 14 keypoints của dự án
    skeleton_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4), # Đầu và vai
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Tay
        (11, 12), (5, 11), (6, 12), (11, 13), (12, 13) # Hông và chân
    ]
    
    # Giả sử keypoints của bạn có thứ tự tương tự COCO:
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    # 13: left_knee (Giả sử, bạn cần xác nhận lại)
    # Bài báo không nói rõ 14 keypoints là gì, nên đây là giả định hợp lý.

    num_persons = poses.shape[0]
    colors = plt.cm.get_cmap('hsv', num_persons + 1)

    for i, person_pose_flat in enumerate(poses):
        person_pose = person_pose_flat.reshape(num_keypoints, 3)
        x, y, z = person_pose[:, 0], person_pose[:, 1], person_pose[:, 2]
        
        ax.scatter(x, y, z, c=[colors(i)], s=15)

        for conn in skeleton_connections:
            # Kiểm tra xem các chỉ số có nằm trong 14 keypoints không
            if conn[0] < num_keypoints and conn[1] < num_keypoints:
                start_point = person_pose[conn[0]]
                end_point = person_pose[conn[1]]
                ax.plot(
                    [start_point[0], end_point[0]],
                    [start_point[1], end_point[1]],
                    [start_point[2], end_point[2]],
                    color=colors(i)
                )

def forward_and_visualize(model, dummy_input):
    """
    Thực hiện forward pass và trích xuất các output trung gian để trực quan hóa.
    """
    model.eval()
    with torch.no_grad():
        # Trích xuất các khối con cần thiết từ mô hình
        backbone = model.backbone
        neck = model.neck
        head = model.bbox_head
        transformer = head.transformer
        
        # 1. Forward qua Backbone và Neck để lấy feature map
        # Dữ liệu đầu vào 'img' đã được đóng gói trong một list
        features = backbone(dummy_input['img'][0])
        mlvl_feats = neck(features)
        
        # 2. Lấy output từ Encoder (kết quả RPN - Region Proposal Network)
        # Hàm forward của PETRHead rất phức tạp, ta sẽ gọi các phần của nó
        # Dữ liệu hình ảnh meta
        img_metas = dummy_input['img_metas'][0]
        # Chạy một phần của head.forward để lấy kết quả từ encoder
        encoder_results = head.forward_train_rpn(mlvl_feats, img_metas)
        kpt_pred_rpn = encoder_results['kpt_pred'] # (num_queries, num_kpts * 3)

        # 3. Lấy output từ Pose Decoder
        # Chạy một phần của head.forward để lấy kết quả từ decoder chính
        # Chúng ta cần chuẩn bị input cho decoder
        batch_size = mlvl_feats[0].size(0)
        query_embeds = head.query_embedding.weight
        
        hs, init_reference, inter_references = transformer(
            mlvl_feats,
            query_embeds,
            head.positional_encoding(mlvl_feats[-1]),
            reg_branches=head.reg_branches if head.with_box_refine else None,
            cls_branches=head.cls_branches if head.as_two_stage else None,
            img_metas=img_metas,
        )
        # hs là output của decoder, shape (num_layers, batch, num_queries, embed_dim)
        # Lấy output của lớp decoder cuối cùng
        decoder_output = hs[-1] # (batch, num_queries, embed_dim)
        
        # Đưa qua các lớp FFN để ra tọa độ
        kpt_pred_decoder = head.reg_branches[-1](decoder_output).squeeze(0) # (num_queries, num_kpts * 3)

        # 4. Lấy output từ Refine Decoder
        # Giả sử chúng ta tinh chỉnh 3 pose có score cao nhất
        # (Ở đây ta lấy 3 pose đầu tiên để minh họa)
        topk_proposals = kpt_pred_decoder[:3, :]
        
        # Chạy refine decoder
        refined_hs = transformer.refine_decoder(topk_proposals.unsqueeze(0).unsqueeze(0), hs[-1][:, :3, :])
        
        # Lấy output của lớp refine cuối cùng và đưa qua FFN
        kpt_pred_refine = head.refine_reg_branches[-1](refined_hs[-1]).squeeze(0) # (num_proposals, num_kpts * 3)


    # --- Trực quan hóa ---
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Trực quan hóa 3 kết quả đầu tiên từ mỗi khối
    visualize_poses_3d(kpt_pred_rpn.cpu().numpy()[:3], ax1, "Output from Encoder (RPN)")
    visualize_poses_3d(kpt_pred_decoder.cpu().numpy()[:3], ax2, "Output from Pose Decoder")
    visualize_poses_3d(kpt_pred_refine.cpu().numpy(), ax3, "Output from Refine Decoder")

    plt.tight_layout()
    plt.show()

def main():
    print("--- Bắt đầu kịch bản trực quan hóa ---")
    
    # Tải cấu hình
    print(f"1. Đang tải cấu hình từ: {CONFIG_FILE}")
    cfg = Config.fromfile(CONFIG_FILE)
    
    # Sửa đổi cấu hình để tương thích với inference trên 1 GPU
    cfg.model.train_cfg = None
    
    # Xây dựng mô hình
    print("2. Đang xây dựng mô hình PETR...")
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # Tải checkpoint
    print(f"3. Đang tải checkpoint từ: {CHECKPOINT_FILE}")
    try:
        checkpoint = load_checkpoint(model, CHECKPOINT_FILE, map_location='cpu')
        print("   Tải checkpoint thành công!")
    except FileNotFoundError:
        print(f"   LỖI: Không tìm thấy tệp checkpoint tại '{CHECKPOINT_FILE}'.")
        print("   Vui lòng cập nhật biến CHECKPOINT_FILE trong script.")
        return
        
    # Chuẩn bị dữ liệu đầu vào giả lập (dummy input)
    print("4. Đang tạo dữ liệu CSI đầu vào giả lập...")
    # Dựa trên cấu hình, backbone là ResNet50, nó cần đầu vào dạng ảnh.
    # WifiPoseDataset sẽ biến đổi dữ liệu CSI (vd: 1x3x3x60x20) thành một
    # tensor giống ảnh, ví dụ (1, C, H, W).
    # Ở đây ta giả lập tensor 'img' này.
    
    # Giả sử dữ liệu CSI được reshape thành một ảnh 3 kênh
    dummy_img = torch.randn(1, 3, 256, 256)
    
    # Pipeline dữ liệu yêu cầu một dict, ta tạo một dict giả lập
    # với các thông tin meta cần thiết
    dummy_input = {
        'img': [dummy_img],
        'img_metas': [[{
            'img_shape': (256, 256, 3),
            'scale_factor': np.array([1., 1., 1., 1.]),
            # Thêm các keys khác nếu mô hình của bạn cần
        }]]
    }
    
    print("5. Thực hiện forward pass và trực quan hóa kết quả...")
    forward_and_visualize(model, dummy_input)
    
    print("--- Kịch bản hoàn tất ---")


if __name__ == '__main__':
    main()