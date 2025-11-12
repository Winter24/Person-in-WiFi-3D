import os
import torch
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from mmcv import Config

# Import các thành phần cần thiết từ codebase 'opera'
from opera.apis import init_detector
from opera.datasets import build_dataset

# --- Cấu hình ---
# !!! HÃY ĐẢM BẢO CÁC ĐƯỜNG DẪN NÀY CHÍNH XÁC TRÊN MÁY CỦA BẠN !!!
CONFIG_FILE = '/home/winter24/Person-in-WiFi-3D-repo/configs/wifi/petr_wifi.py'
CHECKPOINT_FILE = '/home/winter24/Person-in-WiFi-3D-repo/data/wifipose/result/bone_length_loss.pth'

# Ngưỡng điểm tin cậy để lọc kết quả
SCORE_THRESHOLD = 0.3

# Chỉ số của mẫu dữ liệu trong tập test mà bạn muốn visualize
SAMPLE_INDEX = 25

# Chọn thiết bị (GPU hoặc CPU)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def visualize_comparison_3d(pred_keypoints, gt_keypoints, title="So sánh Tư thế 3D"):
    """
    Vẽ cả tư thế dự đoán và ground-truth trên cùng một biểu đồ 3D.
    """
    
    limbs = [
        [0, 1], [1, 2], [2, 5], [3, 0], [4, 2], [5, 7],
        [6, 3], [7, 3], [8, 4], [9, 5], [10, 6], [11, 7],
        [12, 9], [13, 11]
    ]
    
    GT_COLOR = 'blue'
    PRED_COLOR = 'red'

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    all_points = []

    # Hàm phụ để vẽ một bộ xương
    def plot_poses(keypoints_set, color, marker, label):
        if keypoints_set is not None and len(keypoints_set) > 0:
            for person_kpts in keypoints_set:
                all_points.extend(person_kpts)
                x, y, z = person_kpts[:, 0], person_kpts[:, 1], person_kpts[:, 2]
                ax.scatter(x, y, z, c=color, marker=marker, label=f'{label} Joints')
                for limb in limbs:
                    start, end = person_kpts[limb[0]], person_kpts[limb[1]]
                    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color)

    # Vẽ ground-truth
    plot_poses(gt_keypoints, GT_COLOR, 'o', 'Ground-Truth')
    # Vẽ dự đoán
    plot_poses(pred_keypoints, PRED_COLOR, '^', 'Prediction')

    # Thiết lập giới hạn trục dựa trên tất cả các điểm
    if not all_points:
        print("Không có điểm nào để vẽ.")
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    else:
        all_points = np.array(all_points)
        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)
        
        mid_vals = (min_vals + max_vals) / 2
        max_range = (max_vals - min_vals).max() / 2.0
        
        ax.set_xlim(mid_vals[0] - max_range, mid_vals[0] + max_range)
        ax.set_ylim(mid_vals[1] - max_range, mid_vals[1] + max_range)
        
        # --- LẬT TRỤC Z ---
        # Đặt giới hạn Z với max ở dưới và min ở trên để lật trục
        z_min, z_max = mid_vals[2] - max_range, mid_vals[2] + max_range
        ax.set_zlim(z_max, z_min)

    # Thêm chú thích
    legend_elements = [
        Line2D([0], [0], color=GT_COLOR, lw=4, label='Ground-Truth'),
        Line2D([0], [0], color=PRED_COLOR, lw=4, label='Dự đoán')
    ]
    ax.legend(handles=legend_elements)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=20., azim=-75) # Điều chỉnh góc nhìn cho đẹp hơn
    plt.show()


def main():
    print(f"Đang tải mô hình từ:\n- Config: {CONFIG_FILE}\n- Checkpoint: {CHECKPOINT_FILE}")
    model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
    print("Tải mô hình thành công.")

    print(f"Đang tải mẫu dữ liệu thứ {SAMPLE_INDEX} từ tập test...")
    cfg = Config.fromfile(CONFIG_FILE)
    
    vis_pipeline = [
        dict(type='opera.DefaultFormatBundle', extra_keys=['gt_keypoints', 'gt_labels']),
        dict(type='mmdet.Collect', keys=['img', 'gt_keypoints'], meta_keys=['img_name'])
    ]
    cfg.data.test.pipeline = vis_pipeline
    
    dataset = build_dataset(cfg.data.test)
    data = dataset[SAMPLE_INDEX]
    
    img_tensor = data['img'].data.to(DEVICE).unsqueeze(0)
    img_metas = [{'img_name': data['img_metas'].data['img_name']}]

    print("Đang thực hiện suy luận...")
    model.eval()
    with torch.no_grad():
        result = model.simple_test(img_tensor, img_metas, rescale=False)

    print("Suy luận hoàn tất.")

    bbox_kpt_results = result[0]
    pred_bboxes_all = bbox_kpt_results[0][0]
    pred_keypoints_all = bbox_kpt_results[1][0]
    
    scores = pred_bboxes_all[:, -1]
    keep_mask = scores > SCORE_THRESHOLD
    final_pred_keypoints_3d = pred_keypoints_all[keep_mask]
    
    gt_keypoints_3d = data['gt_keypoints'].data.numpy()
    
    print("-" * 30)
    print(f"Tổng số ứng viên dự đoán: {pred_keypoints_all.shape[0]}.")
    print(f"Số người được giữ lại (điểm > {SCORE_THRESHOLD}): {final_pred_keypoints_3d.shape[0]}.")
    print(f"Số người trong ground-truth: {gt_keypoints_3d.shape[0]}.")
    print("-" * 30)

    # Visualize cả hai trên cùng một biểu đồ
    visualize_comparison_3d(
        final_pred_keypoints_3d, 
        gt_keypoints_3d, 
        title=f"So sánh Tư thế 3D (Mẫu {SAMPLE_INDEX})"
    )

if __name__ == '__main__':
    main()