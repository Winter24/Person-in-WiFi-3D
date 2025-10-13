# tools/analysis/visualize_wifi_keypoints.py (PHIÊN BẢN NÂNG CẤP)

import sys
import os
# Thêm thư mục gốc của dự án vào Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from opera.datasets.wifi_pose import WifiPoseDataset

# --- CẤU HÌNH ---
# !!! QUAN TRỌNG: Thay đổi đường dẫn này cho đúng với máy của bạn !!!
DATASET_ROOT = '/home/yankangwei/opera-main/data/wifipose/train_data'
SAMPLE_INDEX_TO_VISUALIZE = 0  # Chọn một mẫu bất kỳ để xem


LIMBS_HYPOTHESIS = [
    (0, 1),  # Đầu -> Cổ
    (0, 2),   # Cổ -> Vai Trái
    (0, 3),   # Cổ -> Vai Phải
    (2, 3),    # Vai Trái -> Vai Phải
    (2, 4),    # Vai Trái -> Khuỷu Trái
    (4, 8),    # Khuỷu Trái -> Cổ Tay Trái
    (3, 6),    # Vai Phải -> Khuỷu Phải
    (6, 10),    # Khuỷu Phải -> Cổ Tay Phải
    (0, 5),   # Cổ -> Hông Trái
    (0, 7),   # Cổ -> Hông Phải
    (5, 7),    # Hông Trái -> Hông Phải
    (5, 9),    # Hông Trái -> Đầu Gối Trái
    (9, 12),   # Đầu Gối Trái -> Mắt Cá Trái
    (7, 11),    # Hông Phải -> Đầu Gối Phải
    (11, 13)    # Đầu Gối Phải -> Mắt Cá Phải
]

def visualize_single_person_skeleton(keypoints_3d, ax, title=""):
    """Vẽ bộ xương 3D và ghi chú chỉ số của từng khớp."""
    
    ax.set_title(title, fontsize=14)
    
    # Vẽ các khớp (joints)
    xs, ys, zs = keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2]
    ax.scatter(xs, ys, zs, c='red', marker='o', s=50, depthshade=True)
    
    # Ghi chú chỉ số của từng khớp
    for i in range(keypoints_3d.shape[0]):
        ax.text(xs[i], ys[i], zs[i], f'{i}', color='blue', fontsize=12, fontweight='bold')
        
    # Vẽ các xương (limbs) dựa trên giả định ban đầu
    for start_idx, end_idx in LIMBS_HYPOTHESIS:
        # Lấy tọa độ của 2 điểm
        p1 = keypoints_3d[start_idx]
        p2 = keypoints_3d[end_idx]
        # Vẽ đường nối
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='green', linewidth=2)
        
def set_axes_equal(ax):
    """Làm cho các trục 3D có cùng tỷ lệ để hình người không bị méo."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def main():
    print(f"Bắt đầu trực quan hóa dữ liệu ground-truth từ mẫu số {SAMPLE_INDEX_TO_VISUALIZE}...")
    
    # --- TẢI DỮ LIỆU ---
    try:
        dataset = WifiPoseDataset(dataset_root=DATASET_ROOT, pipeline=[], mode='train')
        if SAMPLE_INDEX_TO_VISUALIZE >= len(dataset):
            print(f"Lỗi: Chỉ số mẫu {SAMPLE_INDEX_TO_VISUALIZE} vượt quá kích thước dataset ({len(dataset)}).")
            return
        data_sample = dataset[SAMPLE_INDEX_TO_VISUALIZE]
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        print(f"Vui lòng kiểm tra lại đường dẫn: '{DATASET_ROOT}'")
        return
        
    gt_keypoints_all = data_sample['gt_keypoints']
    num_people = gt_keypoints_all.shape[0]
    
    if num_people == 0:
        print(f"Mẫu dữ liệu {SAMPLE_INDEX_TO_VISUALIZE} không có người nào.")
        return
        
    print(f"Tìm thấy {num_people} người trong mẫu. Sẽ trực quan hóa từng người.")
    
    # --- TRỰC QUAN HÓA ---
    # Tạo một figure với nhiều subplot nếu có nhiều người
    fig = plt.figure(figsize=(8 * num_people, 8))
    
    for i in range(num_people):
        person_kpts = gt_keypoints_all[i].numpy()
        
        ax = fig.add_subplot(1, num_people, i + 1, projection='3d')
        visualize_single_person_skeleton(person_kpts, ax, title=f"Người số {i+1}")
        
        ax.set_xlabel('X (ngang)')
        ax.set_ylabel('Y (sâu)')
        ax.set_zlabel('Z (cao)')
        
        # Đặt các trục có cùng tỷ lệ
        set_axes_equal(ax)
        
        # Đảo trục Z để đầu hướng lên trên (matplotlib mặc định ngược)
        ax.invert_zaxis()
        
        # Thiết lập góc nhìn
        ax.view_init(elev=15, azim=-75)

    fig.suptitle(f'Trực quan hóa Ground-Truth Keypoints cho Mẫu {SAMPLE_INDEX_TO_VISUALIZE}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    main()