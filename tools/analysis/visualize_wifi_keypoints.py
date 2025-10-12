# tools/analysis/visualize_wifi_keypoints.py

import sys
import os
# Thêm thư mục gốc của dự án vào Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from opera.datasets.wifi_pose import WifiPoseDataset

def visualize_single_person(keypoints_3d, ax):
    """Vẽ một bộ keypoints 3D lên một trục Axes3D."""
    
    # Tọa độ X, Y, Z
    xs = keypoints_3d[:, 0]
    ys = keypoints_3d[:, 1]
    zs = keypoints_3d[:, 2]
    
    # Vẽ các điểm keypoint
    ax.scatter(xs, ys, zs, c='r', marker='o')
    
    # Ghi chú chỉ số của từng điểm
    for i in range(keypoints_3d.shape[0]):
        ax.text(xs[i], ys[i], zs[i], f'{i}', color='blue', fontsize=12)

def main():
    # --- CẤU HÌNH ---
    # !!! QUAN TRỌNG: Thay đổi đường dẫn này cho đúng với máy của bạn !!!
    dataset_root = '/home/yankangwei/opera-main/data/wifipose/train_data'
    sample_index_to_visualize = 0 # Chọn một mẫu bất kỳ để xem
    
    # --- TẢI DỮ LIỆU ---
    try:
        # Pipeline rỗng để lấy dữ liệu thô
        dataset = WifiPoseDataset(dataset_root=dataset_root, pipeline=[], mode='train')
        data_sample = dataset[sample_index_to_visualize]
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        return
        
    gt_keypoints = data_sample['gt_keypoints'] # Shape: (num_persons, 14, 3)
    
    if gt_keypoints.shape[0] == 0:
        print(f"Mẫu dữ liệu {sample_index_to_visualize} không có người nào.")
        return
        
    # Lấy keypoints của người đầu tiên để trực quan hóa
    first_person_kpts = gt_keypoints[0]
    
    print(f"--- Tọa độ 3D của người đầu tiên trong mẫu {sample_index_to_visualize} ---")
    print(first_person_kpts)
    
    # --- TRỰC QUAN HÓA ---
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    visualize_single_person(first_person_kpts.numpy(), ax)
    
    ax.set_xlabel('X (Chiều ngang)')
    ax.set_ylabel('Y (Chiều sâu)')
    ax.set_zlabel('Z (Chiều cao)')
    ax.set_title(f'Trực quan hóa Keypoints 3D cho Mẫu {sample_index_to_visualize}')
    
    # Đảo ngược trục Y để góc nhìn trực quan hơn (tùy chọn)
    ax.invert_yaxis()
    
    plt.show()

if __name__ == '__main__':
    main()