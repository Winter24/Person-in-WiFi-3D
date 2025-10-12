# tools/analysis/compute_bone_stats.py

import sys
import os
# Thêm thư mục gốc của dự án vào Python path để có thể import các module của opera
# Điều này rất quan trọng để chạy script từ thư mục gốc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import json
from tqdm import tqdm

# Import trực tiếp lớp Dataset từ module của bạn
from opera.datasets.wifi_pose import WifiPoseDataset

def compute_and_save_bone_stats():
    """
    Hàm chính để duyệt qua bộ dữ liệu, tính toán chiều dài xương
    và lưu lại kết quả thống kê.
    """
    print("Bắt đầu tính toán thống kê chiều dài xương từ bộ dữ liệu huấn luyện...")

    # --- ĐỊNH NGHĨA CẤU TRÚC XƯƠNG ---
    # Dựa trên 14 keypoints của bộ dữ liệu WifiPose:
    # 0:L_Shoulder, 1:R_Shoulder, 2:L_Elbow, 3:R_Elbow, 4:L_Wrist, 5:R_Wrist, 
    # 6:L_Hip, 7:R_Hip, 8:L_Knee, 9:R_Knee, 10:L_Ankle, 11:R_Ankle, 12:Head, 13:Neck
    
    # Danh sách các cặp chỉ số keypoint tạo thành "xương"
    BONES = [
        # Cột sống & Vai
        (13, 12), # Neck -> Head
        (0, 13),  # L_Shoulder -> Neck
        (1, 13),  # R_Shoulder -> Neck
        (0, 1),   # L_Shoulder -> R_Shoulder
        (6, 13),  # L_Hip -> Neck (ước lượng cho cột sống)
        (7, 13),  # R_Hip -> Neck (ước lượng cho cột sống)
        (6, 7),   # L_Hip -> R_Hip
        # Tay trái
        (0, 2),   # L_Shoulder -> L_Elbow
        (2, 4),   # L_Elbow -> L_Wrist
        # Tay phải
        (1, 3),   # R_Shoulder -> R_Elbow
        (3, 5),   # R_Elbow -> R_Wrist
        # Chân trái
        (6, 8),   # L_Hip -> L_Knee
        (8, 10),  # L_Knee -> L_Ankle
        # Chân phải
        (7, 9),   # R_Hip -> R_Knee
        (9, 11)   # R_Knee -> R_Ankle
    ]
    
    # (Tùy chọn) Tên tương ứng để dễ debug
    BONE_NAMES = [
        "Neck_Head", "L_Collar", "R_Collar", "Shoulder_Line", 
        "L_Spine", "R_Spine", "Hip_Line",
        "L_UpperArm", "L_LowerArm", "R_UpperArm", "R_LowerArm",
        "L_Thigh", "L_Calf", "R_Thigh", "R_Calf"
    ]
    
    # --- TẢI DỮ LIỆU ---
    # !!! QUAN TRỌNG: Thay đổi đường dẫn này cho đúng với máy của bạn !!!
    dataset_root = '/home/yankangwei/opera-main/data/wifipose/train_data' 
    
    # Pipeline rỗng vì chúng ta chỉ cần đọc dữ liệu thô (keypoints)
    pipeline = []
    
    try:
        dataset = WifiPoseDataset(dataset_root=dataset_root, pipeline=pipeline, mode='train')
    except Exception as e:
        print(f"Lỗi khi khởi tạo Dataset: {e}")
        print(f"Hãy chắc chắn đường dẫn '{dataset_root}' là chính xác và chứa file 'train_data_list.txt'.")
        return

    # Sử dụng DataLoader của PyTorch để duyệt qua dữ liệu dễ dàng hơn
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # --- VÒNG LẶP TÍNH TOÁN ---
    # Khởi tạo một list các list để lưu chiều dài của từng loại xương
    all_bone_lengths = [[] for _ in BONES]
    
    print(f"Duyệt qua {len(dataset)} mẫu dữ liệu...")
    for data_batch in tqdm(dataloader, desc="Đang xử lý dữ liệu"):
        # Dataloader trả về batch, ta lấy phần tử đầu tiên
        gt_keypoints = data_batch['gt_keypoints'][0] # Shape: (num_persons, 14, 3)
        
        # Duyệt qua từng người trong frame
        for person_kpts in gt_keypoints:
            for i, (p1_idx, p2_idx) in enumerate(BONES):
                p1 = person_kpts[p1_idx]
                p2 = person_kpts[p2_idx]
                
                # Chỉ tính toán nếu cả 2 khớp đều hợp lệ (cột cuối > 0)
                if p1[2] > 0 and p2[2] > 0:
                    # Tính khoảng cách Euclid (L2 norm) cho tọa độ 3D
                    length = torch.norm(p1[:3] - p2[:3])
                    all_bone_lengths[i].append(length.item())
                    
    # --- TÍNH TOÁN THỐNG KÊ VÀ LƯU KẾT QUẢ ---
    bone_stats = {
        "mean": [],
        "std": [],
        "bone_names": BONE_NAMES # Lưu lại tên để dễ kiểm tra
    }
    
    print("\n--- Kết quả thống kê chiều dài xương (đơn vị mét) ---")
    for i, lengths in enumerate(all_bone_lengths):
        if len(lengths) > 0:
            mean_val = np.mean(lengths)
            std_val = np.std(lengths)
            bone_stats["mean"].append(mean_val)
            bone_stats["std"].append(std_val)
            print(f"{BONE_NAMES[i]:<15}: Mean = {mean_val:.4f} m, Std = {std_val:.4f} m, ({len(lengths)} mẫu)")
        else:
            # Trường hợp một loại xương không bao giờ hợp lệ trong dataset
            bone_stats["mean"].append(0.0)
            bone_stats["std"].append(0.0)
            print(f"{BONE_NAMES[i]:<15}: Không có mẫu hợp lệ.")

    # Lưu ra file JSON ở thư mục gốc của dự án
    output_path = 'gt_bone_stats.json'
    with open(output_path, 'w') as f:
        json.dump(bone_stats, f, indent=4)
        
    print(f"\nĐã tính toán và lưu xong kết quả vào file: '{output_path}'")

if __name__ == '__main__':
    compute_and_save_bone_stats()