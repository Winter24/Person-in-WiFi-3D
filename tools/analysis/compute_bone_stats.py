# tools/analysis/compute_bone_stats.py (PHIÊN BẢN CUỐI CÙNG)

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import json
from tqdm import tqdm
from opera.datasets.wifi_pose import WifiPoseDataset

def compute_and_save_bone_stats():
    print("Bắt đầu tính toán thống kê chiều dài xương từ bộ dữ liệu huấn luyện...")

    # =================================================================
    # == ĐỊNH NGHĨA XƯƠNG (PHIÊN BẢN ĐÃ XÁC MINH) ==
    # =================================================================
    # Ánh xạ đã xác minh:
    # 0: Head, 1: Neck, 2: R_Shoulder, 3: L_Shoulder, 4: R_Elbow, 5: L_Elbow, 
    # 6: R_Hip, 7: L_Wrist, 8: R_Wrist, 9: R_Knee, 10: R_Ankle, 
    # 11: L_Knee, 12: L_Hip, 13: L_Ankle

    BONES_FULL_BODY = [
        (0, 1), (1, 3), (1, 2), (3, 2), (3, 12), (2, 6), (12, 6),
        (3, 5), (5, 7), (2, 4), (4, 8), (12, 11), (11, 13), (6, 9), (9, 10)
    ]
    
    BONE_NAMES = [
        "Head_Neck", "Neck_LShoulder", "Neck_RShoulder", "LShoulder_RShoulder",
        "LShoulder_LHip", "RShoulder_RHip", "LHip_RHip",
        "LShoulder_LElbow", "LElbow_LWrist", "RShoulder_RElbow", "RElbow_RWrist",
        "LHip_LKnee", "LKnee_LAnkle", "RHip_RKnee", "RKnee_RAnkle"
    ]
    
    # --- TẢI DỮ LIỆU ---
    # !!! QUAN TRỌNG: Thay đổi đường dẫn này nếu cần !!!
    dataset_root = '/home/yankangwei/opera-main/data/wifipose/train_data' 
    
    try:
        dataset = WifiPoseDataset(dataset_root=dataset_root, pipeline=[], mode='train')
    except Exception as e:
        print(f"Lỗi khi khởi tạo Dataset: {e}")
        return

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # --- VÒNG LẶP TÍNH TOÁN ---
    all_bone_lengths = [[] for _ in BONES_FULL_BODY]
    
    print(f"Duyệt qua {len(dataset)} mẫu dữ liệu...")
    for data_batch in tqdm(dataloader, desc="Đang xử lý dữ liệu"):
        gt_keypoints = data_batch['gt_keypoints'][0]
        
        for person_kpts in gt_keypoints:
            for i, (p1_idx, p2_idx) in enumerate(BONES_FULL_BODY):
                p1 = person_kpts[p1_idx]
                p2 = person_kpts[p2_idx]
                
                if p1[2] > 0 and p2[2] > 0:
                    length = torch.norm(p1[:3] - p2[:3])
                    all_bone_lengths[i].append(length.item())
                    
    # --- TÍNH TOÁN VÀ LƯU KẾT QUẢ ---
    bone_stats = {
        "mean": [],
        "std": [],
        "bone_names": BONE_NAMES,
        "bones_definition": BONES_FULL_BODY # Lưu lại định nghĩa để tham chiếu
    }
    
    print("\n--- Kết quả thống kê chiều dài xương (đơn vị mét) ---")
    for i, lengths in enumerate(all_bone_lengths):
        if len(lengths) > 0:
            mean_val = np.mean(lengths)
            std_val = np.std(lengths)
            bone_stats["mean"].append(mean_val)
            bone_stats["std"].append(std_val)
            print(f"{BONE_NAMES[i]:<20}: Mean = {mean_val:.4f} m, Std = {std_val:.4f} m, ({len(lengths)} mẫu)")
        else:
            bone_stats["mean"].append(0.0)
            bone_stats["std"].append(0.0)
            print(f"{BONE_NAMES[i]:<20}: Không có mẫu hợp lệ.")

    output_path = 'gt_bone_stats.json'
    with open(output_path, 'w') as f:
        json.dump(bone_stats, f, indent=4)
        
    print(f"\nĐã tính toán và lưu xong kết quả vào file: '{output_path}'")

if __name__ == '__main__':
    compute_and_save_bone_stats()