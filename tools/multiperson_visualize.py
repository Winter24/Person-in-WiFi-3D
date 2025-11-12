import os
import torch
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from mmcv import Config
from scipy.optimize import linear_sum_assignment

# Import các thành phần cần thiết từ codebase 'opera'
from opera.apis import init_detector
from opera.datasets import build_dataset

# --- CẤU HÌNH ---
CONFIG_FILE = '/home/winter24/Person-in-WiFi-3D-repo/configs/wifi/petr_wifi.py'
CHECKPOINT_FILE = '/home/winter24/Person-in-WiFi-3D-repo/data/wifipose/result/bone_length_loss.pth'

# --- CẤU HÌNH VISUALIZE VÀ PHÂN TÍCH ---
MIN_NUM_PEOPLE = 2
SCORE_THRESHOLD = 0.3
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Tên các khớp theo thứ tự 14 điểm (dựa trên CrowdPose/WifiPose)
JOINT_NAMES = [
    'Vai Trái', 'Vai Phải', 'Khuỷu Trái', 'Khuỷu Phải', 'Cổ Tay Trái', 'Cổ Tay Phải',
    'Hông Trái', 'Hông Phải', 'Đầu Gối Trái', 'Đầu Gối Phải', 'Mắt Cá Trái', 'Mắt Cá Phải',
    'Đỉnh Đầu', 'Cổ'
]
# Khớp dùng để neo nhãn (13 là 'Cổ')
LABEL_ANCHOR_JOINT_IDX = 13

def calculate_detailed_metrics(pred_poses, gt_poses):
    """Tính toán chi tiết các lỗi MPJPE và PJDLE sau khi khớp cặp."""
    num_pred, num_gt = pred_poses.shape[0], gt_poses.shape[0]
    summary = {"matches": 0, "false_positives": num_pred, "false_negatives": num_gt}
    details = []

    if num_pred == 0 or num_gt == 0:
        return {"summary": summary, "details": details}

    cost_matrix = np.zeros((num_gt, num_pred))
    for i in range(num_gt):
        for j in range(num_pred):
            cost_matrix[i, j] = np.mean(np.linalg.norm(gt_poses[i] - pred_poses[j], axis=1))

    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
    
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        gt_person, pred_person = gt_poses[gt_idx], pred_poses[pred_idx]
        per_joint_euclidean = np.linalg.norm(gt_person - pred_person, axis=1) * 1000
        per_joint_dimensional = np.abs(gt_person - pred_person) * 1000
        details.append({
            "gt_index": gt_idx, "pred_index": pred_idx,
            "overall_mpjpe": np.mean(per_joint_euclidean),
            "per_joint_errors": per_joint_euclidean,
            "per_joint_dim_errors": per_joint_dimensional
        })

    summary.update({
        "matches": len(gt_indices), "false_positives": num_pred - len(pred_indices),
        "false_negatives": num_gt - len(gt_indices)
    })
    return {"summary": summary, "details": details}


def print_detailed_metrics(metrics, gt_labels, pred_labels):
    """In ra bảng thống kê lỗi chi tiết, dễ hiểu."""
    summary, details = metrics["summary"], metrics["details"]
    print("\n" + "="*60); print(" " * 17 + "BÁO CÁO PHÂN TÍCH ĐỘ LỆCH"); print("="*60)
    
    print("\n[ TÓM TẮT KHỚP CẶP ]")
    print(f" - Số cặp (Dự đoán - Ground-Truth) được khớp: {summary['matches']}")
    print(f" - Số dự đoán thừa (False Positives): {summary['false_positives']}")
    print(f" - Số tư thế bị bỏ lỡ (False Negatives): {summary['false_negatives']}")
    
    if not details: print("\nKhông có cặp nào được khớp để phân tích chi tiết."); print("="*60); return

    all_joint_errors = np.array([res["per_joint_errors"] for res in details])
    avg_mpjpe_overall = np.mean(all_joint_errors)
    all_dim_errors = np.array([res["per_joint_dim_errors"] for res in details])
    avg_pjdle_overall = np.mean(all_dim_errors, axis=(0, 1))
    
    print("\n[ PHÂN TÍCH TỔNG THỂ (trên các cặp đã khớp) ]")
    print(f" - MPJPE Tổng thể (Lỗi vị trí trung bình): {avg_mpjpe_overall:.2f} mm")
    print(" - PJDLE Tổng thể (Lỗi trung bình theo từng trục):")
    print(f"   - Trục X (ngang): {avg_pjdle_overall[0]:.2f} mm")
    print(f"   - Trục Y (dọc):   {avg_pjdle_overall[1]:.2f} mm")
    print(f"   - Trục Z (sâu):    {avg_pjdle_overall[2]:.2f} mm")

    for result in details:
        gt_label, pred_label = gt_labels[result['gt_index']], pred_labels[result['pred_index']]
        print(f"\n" + "-"*60)
        print(f"--- Phân tích Cặp Khớp: [{gt_label}] (Xanh) <-> [{pred_label}] (Đỏ) ---")
        print(f"  - MPJPE của cặp này: {result['overall_mpjpe']:.2f} mm")
        print("  - Bảng lỗi chi tiết từng khớp (Sắp xếp theo lỗi giảm dần):")
        print("    " + "="*45)
        print(f"    {'Tên Khớp':<20} | {'Lỗi (mm)':>20}")
        print("    " + "="*45)
        sorted_errors = sorted(zip(JOINT_NAMES, result['per_joint_errors']), key=lambda item: item[1], reverse=True)
        for joint_name, error in sorted_errors:
            print(f"    {joint_name:<20} | {error:>20.2f}")
        print("    " + "="*45)
    print("\n" + "="*60)


def visualize_comparison_3d(pred_keypoints, gt_keypoints, pred_labels, gt_labels, title="So sánh Tư thế 3D"):
    """Vẽ và gắn nhãn cả hai bộ xương trên cùng một biểu đồ."""
    limbs = [[0,1], [1,2], [2,5], [3,0], [4,2], [5,7], [6,3], [7,3], [8,4], [9,5], [10,6], [11,7], [12,9], [13,11]]
    GT_COLOR, PRED_COLOR = 'blue', 'red'

    fig = plt.figure(figsize=(12, 12)); ax = fig.add_subplot(111, projection='3d'); ax.set_title(title, fontsize=16)
    all_points = []

    def plot_poses(keypoints_set, color, marker, labels):
        if keypoints_set is not None and len(keypoints_set) > 0:
            for i, person_kpts in enumerate(keypoints_set):
                all_points.extend(person_kpts)
                x, y, z = person_kpts[:, 0], person_kpts[:, 1], person_kpts[:, 2]
                ax.scatter(x, y, z, c=color, marker=marker, s=50) # Tăng kích thước điểm
                for limb in limbs:
                    start, end = person_kpts[limb[0]], person_kpts[limb[1]]
                    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color, linewidth=2.5) # Dày hơn
                # Gắn nhãn cho người
                anchor_joint = person_kpts[LABEL_ANCHOR_JOINT_IDX]
                ax.text(anchor_joint[0], anchor_joint[1], anchor_joint[2] + 0.1, labels[i], color=color, fontsize=12, fontweight='bold')

    plot_poses(gt_keypoints, GT_COLOR, 'o', gt_labels)
    plot_poses(pred_keypoints, PRED_COLOR, '^', pred_labels)

    if not all_points: ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    else:
        all_points = np.array(all_points); min_vals, max_vals = np.min(all_points, axis=0), np.max(all_points, axis=0)
        mid_vals = (min_vals + max_vals) / 2; max_range = (max_vals - min_vals).max() * 0.7
        ax.set_xlim(mid_vals[0] - max_range, mid_vals[0] + max_range)
        ax.set_ylim(mid_vals[1] - max_range, mid_vals[1] + max_range)
        z_min, z_max = mid_vals[2] - max_range, mid_vals[2] + max_range; ax.set_zlim(z_max, z_min)
        
    legend_elements = [Line2D([0], [0], color=GT_COLOR, marker='o', linestyle='-', label='Ground-Truth'),
                       Line2D([0], [0], color=PRED_COLOR, marker='^', linestyle='-', label='Dự đoán')]
    ax.legend(handles=legend_elements, fontsize=12)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.view_init(elev=20., azim=-75); plt.show()


def main():
    cfg = Config.fromfile(CONFIG_FILE)
    
    print(f"Đang tìm kiếm mẫu dữ liệu có >= {MIN_NUM_PEOPLE} người...")
    test_data_root = cfg.data.test.dataset_root; keypoint_folder = os.path.join(test_data_root, 'keypoint')
    data_list_file = os.path.join(test_data_root, 'test_data_list.txt')
    with open(data_list_file, 'r') as f: test_filenames = [line.strip() for line in f.readlines()]
    multi_person_indices = []
    for idx, filename in enumerate(test_filenames):
        keypoint_path = os.path.join(keypoint_folder, f"{filename}.npy")
        if os.path.exists(keypoint_path):
            try:
                if np.load(keypoint_path).shape[0] >= MIN_NUM_PEOPLE: multi_person_indices.append(idx)
            except Exception: continue
    if not multi_person_indices: print(f"!!! Lỗi: Không tìm thấy mẫu nào có >= {MIN_NUM_PEOPLE} người."); return

    SAMPLE_INDEX = 1858
    
    print(f"Đã tìm thấy {len(multi_person_indices)} mẫu. Sẽ visualize mẫu tại chỉ số: {SAMPLE_INDEX}")

    model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
    vis_pipeline = [
        dict(type='opera.DefaultFormatBundle', extra_keys=['gt_keypoints']),
        dict(type='mmdet.Collect', keys=['img', 'gt_keypoints'], meta_keys=['img_name'])
    ]
    cfg.data.test.pipeline = vis_pipeline; dataset = build_dataset(cfg.data.test)
    data = dataset[SAMPLE_INDEX]
    
    img_tensor = data['img'].data.to(DEVICE).unsqueeze(0)
    img_metas = [{'img_name': data['img_metas'].data['img_name']}]

    print("\nĐang thực hiện suy luận..."); model.eval()
    with torch.no_grad():
        result = model.simple_test(img_tensor, img_metas, rescale=False)
    print("Suy luận hoàn tất.")

    bbox_kpt_results = result[0]; pred_bboxes_all = bbox_kpt_results[0][0]; pred_keypoints_all = bbox_kpt_results[1][0]
    scores = pred_bboxes_all[:, -1]; keep_mask = scores > SCORE_THRESHOLD
    final_pred_keypoints_3d = pred_keypoints_all[keep_mask]
    gt_keypoints_3d = data['gt_keypoints'].data.numpy()
    
    # Tạo nhãn để tham chiếu
    gt_labels = [f"GT {i}" for i in range(gt_keypoints_3d.shape[0])]
    pred_labels = [f"Pred {i}" for i in range(final_pred_keypoints_3d.shape[0])]
    
    # Tính toán và in ra các thông số lỗi chi tiết
    metrics = calculate_detailed_metrics(final_pred_keypoints_3d, gt_keypoints_3d)
    print_detailed_metrics(metrics, gt_labels, pred_labels)

    # Visualize
    visualize_comparison_3d(
        final_pred_keypoints_3d, gt_keypoints_3d, 
        pred_labels, gt_labels,
        title=f"So sánh Tư thế 3D cho {gt_keypoints_3d.shape[0]} người (Mẫu {SAMPLE_INDEX})"
    )

if __name__ == '__main__':
    main()