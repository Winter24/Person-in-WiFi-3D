import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D
from mmcv import Config
from scipy.optimize import linear_sum_assignment

from opera.apis import init_detector
from opera.datasets import build_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_FILE = PROJECT_ROOT / 'configs' / 'wifi' / 'petr_wifi.py'
DEFAULT_CHECKPOINT_FILE = PROJECT_ROOT / 'work_dirs' / 'petr_wifi' / 'epoch_47.pth'

MIN_NUM_PEOPLE = 2
SCORE_THRESHOLD = 0.2
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

JOINT_NAMES = [
    'Vai Trai', 'Vai Phai', 'Khuyu Trai', 'Khuyu Phai', 'Co Tay Trai', 'Co Tay Phai',
    'Hong Trai', 'Hong Phai', 'Dau Goi Trai', 'Dau Goi Phai', 'Mat Ca Trai', 'Mat Ca Phai',
    'Dinh Dau', 'Co'
]
LABEL_ANCHOR_JOINT_IDX = 13


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize multi-person WiFi pose samples.')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_FILE))
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT_FILE))
    return parser.parse_args()


def calculate_detailed_metrics(pred_poses, gt_poses):
    """Compute matching and pose error summaries."""
    num_pred, num_gt = pred_poses.shape[0], gt_poses.shape[0]
    summary = {'matches': 0, 'false_positives': num_pred, 'false_negatives': num_gt}
    details = []

    if num_pred == 0 or num_gt == 0:
        return {'summary': summary, 'details': details}

    cost_matrix = np.zeros((num_gt, num_pred))
    for i in range(num_gt):
        for j in range(num_pred):
            cost_matrix[i, j] = np.mean(np.linalg.norm(gt_poses[i] - pred_poses[j], axis=1))

    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        gt_person = gt_poses[gt_idx]
        pred_person = pred_poses[pred_idx]
        per_joint_euclidean = np.linalg.norm(gt_person - pred_person, axis=1) * 1000
        per_joint_dimensional = np.abs(gt_person - pred_person) * 1000
        details.append({
            'gt_index': gt_idx,
            'pred_index': pred_idx,
            'overall_mpjpe': np.mean(per_joint_euclidean),
            'per_joint_errors': per_joint_euclidean,
            'per_joint_dim_errors': per_joint_dimensional
        })

    summary.update({
        'matches': len(gt_indices),
        'false_positives': num_pred - len(pred_indices),
        'false_negatives': num_gt - len(gt_indices)
    })
    return {'summary': summary, 'details': details}


def print_detailed_metrics(metrics, gt_labels, pred_labels):
    """Print a compact metric report for matched poses."""
    summary, details = metrics['summary'], metrics['details']
    print('\n' + '=' * 60)
    print(' ' * 17 + 'BAO CAO PHAN TICH DO LECH')
    print('=' * 60)

    print('\n[ TOM TAT KHOP CAP ]')
    print(f" - So cap (Du doan - Ground-Truth) duoc khop: {summary['matches']}")
    print(f" - So du doan thua (False Positives): {summary['false_positives']}")
    print(f" - So tu the bi bo lo (False Negatives): {summary['false_negatives']}")

    if not details:
        print('\nKhong co cap nao duoc khop de phan tich chi tiet.')
        print('=' * 60)
        return

    all_joint_errors = np.array([res['per_joint_errors'] for res in details])
    avg_mpjpe_overall = np.mean(all_joint_errors)
    all_dim_errors = np.array([res['per_joint_dim_errors'] for res in details])
    avg_pjdle_overall = np.mean(all_dim_errors, axis=(0, 1))

    print('\n[ PHAN TICH TONG THE (tren cac cap da khop) ]')
    print(f' - MPJPE Tong the: {avg_mpjpe_overall:.2f} mm')
    print(' - PJDLE Tong the:')
    print(f'   - Truc X: {avg_pjdle_overall[0]:.2f} mm')
    print(f'   - Truc Y: {avg_pjdle_overall[1]:.2f} mm')
    print(f'   - Truc Z: {avg_pjdle_overall[2]:.2f} mm')

    for result in details:
        gt_label = gt_labels[result['gt_index']]
        pred_label = pred_labels[result['pred_index']]
        print('\n' + '-' * 60)
        print(f'--- Phan tich cap khop: [{gt_label}] <-> [{pred_label}] ---')
        print(f"  - MPJPE cua cap nay: {result['overall_mpjpe']:.2f} mm")
        sorted_errors = sorted(
            zip(JOINT_NAMES, result['per_joint_errors']),
            key=lambda item: item[1],
            reverse=True)
        for joint_name, error in sorted_errors:
            print(f'    {joint_name:<20} | {error:>8.2f}')
    print('\n' + '=' * 60)


def visualize_comparison_3d(pred_keypoints, gt_keypoints, pred_labels, gt_labels,
                            title='So sanh Tu the 3D'):
    """Plot predicted and ground-truth multi-person poses."""
    limbs = [[0, 1], [1, 2], [2, 5], [3, 0], [4, 2], [5, 7], [6, 3], [7, 3],
             [8, 4], [9, 5], [10, 6], [11, 7], [12, 9], [13, 11]]
    gt_color = 'blue'
    pred_color = 'red'

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title, fontsize=16)
    all_points = []

    def plot_poses(keypoints_set, color, marker, labels):
        if keypoints_set is None or len(keypoints_set) == 0:
            return
        for i, person_kpts in enumerate(keypoints_set):
            all_points.extend(person_kpts)
            x, y, z = person_kpts[:, 0], person_kpts[:, 1], person_kpts[:, 2]
            ax.scatter(x, y, z, c=color, marker=marker, s=50)
            for limb in limbs:
                start, end = person_kpts[limb[0]], person_kpts[limb[1]]
                ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color=color,
                    linewidth=2.5)
            anchor_joint = person_kpts[LABEL_ANCHOR_JOINT_IDX]
            ax.text(
                anchor_joint[0],
                anchor_joint[1],
                anchor_joint[2] + 0.1,
                labels[i],
                color=color,
                fontsize=12,
                fontweight='bold')

    plot_poses(gt_keypoints, gt_color, 'o', gt_labels)
    plot_poses(pred_keypoints, pred_color, '^', pred_labels)

    if not all_points:
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
    else:
        all_points = np.array(all_points)
        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)
        mid_vals = (min_vals + max_vals) / 2
        max_range = (max_vals - min_vals).max() * 0.7
        ax.set_xlim(mid_vals[0] - max_range, mid_vals[0] + max_range)
        ax.set_ylim(mid_vals[1] - max_range, mid_vals[1] + max_range)
        z_min = mid_vals[2] - max_range
        z_max = mid_vals[2] + max_range
        ax.set_zlim(z_max, z_min)

    legend_elements = [
        Line2D([0], [0], color=gt_color, marker='o', linestyle='-', label='Ground-Truth'),
        Line2D([0], [0], color=pred_color, marker='^', linestyle='-', label='Du doan')
    ]
    ax.legend(handles=legend_elements, fontsize=12)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.view_init(elev=20., azim=-75)
    plt.show()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    print(f'Dang tim kiem mau du lieu co >= {MIN_NUM_PEOPLE} nguoi...')
    test_data_root = cfg.data.test.dataset_root
    keypoint_folder = Path(test_data_root) / 'keypoint'
    data_list_file = Path(test_data_root) / 'test_data_list.txt'
    test_filenames = [line.strip() for line in data_list_file.read_text(encoding='utf-8').splitlines()]

    multi_person_indices = []
    for idx, filename in enumerate(test_filenames):
        keypoint_path = keypoint_folder / f'{filename}.npy'
        if not keypoint_path.exists():
            continue
        try:
            if np.load(keypoint_path).shape[0] >= MIN_NUM_PEOPLE:
                multi_person_indices.append(idx)
        except Exception:
            continue

    if not multi_person_indices:
        print(f'Khong tim thay mau nao co >= {MIN_NUM_PEOPLE} nguoi.')
        return

    sample_index = multi_person_indices[0]
    print(f'Da tim thay {len(multi_person_indices)} mau. Se visualize mau tai chi so: {sample_index}')

    model = init_detector(args.config, args.checkpoint, device=DEVICE)
    vis_pipeline = [
        dict(type='opera.DefaultFormatBundle', extra_keys=['gt_keypoints']),
        dict(type='mmdet.Collect', keys=['img', 'gt_keypoints'], meta_keys=['img_name'])
    ]
    cfg.data.test.pipeline = vis_pipeline
    dataset = build_dataset(cfg.data.test)
    data = dataset[sample_index]

    img_tensor = data['img'].data.to(DEVICE).unsqueeze(0)
    img_metas = [{'img_name': data['img_metas'].data['img_name']}]

    print('\nDang thuc hien suy luan...')
    model.eval()
    with torch.no_grad():
        result = model.simple_test(img_tensor, img_metas, rescale=False)
    print('Suy luan hoan tat.')

    bbox_kpt_results = result[0]
    pred_bboxes_all = bbox_kpt_results[0][0]
    pred_keypoints_all = bbox_kpt_results[1][0]
    scores = pred_bboxes_all[:, -1]
    keep_mask = scores > SCORE_THRESHOLD
    final_pred_keypoints_3d = pred_keypoints_all[keep_mask]
    gt_keypoints_3d = data['gt_keypoints'].data.numpy()

    gt_labels = [f'GT {i}' for i in range(gt_keypoints_3d.shape[0])]
    pred_labels = [f'Pred {i}' for i in range(final_pred_keypoints_3d.shape[0])]

    metrics = calculate_detailed_metrics(final_pred_keypoints_3d, gt_keypoints_3d)
    print_detailed_metrics(metrics, gt_labels, pred_labels)

    visualize_comparison_3d(
        final_pred_keypoints_3d,
        gt_keypoints_3d,
        pred_labels,
        gt_labels,
        title=f'So sanh Tu the 3D cho {gt_keypoints_3d.shape[0]} nguoi (Mau {sample_index})')


if __name__ == '__main__':
    main()
