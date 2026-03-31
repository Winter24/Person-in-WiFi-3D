import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D
from mmcv import Config

from opera.apis import init_detector
from opera.datasets import build_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_FILE = PROJECT_ROOT / 'configs' / 'wifi' / 'petr_wifi.py'
DEFAULT_CHECKPOINT_FILE = PROJECT_ROOT / 'work_dirs' / 'petr_wifi' / 'epoch_47.pth'

SCORE_THRESHOLD = 0.1
SAMPLE_INDEX = 25
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize one WiFi pose sample.')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_FILE))
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT_FILE))
    return parser.parse_args()


def visualize_comparison_3d(pred_keypoints, gt_keypoints,
                            title='So sanh Tu the 3D'):
    """Plot predicted and ground-truth poses on the same 3D figure."""
    limbs = [
        [0, 1], [1, 2], [2, 5], [3, 0], [4, 2], [5, 7],
        [6, 3], [7, 3], [8, 4], [9, 5], [10, 6], [11, 7],
        [12, 9], [13, 11]
    ]

    gt_color = 'blue'
    pred_color = 'red'

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    all_points = []

    def plot_poses(keypoints_set, color, marker):
        if keypoints_set is None or len(keypoints_set) == 0:
            return
        for person_kpts in keypoints_set:
            all_points.extend(person_kpts)
            x, y, z = person_kpts[:, 0], person_kpts[:, 1], person_kpts[:, 2]
            ax.scatter(x, y, z, c=color, marker=marker)
            for limb in limbs:
                start, end = person_kpts[limb[0]], person_kpts[limb[1]]
                ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color=color)

    plot_poses(gt_keypoints, gt_color, 'o')
    plot_poses(pred_keypoints, pred_color, '^')

    if not all_points:
        print('Khong co diem nao de ve.')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
    else:
        all_points = np.array(all_points)
        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)

        mid_vals = (min_vals + max_vals) / 2
        max_range = (max_vals - min_vals).max() / 2.0

        ax.set_xlim(mid_vals[0] - max_range, mid_vals[0] + max_range)
        ax.set_ylim(mid_vals[1] - max_range, mid_vals[1] + max_range)

        z_min = mid_vals[2] - max_range
        z_max = mid_vals[2] + max_range
        ax.set_zlim(z_max, z_min)

    legend_elements = [
        Line2D([0], [0], color=gt_color, lw=4, label='Ground-Truth'),
        Line2D([0], [0], color=pred_color, lw=4, label='Du doan')
    ]
    ax.legend(handles=legend_elements)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20., azim=-75)
    plt.show()


def main():
    args = parse_args()
    print(f'Dang tai mo hinh tu:\n- Config: {args.config}\n- Checkpoint: {args.checkpoint}')
    model = init_detector(args.config, args.checkpoint, device=DEVICE)
    print('Tai mo hinh thanh cong.')

    print(f'Dang tai mau du lieu thu {SAMPLE_INDEX} tu tap test...')
    cfg = Config.fromfile(args.config)

    vis_pipeline = [
        dict(type='opera.DefaultFormatBundle', extra_keys=['gt_keypoints', 'gt_labels']),
        dict(type='mmdet.Collect', keys=['img', 'gt_keypoints'], meta_keys=['img_name'])
    ]
    cfg.data.test.pipeline = vis_pipeline

    dataset = build_dataset(cfg.data.test)
    data = dataset[SAMPLE_INDEX]

    img_tensor = data['img'].data.to(DEVICE).unsqueeze(0)
    img_metas = [{'img_name': data['img_metas'].data['img_name']}]

    print('Dang thuc hien suy luan...')
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

    print('-' * 30)
    print(f'Tong so ung vien du doan: {pred_keypoints_all.shape[0]}.')
    print(f'So nguoi duoc giu lai (diem > {SCORE_THRESHOLD}): {final_pred_keypoints_3d.shape[0]}.')
    print(f'So nguoi trong ground-truth: {gt_keypoints_3d.shape[0]}.')
    print('-' * 30)

    visualize_comparison_3d(
        final_pred_keypoints_3d,
        gt_keypoints_3d,
        title=f'So sanh Tu the 3D (Mau {SAMPLE_INDEX})')


if __name__ == '__main__':
    main()
