import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAPER_DIR = PROJECT_ROOT / 'work_dirs' / 'paper_M1-5'
DEFAULT_EXPERIMENT_LOG = DEFAULT_PAPER_DIR / 'logs' / 'experiment_log.csv'
DEFAULT_OUTPUT_PREFIX = DEFAULT_PAPER_DIR / 'figures' / 'figure4_qualitative'
DEFAULT_MODELS = ['M0', 'M3', 'M4']
MODEL_DISPLAY_NAMES = {
    'M0': 'Person-in-WiFi 3D',
    'M1': 'Person-in-WiFi 3D + Spectral Tokens',
    'M2': 'Person-in-WiFi 3D + Mamba',
    'M3': 'FlowPose-WiFi (Transformer)',
    'M4': 'FlowPose-WiFi (Ours)',
}
LIMBS = [
    [0, 1], [1, 2], [2, 5], [3, 0], [4, 2], [5, 7], [6, 3], [7, 3],
    [8, 4], [9, 5], [10, 6], [11, 7], [12, 9], [13, 11],
]
LABEL_ANCHOR_JOINT_IDX = 13


def load_experiment_specs(csv_path):
    specs = {}
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return specs
    with csv_path.open('r', newline='', encoding='utf-8') as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            experiment_id = (row.get('experiment_id') or '').strip()
            if not experiment_id:
                continue
            config_value = (row.get('config') or '').strip()
            specs[experiment_id] = {
                'config': config_value,
                'config_name': Path(config_value).name if config_value else '',
                'checkpoint': (row.get('checkpoint') or '').strip(),
                'notes': (row.get('notes') or '').strip(),
            }
    return specs


def resolve_model_assets(project_root, paper_dir, model_id, specs, config_override=None, checkpoint_override=None):
    project_root = Path(project_root)
    paper_dir = Path(paper_dir)
    spec = specs.get(model_id, {})

    checkpoint = Path(checkpoint_override) if checkpoint_override else paper_dir / model_id / 'epoch_20.pth'
    config_name = Path(config_override).name if config_override else spec.get('config_name', '')
    if config_override:
        config = Path(config_override)
    elif config_name:
        config = project_root / 'configs' / 'wifi' / config_name
    else:
        raise FileNotFoundError(f'Cannot resolve config for {model_id}')

    if not checkpoint.exists():
        raise FileNotFoundError(f'Missing checkpoint for {model_id}: {checkpoint}')
    if not config.exists():
        raise FileNotFoundError(f'Missing config for {model_id}: {config}')

    return {
        'model_id': model_id,
        'config': config,
        'checkpoint': checkpoint,
        'display_name': MODEL_DISPLAY_NAMES.get(model_id, model_id),
        'notes': spec.get('notes', ''),
    }


def build_panel_titles(model_ids):
    return ['Ground Truth'] + [
        f'{model_id}: {MODEL_DISPLAY_NAMES.get(model_id, model_id)}'
        for model_id in model_ids
    ]


def parse_args():
    parser = argparse.ArgumentParser(description='Render Figure 4 qualitative comparisons for WiFi pose models.')
    parser.add_argument('--paper-dir', default=str(DEFAULT_PAPER_DIR))
    parser.add_argument('--experiment-log', default=str(DEFAULT_EXPERIMENT_LOG))
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
    parser.add_argument('--sample-indices', type=int, nargs='*', default=None)
    parser.add_argument('--num-samples', type=int, default=3)
    parser.add_argument('--min-people', type=int, default=2)
    parser.add_argument('--score-thr', type=float, default=0.2)
    parser.add_argument('--output-prefix', default=str(DEFAULT_OUTPUT_PREFIX))
    parser.add_argument('--device', default=None)
    parser.add_argument('--dpi', type=int, default=220)
    parser.add_argument('--show-unmatched', action='store_true')
    return parser.parse_args()


def _lazy_runtime_imports():
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mmcv import Config
    from scipy.optimize import linear_sum_assignment
    from opera.apis import init_detector
    from opera.datasets import build_dataset
    return np, torch, plt, Line2D, Config, linear_sum_assignment, init_detector, build_dataset


def _build_visual_dataset(config_path, Config, build_dataset):
    cfg = Config.fromfile(str(config_path))
    vis_pipeline = [
        dict(type='opera.DefaultFormatBundle', extra_keys=['gt_keypoints']),
        dict(type='mmdet.Collect', keys=['img', 'gt_keypoints'], meta_keys=['img_name']),
    ]
    cfg.data.test.pipeline = vis_pipeline
    dataset = build_dataset(cfg.data.test)
    return cfg, dataset


def collect_candidate_indices(config_path, min_people):
    _, _, _, _, Config, _, _, build_dataset = _lazy_runtime_imports()
    cfg, dataset = _build_visual_dataset(config_path, Config, build_dataset)
    candidates = []
    for index in range(len(dataset)):
        data = dataset[index]
        gt_keypoints = data['gt_keypoints'].data.numpy()
        if gt_keypoints.shape[0] >= min_people:
            candidates.append(index)
    return candidates, cfg


def _normalize_device(device, torch):
    if device:
        return device
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def _predict_sample(model_asset, sample_index, score_thr, device):
    np, torch, _, _, Config, _, init_detector, build_dataset = _lazy_runtime_imports()
    cfg, dataset = _build_visual_dataset(model_asset['config'], Config, build_dataset)
    data = dataset[sample_index]
    img_tensor = data['img'].data.to(device).unsqueeze(0)
    img_metas = [{'img_name': data['img_metas'].data['img_name']}]

    model = init_detector(str(model_asset['config']), str(model_asset['checkpoint']), device=device)
    model.eval()
    with torch.no_grad():
        result = model.simple_test(img_tensor, img_metas, rescale=False)

    bbox_kpt_results = result[0]
    pred_bboxes_all = bbox_kpt_results[0][0]
    pred_keypoints_all = bbox_kpt_results[1][0]
    scores = pred_bboxes_all[:, -1]
    keep_mask = scores > score_thr
    return {
        'gt_keypoints': data['gt_keypoints'].data.numpy(),
        'pred_keypoints': pred_keypoints_all[keep_mask],
        'img_name': data['img_metas'].data['img_name'],
        'cfg': cfg,
        'num_pred': int(keep_mask.sum()),
    }


def _match_predictions(pred_keypoints, gt_keypoints, linear_sum_assignment, np):
    matches = []
    if len(pred_keypoints) == 0 or len(gt_keypoints) == 0:
        return matches
    cost_matrix = np.zeros((len(gt_keypoints), len(pred_keypoints)))
    for gt_idx in range(len(gt_keypoints)):
        for pred_idx in range(len(pred_keypoints)):
            cost_matrix[gt_idx, pred_idx] = np.mean(
                np.linalg.norm(gt_keypoints[gt_idx] - pred_keypoints[pred_idx], axis=1))
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        matches.append((int(gt_idx), int(pred_idx)))
    return matches


def _panel_colors(count):
    return ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2'][:count]


def prepare_display_predictions(pred_keypoints, matches, gt_colors, gt_labels, show_unmatched=False):
    pred_keypoints = list(pred_keypoints)
    matched_by_gt = sorted(matches, key=lambda item: item[0])
    shown_poses = []
    shown_colors = []
    shown_labels = []
    used_pred_indices = set()

    for gt_idx, pred_idx in matched_by_gt:
        shown_poses.append(pred_keypoints[pred_idx])
        shown_colors.append(gt_colors[gt_idx])
        shown_labels.append(gt_labels[gt_idx])
        used_pred_indices.add(pred_idx)

    if show_unmatched:
        for pred_idx, pose in enumerate(pred_keypoints):
            if pred_idx in used_pred_indices:
                continue
            shown_poses.append(pose)
            shown_colors.append('#7f7f7f')
            shown_labels.append(f'U{pred_idx + 1}')

    return {
        'poses': shown_poses,
        'colors': shown_colors,
        'labels': shown_labels,
        'shown_count': len(shown_poses),
        'matched_count': len(matched_by_gt),
        'total_count': len(pred_keypoints),
        'hidden_unmatched': len(pred_keypoints) - len(matched_by_gt),
    }


def _plot_pose_set(ax, poses, colors, labels, title, Line2D):
    ax.set_title(title, fontsize=12, pad=10)
    all_points = []
    for person_idx, person_kpts in enumerate(poses):
        color = colors[person_idx]
        all_points.extend(person_kpts)
        x, y, z = person_kpts[:, 0], person_kpts[:, 1], person_kpts[:, 2]
        ax.scatter(x, y, z, c=color, marker='o', s=18)
        for limb in LIMBS:
            start, end = person_kpts[limb[0]], person_kpts[limb[1]]
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=color,
                linewidth=2.0)
        anchor_joint = person_kpts[LABEL_ANCHOR_JOINT_IDX]
        ax.text(anchor_joint[0], anchor_joint[1], anchor_joint[2] + 0.05, labels[person_idx], color=color, fontsize=8)

    if all_points:
        import numpy as np  # local only for plotting bounds
        all_points = np.array(all_points)
        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)
        mid_vals = (min_vals + max_vals) / 2
        max_range = max((max_vals - min_vals).max() * 0.6, 0.35)
        ax.set_xlim(mid_vals[0] - max_range, mid_vals[0] + max_range)
        ax.set_ylim(mid_vals[1] - max_range, mid_vals[1] + max_range)
        z_min = mid_vals[2] - max_range
        z_max = mid_vals[2] + max_range
        ax.set_zlim(z_max, z_min)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.view_init(elev=20.0, azim=-75)
    ax.grid(False)


def render_qualitative_figure(model_assets, sample_indices, output_prefix, score_thr=0.2, device=None, dpi=220,
                              show_unmatched=False):
    np, torch, plt, Line2D, _, linear_sum_assignment, _, _ = _lazy_runtime_imports()
    device = _normalize_device(device, torch)
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for sample_index in sample_indices:
        row = {'sample_index': sample_index, 'models': []}
        gt_keypoints = None
        img_name = None
        for asset in model_assets:
            result = _predict_sample(asset, sample_index, score_thr=score_thr, device=device)
            if gt_keypoints is None:
                gt_keypoints = result['gt_keypoints']
                img_name = result['img_name']
            row['models'].append({
                'asset': asset,
                'pred_keypoints': result['pred_keypoints'],
            })
        row['gt_keypoints'] = gt_keypoints
        row['img_name'] = img_name
        rows.append(row)

    fig = plt.figure(figsize=(4.4 * (len(model_assets) + 1), 4.1 * len(rows)))
    titles = build_panel_titles([asset['model_id'] for asset in model_assets])

    for row_idx, row in enumerate(rows):
        gt_keypoints = row['gt_keypoints']
        gt_labels = [f'P{i + 1}' for i in range(len(gt_keypoints))]
        gt_colors = _panel_colors(len(gt_keypoints))
        gt_ax = fig.add_subplot(len(rows), len(model_assets) + 1, row_idx * (len(model_assets) + 1) + 1, projection='3d')
        gt_title = titles[0] if row_idx == 0 else f'Ground Truth\nSample {row["sample_index"]}'
        _plot_pose_set(gt_ax, gt_keypoints, gt_colors, gt_labels, gt_title, Line2D)

        for col_idx, model_entry in enumerate(row['models'], start=1):
            model_asset = model_entry['asset']
            pred_keypoints = model_entry['pred_keypoints']
            matches = _match_predictions(pred_keypoints, gt_keypoints, linear_sum_assignment, np)
            display = prepare_display_predictions(
                pred_keypoints=pred_keypoints,
                matches=matches,
                gt_colors=gt_colors,
                gt_labels=gt_labels,
                show_unmatched=show_unmatched)

            axis = fig.add_subplot(
                len(rows),
                len(model_assets) + 1,
                row_idx * (len(model_assets) + 1) + col_idx + 1,
                projection='3d')
            if row_idx == 0:
                title = titles[col_idx]
            else:
                title = f'{model_asset["model_id"]}\nSample {row["sample_index"]}'
            _plot_pose_set(axis, display['poses'], display['colors'], display['labels'], title, Line2D)
            axis.text2D(
                0.02,
                0.02,
                f'Shown: {display["shown_count"]} | Matched: {display["matched_count"]} | Total: {display["total_count"]} | GT: {len(gt_keypoints)}',
                transform=axis.transAxes,
                fontsize=8)

    fig.suptitle('Figure 4. Qualitative comparison on challenging multi-person WiFi pose samples.', fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    outputs = []
    for suffix in ('.png', '.pdf', '.svg'):
        output_path = output_prefix.with_suffix(suffix)
        fig.savefig(output_path, dpi=dpi if suffix == '.png' else None, bbox_inches='tight')
        outputs.append(output_path)
    plt.close(fig)
    return outputs


def main():
    args = parse_args()
    specs = load_experiment_specs(args.experiment_log)
    model_assets = [
        resolve_model_assets(PROJECT_ROOT, args.paper_dir, model_id, specs)
        for model_id in args.models
    ]
    if args.sample_indices:
        sample_indices = args.sample_indices
    else:
        candidates, _ = collect_candidate_indices(model_assets[0]['config'], args.min_people)
        sample_indices = candidates[:args.num_samples]
    if not sample_indices:
        raise RuntimeError('No candidate samples found for qualitative rendering.')

    outputs = render_qualitative_figure(
        model_assets=model_assets,
        sample_indices=sample_indices,
        output_prefix=args.output_prefix,
        score_thr=args.score_thr,
        device=args.device,
        dpi=args.dpi,
        show_unmatched=args.show_unmatched)
    print('Figure 4 outputs:')
    for output_path in outputs:
        print(output_path)


if __name__ == '__main__':
    main()
