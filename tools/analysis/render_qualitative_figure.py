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
    parser.add_argument('--min-people', type=int, default=1)
    parser.add_argument('--score-thr', type=float, default=0.2)
    parser.add_argument('--output-prefix', default=str(DEFAULT_OUTPUT_PREFIX))
    parser.add_argument('--device', default=None)
    parser.add_argument('--dpi', type=int, default=220)
    parser.add_argument('--show-unmatched', action='store_true')
    parser.add_argument('--match-quality-thr-mm', type=float, default=200.0)
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


def extract_test_dataset_signature(cfg):
    test_cfg = cfg.data.test
    dataset_root = getattr(test_cfg, 'dataset_root', None)
    mode = getattr(test_cfg, 'mode', None)
    return (dataset_root, mode)


def validate_dataset_signatures(signatures):
    if not signatures:
        return
    baseline_model, baseline_signature = signatures[0]
    for model_id, signature in signatures[1:]:
        if signature != baseline_signature:
            raise ValueError(
                f'Mismatched test dataset signature: {model_id}={signature} differs from '
                f'{baseline_model}={baseline_signature}')


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


def _predict_from_result(result, score_thr):
    bbox_kpt_results = result[0]
    pred_bboxes_all = bbox_kpt_results[0][0]
    pred_keypoints_all = bbox_kpt_results[1][0]
    scores = pred_bboxes_all[:, -1]
    keep_mask = scores > score_thr
    return pred_keypoints_all[keep_mask]


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


def compute_shared_pose_bounds(pose_sets, min_range=0.35, margin_scale=0.6):
    points = []
    for poses in pose_sets:
        for pose in poses:
            points.extend(pose)
    if not points:
        return {
            'xlim': (-1.0, 1.0),
            'ylim': (-1.0, 1.0),
            'zlim': (1.0, -1.0),
        }

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    min_vals = [min(xs), min(ys), min(zs)]
    max_vals = [max(xs), max(ys), max(zs)]
    spans = [max_vals[i] - min_vals[i] for i in range(3)]
    max_span = max(max(spans), min_range)
    margin = max_span * margin_scale

    def _axis_bounds(axis_index):
        center = (min_vals[axis_index] + max_vals[axis_index]) / 2.0
        half = max(max_vals[axis_index] - min_vals[axis_index], min_range) / 2.0 + margin
        return (center - half, center + half)

    xlim = _axis_bounds(0)
    ylim = _axis_bounds(1)
    z_low, z_high = _axis_bounds(2)
    return {
        'xlim': xlim,
        'ylim': ylim,
        'zlim': (z_high, z_low),
    }


def build_match_details(pred_keypoints, gt_keypoints, matches, np):
    details = []
    for gt_idx, pred_idx in sorted(matches, key=lambda item: item[0]):
        error_mm = float(np.mean(np.linalg.norm(gt_keypoints[gt_idx] - pred_keypoints[pred_idx], axis=1)) * 1000.0)
        details.append({
            'gt_idx': gt_idx,
            'pred_idx': pred_idx,
            'error_mm': error_mm,
        })
    return details


def prepare_display_predictions(pred_keypoints, match_details, gt_colors, gt_labels, show_unmatched=False,
                                match_quality_thr_mm=None):
    pred_keypoints = list(pred_keypoints)
    matched_by_gt = sorted(match_details, key=lambda item: item['gt_idx'])
    shown_poses = []
    shown_colors = []
    shown_labels = []
    used_pred_indices = set()
    matched_poses = []
    poor_match_count = 0

    for detail in matched_by_gt:
        gt_idx = detail['gt_idx']
        pred_idx = detail['pred_idx']
        shown_poses.append(pred_keypoints[pred_idx])
        matched_poses.append(pred_keypoints[pred_idx])
        if match_quality_thr_mm is not None and detail['error_mm'] > match_quality_thr_mm:
            shown_colors.append('#7f7f7f')
            poor_match_count += 1
        else:
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
        'matched_poses': matched_poses,
        'colors': shown_colors,
        'labels': shown_labels,
        'shown_count': len(shown_poses),
        'matched_count': len(matched_by_gt),
        'total_count': len(pred_keypoints),
        'hidden_unmatched': len(pred_keypoints) - len(matched_by_gt),
        'poor_match_count': poor_match_count,
    }


def summarize_prediction_quality(pred_keypoints, gt_keypoints, matches, np):
    match_details = build_match_details(pred_keypoints, gt_keypoints, matches, np)
    matched_errors = [detail['error_mm'] for detail in match_details]
    matched_error_mm = float(sum(matched_errors) / len(matched_errors)) if matched_errors else None
    return {
        'match_details': match_details,
        'matched_error_mm': matched_error_mm,
        'matched_count': len(matches),
        'false_positives': max(0, len(pred_keypoints) - len(matches)),
        'false_negatives': max(0, len(gt_keypoints) - len(matches)),
    }


def compute_crowding_score(gt_keypoints, np):
    if len(gt_keypoints) < 2:
        return 0.0
    centers = [np.mean(person, axis=0) for person in gt_keypoints]
    nearest_distances = []
    for idx, center in enumerate(centers):
        others = [np.linalg.norm(center - other) for j, other in enumerate(centers) if j != idx]
        if others:
            nearest_distances.append(min(others))
    if not nearest_distances:
        return 0.0
    mean_nearest = float(sum(nearest_distances) / len(nearest_distances))
    return 1.0 / max(mean_nearest, 1e-6)


def score_sample_candidate(summary):
    metrics = summary.get('metrics', {})
    gt_count = summary.get('gt_count', 0)
    crowding_score = summary.get('crowding_score', 0.0)
    all_metrics = list(metrics.values())
    avg_false_positives = sum(metric.get('false_positives', 0) for metric in all_metrics) / max(len(all_metrics), 1)
    avg_false_negatives = sum(metric.get('false_negatives', 0) for metric in all_metrics) / max(len(all_metrics), 1)
    matched_errors = [metric.get('matched_error_mm') for metric in all_metrics if metric.get('matched_error_mm') is not None]
    error_spread = (max(matched_errors) - min(matched_errors)) if len(matched_errors) >= 2 else 0.0

    score = 40.0 * gt_count
    score += 25.0 * crowding_score
    score += 6.0 * avg_false_positives
    score += 10.0 * avg_false_negatives
    score += 0.05 * error_spread
    return score


def select_best_sample_indices(sample_summaries, num_samples):
    ranked = sorted(
        sample_summaries,
        key=lambda item: (-score_sample_candidate(item), -item.get('gt_count', 0), item.get('sample_index', 0)))
    selected = []
    selected_indices = set()

    desired_buckets = [1, 2, 3] if num_samples >= 3 else []
    for bucket in desired_buckets:
        for item in ranked:
            if item.get('gt_count') != bucket:
                continue
            sample_index = item.get('sample_index')
            if sample_index in selected_indices:
                continue
            selected.append(sample_index)
            selected_indices.add(sample_index)
            break

    for item in ranked:
        if len(selected) >= num_samples:
            break
        sample_index = item.get('sample_index')
        if sample_index in selected_indices:
            continue
        selected.append(sample_index)
        selected_indices.add(sample_index)

    return selected[:num_samples]


def format_panel_footer(sample_index, img_name, gt_count, matched_count, false_positives, matched_error_mm,
                        poor_match_count=0):
    error_text = 'n/a' if matched_error_mm is None else f'{matched_error_mm:.1f} mm'
    return (
        f'Sample {sample_index} | {img_name}\n'
        f'Match {matched_count}/{gt_count} | FP {false_positives} | Poor {poor_match_count} | Matched Error {error_text}'
    )


def _plot_pose_set(ax, poses, colors, labels, title, Line2D, bounds=None):
    ax.set_title(title, fontsize=12, pad=10)
    for person_idx, person_kpts in enumerate(poses):
        color = colors[person_idx]
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

    if bounds:
        ax.set_xlim(*bounds['xlim'])
        ax.set_ylim(*bounds['ylim'])
        ax.set_zlim(*bounds['zlim'])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.view_init(elev=20.0, azim=-75)
    ax.grid(False)


def prepare_runtime(model_assets, device=None):
    np, torch, plt, Line2D, Config, linear_sum_assignment, init_detector, build_dataset = _lazy_runtime_imports()
    device = _normalize_device(device, torch)
    loaded_cfgs = []
    for asset in model_assets:
        cfg, _ = _build_visual_dataset(asset['config'], Config, build_dataset)
        loaded_cfgs.append((asset['model_id'], cfg))
    validate_dataset_signatures([
        (model_id, extract_test_dataset_signature(cfg))
        for model_id, cfg in loaded_cfgs
    ])
    dataset = _build_visual_dataset(model_assets[0]['config'], Config, build_dataset)[1]
    models = {}
    for asset in model_assets:
        model = init_detector(str(asset['config']), str(asset['checkpoint']), device=device)
        model.eval()
        models[asset['model_id']] = model
    return {
        'np': np,
        'torch': torch,
        'plt': plt,
        'Line2D': Line2D,
        'linear_sum_assignment': linear_sum_assignment,
        'dataset': dataset,
        'device': device,
        'models': models,
    }


def collect_sample_summary(runtime, model_assets, sample_index, score_thr=0.2):
    np = runtime['np']
    torch = runtime['torch']
    linear_sum_assignment = runtime['linear_sum_assignment']
    data = runtime['dataset'][sample_index]
    gt_keypoints = data['gt_keypoints'].data.numpy()
    img_tensor = data['img'].data.to(runtime['device']).unsqueeze(0)
    img_metas = [{'img_name': data['img_metas'].data['img_name']}]

    summary = {
        'sample_index': sample_index,
        'img_name': data['img_metas'].data['img_name'],
        'gt_keypoints': gt_keypoints,
        'gt_count': len(gt_keypoints),
        'crowding_score': compute_crowding_score(gt_keypoints, np),
        'models': [],
        'metrics': {},
    }

    for asset in model_assets:
        model = runtime['models'][asset['model_id']]
        with torch.no_grad():
            result = model.simple_test(img_tensor, img_metas, rescale=False)
        pred_keypoints = _predict_from_result(result, score_thr=score_thr)
        matches = _match_predictions(pred_keypoints, gt_keypoints, linear_sum_assignment, np)
        metrics = summarize_prediction_quality(pred_keypoints, gt_keypoints, matches, np)
        summary['models'].append({
            'asset': asset,
            'pred_keypoints': pred_keypoints,
            'matches': matches,
        })
        summary['metrics'][asset['model_id']] = metrics
    return summary


def render_qualitative_figure(model_assets, sample_indices, output_prefix, score_thr=0.2, device=None, dpi=220,
                              show_unmatched=False, precomputed_rows=None, match_quality_thr_mm=200.0):
    runtime = None
    if precomputed_rows is None:
        runtime = prepare_runtime(model_assets, device=device)
        plt = runtime['plt']
        Line2D = runtime['Line2D']
    else:
        _, _, plt, Line2D, _, _, _, _ = _lazy_runtime_imports()
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    if precomputed_rows is None:
        rows = [collect_sample_summary(runtime, model_assets, sample_index, score_thr=score_thr)
                for sample_index in sample_indices]
    else:
        row_lookup = {row['sample_index']: row for row in precomputed_rows}
        rows = [row_lookup[sample_index] for sample_index in sample_indices if sample_index in row_lookup]

    fig = plt.figure(figsize=(4.4 * (len(model_assets) + 1), 4.1 * len(rows)))
    titles = build_panel_titles([asset['model_id'] for asset in model_assets])

    for row_idx, row in enumerate(rows):
        gt_keypoints = row['gt_keypoints']
        gt_labels = [f'P{i + 1}' for i in range(len(gt_keypoints))]
        gt_colors = _panel_colors(len(gt_keypoints))
        gt_footer = format_panel_footer(
            sample_index=row['sample_index'],
            img_name=row['img_name'],
            gt_count=len(gt_keypoints),
            matched_count=len(gt_keypoints),
            false_positives=0,
            matched_error_mm=None)
        row_display_sets = [gt_keypoints]
        prepared_displays = []
        for model_entry in row['models']:
            metric = row['metrics'][model_entry['asset']['model_id']]
            display = prepare_display_predictions(
                pred_keypoints=model_entry['pred_keypoints'],
                match_details=metric['match_details'],
                gt_colors=gt_colors,
                gt_labels=gt_labels,
                show_unmatched=show_unmatched,
                match_quality_thr_mm=match_quality_thr_mm)
            prepared_displays.append(display)
            row_display_sets.append(display['matched_poses'])
        row_bounds = compute_shared_pose_bounds(row_display_sets)

        gt_ax = fig.add_subplot(len(rows), len(model_assets) + 1, row_idx * (len(model_assets) + 1) + 1, projection='3d')
        gt_title = titles[0] if row_idx == 0 else f'Ground Truth\nSample {row["sample_index"]}'
        _plot_pose_set(gt_ax, gt_keypoints, gt_colors, gt_labels, gt_title, Line2D, bounds=row_bounds)
        gt_ax.text2D(
            0.02,
            0.02,
            gt_footer,
            transform=gt_ax.transAxes,
            fontsize=7.5)

        for col_idx, (model_entry, display) in enumerate(zip(row['models'], prepared_displays), start=1):
            model_asset = model_entry['asset']
            metric = row['metrics'][model_asset['model_id']]

            axis = fig.add_subplot(
                len(rows),
                len(model_assets) + 1,
                row_idx * (len(model_assets) + 1) + col_idx + 1,
                projection='3d')
            if row_idx == 0:
                title = titles[col_idx]
            else:
                title = f'{model_asset["model_id"]}\nSample {row["sample_index"]}'
            _plot_pose_set(axis, display['poses'], display['colors'], display['labels'], title, Line2D, bounds=row_bounds)
            axis.text2D(
                0.02,
                0.02,
                format_panel_footer(
                    sample_index=row['sample_index'],
                    img_name=row['img_name'],
                    gt_count=len(gt_keypoints),
                    matched_count=metric['matched_count'],
                    false_positives=metric['false_positives'],
                    matched_error_mm=metric['matched_error_mm'],
                    poor_match_count=display['poor_match_count']),
                transform=axis.transAxes,
                fontsize=7.5)

    fig.suptitle('Figure 4. Selected challenging multi-person WiFi pose samples.', fontsize=16, y=0.99)
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
        precomputed_rows = None
    else:
        candidates, _ = collect_candidate_indices(model_assets[0]['config'], args.min_people)
        runtime = prepare_runtime(model_assets, device=args.device)
        summaries = [
            collect_sample_summary(runtime, model_assets, sample_index, score_thr=args.score_thr)
            for sample_index in candidates
        ]
        sample_indices = select_best_sample_indices(summaries, args.num_samples)
        print('Auto-selected sample indices:', sample_indices)
        precomputed_rows = summaries
    if not sample_indices:
        raise RuntimeError('No candidate samples found for qualitative rendering.')

    outputs = render_qualitative_figure(
        model_assets=model_assets,
        sample_indices=sample_indices,
        output_prefix=args.output_prefix,
        score_thr=args.score_thr,
        device=args.device,
        dpi=args.dpi,
        show_unmatched=args.show_unmatched,
        precomputed_rows=precomputed_rows,
        match_quality_thr_mm=args.match_quality_thr_mm)
    print('Figure 4 outputs:')
    for output_path in outputs:
        print(output_path)


if __name__ == '__main__':
    main()
