import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis.model_palette import get_model_color
from tools.analysis.model_labels import PUBLIC_MODEL_LABELS


DEFAULT_PAPER_DIR = PROJECT_ROOT / 'work_dirs' / 'paper_M1-5'
DEFAULT_EXPERIMENT_LOG = DEFAULT_PAPER_DIR / 'logs' / 'experiment_log.csv'
DEFAULT_OUTPUT_PREFIX = DEFAULT_PAPER_DIR / 'figures' / 'figure4_qualitative'
DEFAULT_MODELS = ['M0', 'M3', 'M4']
FULL_PAPER_FIGURE_DIR = (
    PROJECT_ROOT / 'paper_assets' / 'manuscript_latex' /
    'resfes2026_witidar' / 'figures')
DEFAULT_FULL_PAPER_OUTPUT_PREFIX = FULL_PAPER_FIGURE_DIR / 'fig_qualitative_full_paper'
DEFAULT_FULL_PAPER_RGB_DIR = FULL_PAPER_FIGURE_DIR / 'qualitative_rgb'
FULL_PAPER_SAMPLE_INDICES = [231, 7218, 4416]
FULL_PAPER_SAMPLE_NAMES = ['S11_06_319', 'S52_40_322', 'S23_12_337']
MODEL_DISPLAY_NAMES = dict(PUBLIC_MODEL_LABELS)
FULL_PAPER_MODEL_PATHS = {
    'M0': {
        'config': 'work_dirs/full_alation_20e/M0/petr_wifi.py',
        'checkpoint': 'work_dirs/full_alation_20e/M0/latest.pth',
    },
    'M7': {
        'config': 'work_dirs/full_alation_20e/M7/wi_tidir_wifi_transformer.py',
        'checkpoint': 'work_dirs/full_alation_20e/M7/latest.pth',
    },
    'M9_RF2': {
        'config': 'configs/wifi/wi_tidir_wifi_mamba2_flattened_eval_rf_2step.py',
        'checkpoint': 'work_dirs/full_alation_20e/M9/latest.pth',
    },
}
LIMBS = [
    [0, 1], [1, 2], [2, 5], [3, 0], [4, 2], [5, 7], [6, 3], [7, 3],
    [8, 4], [9, 5], [10, 6], [11, 7], [12, 9], [13, 11],
]
LABEL_ANCHOR_JOINT_IDX = 13
QUALITATIVE_ACCENT_MODELS = frozenset({'M0', 'M7', 'M9_RF2'})


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


def resolve_full_paper_model_asset(project_root, model_id):
    project_root = Path(project_root)
    if model_id not in FULL_PAPER_MODEL_PATHS:
        raise KeyError(f'Unsupported full-paper model: {model_id}')
    paths = FULL_PAPER_MODEL_PATHS[model_id]
    config = project_root / paths['config']
    checkpoint = project_root / paths['checkpoint']
    if not config.exists():
        raise FileNotFoundError(f'Missing config for {model_id}: {config}')
    if not checkpoint.exists():
        raise FileNotFoundError(f'Missing checkpoint for {model_id}: {checkpoint}')
    return {
        'model_id': model_id,
        'config': config,
        'checkpoint': checkpoint,
        'display_name': MODEL_DISPLAY_NAMES[model_id],
        'notes': '',
    }


def resolve_rgb_frame_path(rgb_frame_dir, sample_name):
    path = Path(rgb_frame_dir) / f'{sample_name}.png'
    if not path.exists():
        raise FileNotFoundError(f'Missing RGB frame for {sample_name}: {path}')
    return path


def build_panel_titles(model_ids):
    return ['Ground Truth'] + [
        f'{model_id}: {MODEL_DISPLAY_NAMES.get(model_id, model_id)}'
        for model_id in model_ids
    ]


def get_model_title_accent(model_id):
    if model_id not in QUALITATIVE_ACCENT_MODELS:
        return None
    return get_model_color(model_id)


def get_panel_grid_position(sample_idx, panel_key, include_rgb=True):
    panel_offsets = {
        'rgb': (0, 0),
        'gt': (0, 1),
        'model_0': (0, 2),
        'model_1': (0, 3),
        'model_2': (0, 4),
    }
    if panel_key not in panel_offsets:
        raise KeyError(f'Unsupported panel key: {panel_key}')
    row_offset, col_idx = panel_offsets[panel_key]
    if not include_rgb:
        if panel_key == 'rgb':
            raise KeyError('RGB panel requested when include_rgb=False.')
        col_idx -= 1
    return sample_idx + row_offset, col_idx


def parse_args():
    parser = argparse.ArgumentParser(description='Render Figure 4 qualitative comparisons for WiFi pose models.')
    parser.add_argument('--paper-dir', default=str(DEFAULT_PAPER_DIR))
    parser.add_argument('--experiment-log', default=str(DEFAULT_EXPERIMENT_LOG))
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
    parser.add_argument('--sample-indices', type=int, nargs='*', default=None)
    parser.add_argument('--num-samples', type=int, default=3)
    parser.add_argument('--min-people', type=int, default=1)
    parser.add_argument('--score-thr', type=float, default=0.2)
    parser.add_argument('--output-prefix', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--dpi', type=int, default=220)
    parser.add_argument('--show-unmatched', action='store_true')
    parser.add_argument('--match-quality-thr-mm', type=float, default=200.0)
    parser.add_argument('--match-threshold-mm', type=float, default=500.0)
    parser.add_argument('--rgb-frame-dir', default=None)
    parser.add_argument(
        '--source-video-root',
        default=None,
        help='Directory containing video folders; used to extract RGB frames for selected samples.')
    parser.add_argument(
        '--auto-select-samples',
        action='store_true',
        help='Select full-paper samples from available videos using matching diagnostics.')
    parser.add_argument(
        '--bounds-mode',
        choices=['matched', 'gt'],
        default=None,
        help='Pose axis bounds policy. Full-paper defaults to gt; other renders default to matched.')
    parser.add_argument(
        '--full-paper',
        action='store_true',
        help='Render M0/M7/M9_RF2 comparisons for manuscript Figure 8.')
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


def extract_test_dataset_signature(cfg, dataset=None):
    test_cfg = cfg.data.test
    dataset_root = getattr(test_cfg, 'dataset_root', None)
    mode = getattr(test_cfg, 'mode', None)
    if dataset_root is not None:
        dataset_root = Path(dataset_root).expanduser()
        if not dataset_root.is_absolute():
            dataset_root = PROJECT_ROOT / dataset_root
        dataset_root = str(dataset_root.resolve())
    if dataset is None:
        return (dataset_root, mode)
    sample_names = getattr(dataset, 'filename_list', None)
    if sample_names is None:
        raise ValueError('Dataset does not expose filename_list for signature validation.')
    digest = hashlib.sha256('\n'.join(map(str, sample_names)).encode('utf-8')).hexdigest()
    return (dataset_root, mode, digest)


def validate_dataset_signatures(signatures):
    if not signatures:
        return
    baseline_model, baseline_signature = signatures[0]
    for model_id, signature in signatures[1:]:
        if signature != baseline_signature:
            raise ValueError(
                f'Mismatched test dataset signature: {model_id}={signature} differs from '
                f'{baseline_model}={baseline_signature}')


def sample_name_to_video_id(sample_name):
    return str(sample_name).rsplit('_', 1)[0]


def available_video_ids(source_video_root):
    source_video_root = Path(source_video_root)
    if not source_video_root.exists():
        return set()
    return {
        path.name
        for path in source_video_root.iterdir()
        if path.is_dir() and (path / 'output.mkv').exists() and (path / 'time_list.txt').exists()
    }


def collect_candidate_indices(config_path, min_people, allowed_video_ids=None):
    _, _, _, _, Config, _, _, build_dataset = _lazy_runtime_imports()
    cfg, dataset = _build_visual_dataset(config_path, Config, build_dataset)
    candidates = []
    for index in range(len(dataset)):
        data = dataset[index]
        gt_keypoints = data['gt_keypoints'].data.numpy()
        if allowed_video_ids is not None:
            sample_name = data['img_metas'].data['img_name']
            if sample_name_to_video_id(sample_name) not in allowed_video_ids:
                continue
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


def _match_predictions(pred_keypoints, gt_keypoints, linear_sum_assignment, np,
                       max_distance_mm=None):
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
        distance_mm = float(cost_matrix[gt_idx, pred_idx]) * 1000.0
        if max_distance_mm is not None and distance_mm > max_distance_mm:
            continue
        matches.append((int(gt_idx), int(pred_idx)))
    return matches


def _panel_colors(count):
    return ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2'][:count]


def compute_shared_pose_bounds(pose_sets, min_range=0.35, margin_scale=0.6,
                               equal_axes=False):
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
        axis_span = max_span if equal_axes else max_vals[axis_index] - min_vals[axis_index]
        half = max(axis_span, min_range) / 2.0 + margin
        return (center - half, center + half)

    xlim = _axis_bounds(0)
    ylim = _axis_bounds(1)
    z_low, z_high = _axis_bounds(2)
    return {
        'xlim': xlim,
        'ylim': ylim,
        'zlim': (z_high, z_low),
    }


def compute_row_pose_bounds(gt_keypoints, prepared_displays, bounds_mode='matched',
                            min_range=0.35, margin_scale=0.6, equal_axes=False):
    if bounds_mode == 'gt':
        pose_sets = [gt_keypoints]
    elif bounds_mode == 'matched':
        pose_sets = [gt_keypoints]
        pose_sets.extend(display['matched_poses'] for display in prepared_displays)
    else:
        raise ValueError(f'Unsupported bounds mode: {bounds_mode}')
    return compute_shared_pose_bounds(
        pose_sets,
        min_range=min_range,
        margin_scale=margin_scale,
        equal_axes=equal_axes)


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


def write_matching_diagnostics(rows, output_path, score_threshold, match_threshold_mm):
    panels = []
    for row in rows:
        for model_id, metrics in row['metrics'].items():
            panels.append({
                'sample_index': row['sample_index'],
                'sample_name': row['img_name'],
                'ground_truth_count': row['gt_count'],
                'model_id': model_id,
                'matched_count': metrics['matched_count'],
                'false_positives': metrics['false_positives'],
                'false_negatives': metrics['false_negatives'],
                'matched_mpjpe_mm': metrics['matched_error_mm'],
            })
    payload = {
        'score_threshold': score_threshold,
        'match_threshold_mm': match_threshold_mm,
        'panels': panels,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + '\n',
        encoding='utf-8')
    return output_path


def extract_rgb_frames_for_rows(source_video_root, rows, output_dir):
    from tools.analysis.extract_qualitative_rgb_frames import extract_frames_for_sample_names

    sample_names = [row['img_name'] for row in rows]
    return extract_frames_for_sample_names(source_video_root, sample_names, output_dir=output_dir)


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


def score_sample_candidate(summary, target_model_id=None):
    metrics = summary.get('metrics', {})
    gt_count = summary.get('gt_count', 0)
    crowding_score = summary.get('crowding_score', 0.0)
    all_metrics = list(metrics.values())
    avg_false_positives = sum(metric.get('false_positives', 0) for metric in all_metrics) / max(len(all_metrics), 1)
    avg_false_negatives = sum(metric.get('false_negatives', 0) for metric in all_metrics) / max(len(all_metrics), 1)
    matched_errors = [metric.get('matched_error_mm') for metric in all_metrics if metric.get('matched_error_mm') is not None]
    error_spread = (max(matched_errors) - min(matched_errors)) if len(matched_errors) >= 2 else 0.0

    if target_model_id and target_model_id in metrics:
        target = metrics[target_model_id]
        target_error = target.get('matched_error_mm')
        target_matched = target.get('matched_count', 0)
        target_false_negatives = target.get('false_negatives', max(0, gt_count - target_matched))
        target_false_positives = target.get('false_positives', 0)
        baseline_errors = [
            metric.get('matched_error_mm')
            for model_id, metric in metrics.items()
            if model_id != target_model_id and metric.get('matched_error_mm') is not None
        ]
        total_matched = sum(metric.get('matched_count', 0) for metric in all_metrics)
        total_false_negatives = sum(metric.get('false_negatives', 0) for metric in all_metrics)
        total_false_positives = sum(metric.get('false_positives', 0) for metric in all_metrics)

        score = 10.0 * gt_count
        score += 8.0 * crowding_score
        score += 120.0 * target_matched
        score += 35.0 * total_matched
        score -= 180.0 * target_false_negatives
        score -= 90.0 * total_false_negatives
        score -= 10.0 * target_false_positives
        score -= 4.0 * total_false_positives
        if target_error is not None:
            score -= 0.30 * target_error
            if baseline_errors:
                score += 0.45 * (sum(baseline_errors) / len(baseline_errors) - target_error)
        return score

    score = 40.0 * gt_count
    score += 25.0 * crowding_score
    score += 6.0 * avg_false_positives
    score += 10.0 * avg_false_negatives
    score += 0.05 * error_spread
    return score


def target_sample_quality_rank(summary, target_model_id):
    metrics = summary.get('metrics', {})
    gt_count = summary.get('gt_count', 0)
    target = metrics.get(target_model_id)
    if not target:
        return 9

    def _complete(metric):
        return (
            metric.get('matched_count', 0) >= gt_count and
            metric.get('false_negatives', max(0, gt_count - metric.get('matched_count', 0))) == 0)

    target_complete = _complete(target)
    if not target_complete:
        return 3
    if metrics and all(_complete(metric) for metric in metrics.values()):
        return 0
    baseline = metrics.get('M0')
    if baseline is None or _complete(baseline):
        return 1
    return 2


def select_best_sample_indices(sample_summaries, num_samples, target_model_id=None):
    def _rank_key(item):
        quality_rank = (
            target_sample_quality_rank(item, target_model_id)
            if target_model_id else 0)
        return (
            quality_rank,
            -score_sample_candidate(item, target_model_id=target_model_id),
            -item.get('gt_count', 0),
            item.get('sample_index', 0))

    ranked = sorted(
        sample_summaries,
        key=_rank_key)
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


def resolve_render_options(full_paper=False, explicit_show_unmatched=False, dpi=220,
                           bounds_mode=None):
    if full_paper:
        return {
            'show_unmatched': False,
            'dpi': max(dpi, 300),
            'target_model_id': 'M9_RF2',
            'bounds_mode': bounds_mode or 'gt',
        }
    return {
        'show_unmatched': bool(explicit_show_unmatched),
        'dpi': dpi,
        'target_model_id': None,
        'bounds_mode': bounds_mode or 'matched',
    }


def format_panel_footer(sample_index, img_name, gt_count, matched_count, false_positives,
                        matched_error_mm, false_negatives=0):
    error_text = 'n/a' if matched_error_mm is None else f'{matched_error_mm:.1f} mm'
    return (
        f'S{sample_index} | {img_name}\n'
        f'M{matched_count}/{gt_count} | FP {false_positives} | FN {false_negatives} | '
        f'E{error_text}'
    )


def format_panel_title(base_title, sample_index=None, top_row=False,
                       column_header_only=False):
    if column_header_only and not top_row:
        return ''
    if top_row:
        if ': ' in base_title:
            prefix, remainder = base_title.split(': ', 1)
            remainder = remainder.replace(' with ', '\nwith ')
            return f'{prefix}:\n{remainder}'
        return base_title
    prefix = base_title.split(':', 1)[0]
    if sample_index is None:
        return prefix
    return f'{prefix}\nS{sample_index}'


def get_render_style_config():
    return {
        'title_font_size': 12.5,
        'title_pad': 3,
        'footer_font_size': 7.7,
        'bounds_min_range': 0.12,
        'bounds_margin_scale': 0.04,
        'equal_axes': True,
    }


def _plot_pose_set(ax, poses, colors, labels, title, Line2D, bounds=None,
                   title_accent=None):
    style_config = get_render_style_config()
    if title:
        ax.set_title(title, fontsize=style_config['title_font_size'], pad=style_config['title_pad'])
    if title_accent and title:
        ax.annotate(
            '',
            xy=(0.36, 1.01),
            xytext=(0.64, 1.01),
            xycoords='axes fraction',
            arrowprops={'arrowstyle': '-', 'color': title_accent, 'lw': 3.0})
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
        cfg, dataset = _build_visual_dataset(asset['config'], Config, build_dataset)
        loaded_cfgs.append((asset['model_id'], cfg, dataset))
    signatures = [
        (model_id, extract_test_dataset_signature(cfg, dataset))
        for model_id, cfg, dataset in loaded_cfgs
    ]
    validate_dataset_signatures(signatures)
    print(
        'Validated shared test dataset: '
        f'root={signatures[0][1][0]} mode={signatures[0][1][1]} '
        f'ordered_sample_sha256={signatures[0][1][2]}')
    dataset = loaded_cfgs[0][2]
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


def collect_sample_summary(runtime, model_assets, sample_index, score_thr=0.2,
                           match_threshold_mm=500.0):
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
        matches = _match_predictions(
            pred_keypoints,
            gt_keypoints,
            linear_sum_assignment,
            np,
            max_distance_mm=match_threshold_mm)
        metrics = summarize_prediction_quality(pred_keypoints, gt_keypoints, matches, np)
        summary['models'].append({
            'asset': asset,
            'pred_keypoints': pred_keypoints,
            'matches': matches,
        })
        summary['metrics'][asset['model_id']] = metrics
    return summary


def render_qualitative_figure(model_assets, sample_indices, output_prefix, score_thr=0.2, device=None, dpi=220,
                              show_unmatched=False, precomputed_rows=None, match_quality_thr_mm=200.0,
                              match_threshold_mm=500.0, rgb_frame_dir=None, bounds_mode='matched'):
    runtime = None
    if precomputed_rows is None:
        runtime = prepare_runtime(model_assets, device=device)
        plt = runtime['plt']
        Line2D = runtime['Line2D']
    else:
        _, _, plt, Line2D, _, _, _, _ = _lazy_runtime_imports()
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    style_config = get_render_style_config()

    if precomputed_rows is None:
        rows = [collect_sample_summary(
                    runtime,
                    model_assets,
                    sample_index,
                    score_thr=score_thr,
                    match_threshold_mm=match_threshold_mm)
                for sample_index in sample_indices]
    else:
        row_lookup = {row['sample_index']: row for row in precomputed_rows}
        rows = [row_lookup[sample_index] for sample_index in sample_indices if sample_index in row_lookup]

    grid_rows = len(rows)
    include_rgb = rgb_frame_dir is not None
    grid_cols = len(model_assets) + (2 if include_rgb else 1)
    figure_width = 3.05 * grid_cols
    figure_height = 3.55 * grid_rows
    fig = plt.figure(figsize=(figure_width, figure_height))
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
            false_negatives=0,
            matched_error_mm=None)
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
        row_bounds = compute_row_pose_bounds(
            gt_keypoints,
            prepared_displays,
            bounds_mode=bounds_mode,
            min_range=style_config['bounds_min_range'],
            margin_scale=style_config['bounds_margin_scale'],
            equal_axes=style_config['equal_axes'])

        if include_rgb:
            rgb_row_idx, rgb_col_idx = get_panel_grid_position(
                row_idx, 'rgb', include_rgb=include_rgb)
            rgb_subplot_idx = rgb_row_idx * grid_cols + rgb_col_idx + 1
            rgb_ax = fig.add_subplot(grid_rows, grid_cols, rgb_subplot_idx)
            rgb_path = resolve_rgb_frame_path(rgb_frame_dir, row['img_name'])
            rgb_ax.imshow(plt.imread(rgb_path))
            rgb_title = 'RGB Scene' if row_idx == 0 else ''
            rgb_ax.set_title(
                f'{rgb_title}\n{len(gt_keypoints)}-person'.strip(),
                fontsize=style_config['title_font_size'],
                pad=style_config['title_pad'])
            rgb_ax.axis('off')
            rgb_ax.text(
                0.02,
                0.01,
                row['img_name'],
                transform=rgb_ax.transAxes,
                fontsize=style_config['footer_font_size'],
                color='white',
                bbox={'facecolor': 'black', 'alpha': 0.65, 'pad': 1.5})

        gt_row_idx, gt_col_idx = get_panel_grid_position(
            row_idx, 'gt', include_rgb=include_rgb)
        gt_subplot_idx = gt_row_idx * grid_cols + gt_col_idx + 1
        gt_title = format_panel_title(
            titles[0],
            sample_index=row['sample_index'],
            top_row=(row_idx == 0),
            column_header_only=True)
        gt_ax = fig.add_subplot(grid_rows, grid_cols, gt_subplot_idx, projection='3d')
        _plot_pose_set(gt_ax, gt_keypoints, gt_colors, gt_labels, gt_title, Line2D, bounds=row_bounds)
        gt_ax.text2D(
            0.02,
            0.01,
            gt_footer,
            transform=gt_ax.transAxes,
            fontsize=style_config['footer_font_size'])

        for col_idx, (model_entry, display) in enumerate(zip(row['models'], prepared_displays), start=1):
            model_asset = model_entry['asset']
            metric = row['metrics'][model_asset['model_id']]

            grid_row_idx, grid_col_idx = get_panel_grid_position(
                row_idx,
                f'model_{col_idx - 1}',
                include_rgb=include_rgb)
            subplot_idx = grid_row_idx * grid_cols + grid_col_idx + 1
            title = format_panel_title(
                titles[col_idx],
                sample_index=row['sample_index'],
                top_row=(row_idx == 0),
                column_header_only=True)
            axis = fig.add_subplot(
                grid_rows,
                grid_cols,
                subplot_idx,
                projection='3d')
            _plot_pose_set(
                axis,
                display['poses'],
                display['colors'],
                display['labels'],
                title,
                Line2D,
                bounds=row_bounds,
                title_accent=get_model_title_accent(model_asset['model_id']))
            axis.text2D(
                0.02,
                0.01,
                format_panel_footer(
                    sample_index=row['sample_index'],
                    img_name=row['img_name'],
                    gt_count=len(gt_keypoints),
                    matched_count=metric['matched_count'],
                    false_positives=metric['false_positives'],
                    false_negatives=metric['false_negatives'],
                    matched_error_mm=metric['matched_error_mm']),
                transform=axis.transAxes,
                fontsize=style_config['footer_font_size'])

    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.01, top=0.98, wspace=0.02, hspace=0.10)

    outputs = []
    for suffix in ('.png', '.pdf'):
        output_path = output_prefix.with_suffix(suffix)
        fig.savefig(output_path, dpi=dpi if suffix == '.png' else None, bbox_inches='tight')
        outputs.append(output_path)
    plt.close(fig)
    diagnostics_path = output_prefix.with_name(
        f'{output_prefix.name}_diagnostics').with_suffix('.json')
    outputs.append(write_matching_diagnostics(
        rows,
        diagnostics_path,
        score_threshold=score_thr,
        match_threshold_mm=match_threshold_mm))
    return outputs


def main():
    args = parse_args()
    render_options = resolve_render_options(
        full_paper=args.full_paper,
        explicit_show_unmatched=args.show_unmatched,
        dpi=args.dpi,
        bounds_mode=args.bounds_mode)
    if args.full_paper:
        model_assets = [
            resolve_full_paper_model_asset(PROJECT_ROOT, model_id)
            for model_id in ('M0', 'M7', 'M9_RF2')
        ]
        for asset in model_assets:
            print(
                f"{asset['model_id']}: config={asset['config']} "
                f"checkpoint={asset['checkpoint']}")
        sample_indices = args.sample_indices or FULL_PAPER_SAMPLE_INDICES
        output_prefix = args.output_prefix or DEFAULT_FULL_PAPER_OUTPUT_PREFIX
        rgb_frame_dir = args.rgb_frame_dir or DEFAULT_FULL_PAPER_RGB_DIR
        show_unmatched = render_options['show_unmatched']
        dpi = render_options['dpi']
        precomputed_rows = None
    else:
        specs = load_experiment_specs(args.experiment_log)
        model_assets = [
            resolve_model_assets(PROJECT_ROOT, args.paper_dir, model_id, specs)
            for model_id in args.models
        ]
        sample_indices = args.sample_indices
        output_prefix = args.output_prefix or DEFAULT_OUTPUT_PREFIX
        rgb_frame_dir = args.rgb_frame_dir
        show_unmatched = render_options['show_unmatched']
        dpi = render_options['dpi']

    if args.full_paper and args.auto_select_samples:
        allowed_video_ids = (
            available_video_ids(args.source_video_root)
            if args.source_video_root else None)
        if args.source_video_root and not allowed_video_ids:
            raise RuntimeError(
                f'No usable video folders found in source root: {args.source_video_root}')
        candidates, _ = collect_candidate_indices(
            model_assets[0]['config'],
            args.min_people,
            allowed_video_ids=allowed_video_ids)
        runtime = prepare_runtime(model_assets, device=args.device)
        summaries = [
            collect_sample_summary(
                runtime,
                model_assets,
                sample_index,
                score_thr=args.score_thr,
                match_threshold_mm=args.match_threshold_mm)
            for sample_index in candidates
        ]
        sample_indices = select_best_sample_indices(
            summaries,
            args.num_samples,
            target_model_id=render_options['target_model_id'])
        print('Auto-selected sample indices:', sample_indices)
        precomputed_rows = summaries
    elif args.full_paper:
        precomputed_rows = None
    elif sample_indices:
        precomputed_rows = None
    else:
        candidates, _ = collect_candidate_indices(model_assets[0]['config'], args.min_people)
        runtime = prepare_runtime(model_assets, device=args.device)
        summaries = [
            collect_sample_summary(
                runtime,
                model_assets,
                sample_index,
                score_thr=args.score_thr,
                match_threshold_mm=args.match_threshold_mm)
            for sample_index in candidates
        ]
        sample_indices = select_best_sample_indices(
            summaries,
            args.num_samples,
            target_model_id=render_options['target_model_id'])
        print('Auto-selected sample indices:', sample_indices)
        precomputed_rows = summaries
    if not sample_indices:
        raise RuntimeError('No candidate samples found for qualitative rendering.')

    if args.full_paper and args.source_video_root:
        if precomputed_rows is not None:
            row_lookup = {row['sample_index']: row for row in precomputed_rows}
            rgb_rows = [row_lookup[index] for index in sample_indices if index in row_lookup]
        elif sample_indices == FULL_PAPER_SAMPLE_INDICES:
            rgb_rows = [{'img_name': sample_name} for sample_name in FULL_PAPER_SAMPLE_NAMES]
        else:
            raise RuntimeError(
                'Cannot extract RGB frames for custom sample indices without precomputed rows. '
                'Use --auto-select-samples or pre-extract --rgb-frame-dir manually.')
        extract_rgb_frames_for_rows(args.source_video_root, rgb_rows, rgb_frame_dir)

    outputs = render_qualitative_figure(
        model_assets=model_assets,
        sample_indices=sample_indices,
        output_prefix=output_prefix,
        score_thr=args.score_thr,
        device=args.device,
        dpi=dpi,
        show_unmatched=show_unmatched,
        precomputed_rows=precomputed_rows,
        match_quality_thr_mm=args.match_quality_thr_mm,
        match_threshold_mm=args.match_threshold_mm,
        rgb_frame_dir=rgb_frame_dir,
        bounds_mode=render_options['bounds_mode'])
    print('Qualitative figure outputs:')
    for output_path in outputs:
        print(output_path)


if __name__ == '__main__':
    main()
