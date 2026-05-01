"""Render a presentation video from WiFi CSI samples and model predictions.

This script builds a frame sequence from train/test sample lists such as
``S11_01_308`` -> video ``S11_01`` frame ``308``. It can render skeleton-only
videos from CSI + keypoints, or add a source RGB/video panel when the original
camera video is available outside the WiFi dataset.
"""

import argparse
import importlib.util
import re
from pathlib import Path
from typing import NamedTuple, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / 'data' / 'wifipose'
DEFAULT_PAPER_DIR = PROJECT_ROOT / 'work_dirs' / 'paper_M1-5'
DEFAULT_EXPERIMENT_LOG = DEFAULT_PAPER_DIR / 'logs' / 'experiment_log.csv'
DEFAULT_OUTPUT_DIR = DEFAULT_PAPER_DIR / 'figures'
DEFAULT_SPLITS = ['train_data', 'test_data']
DEFAULT_MODEL = 'M4'
DEFAULT_SOURCE_VIDEO_FILENAMES = ['output.mkv', 'output.mp4', 'output.avi', 'output.mov']


class ParsedSample(NamedTuple):
    sample_name: str
    video_id: str
    frame_id: int


class SequenceRecord(NamedTuple):
    sample_name: str
    video_id: str
    frame_id: int
    split: str
    dataset_index: int
    csi_path: Path
    keypoint_path: Path


def parse_sample_name(sample_name):
    parts = sample_name.strip().split('_')
    if len(parts) < 3:
        raise ValueError(f'Invalid sample name: {sample_name!r}')
    frame_text = parts[-1]
    if not frame_text.isdigit():
        raise ValueError(f'Invalid frame id in sample name: {sample_name!r}')
    return ParsedSample(
        sample_name=sample_name.strip(),
        video_id='_'.join(parts[:-1]),
        frame_id=int(frame_text))


def _split_mode(split):
    if split == 'train_data':
        return 'train'
    if split == 'test_data':
        return 'test'
    if split.endswith('_data'):
        return split[:-5]
    return split


def _list_path_for_split(split_root, split):
    return Path(split_root) / f'{split}_list.txt'


def _read_split_names(split_root, split):
    list_path = _list_path_for_split(split_root, split)
    if not list_path.exists():
        raise FileNotFoundError(f'Missing split list: {list_path}')
    return [line.strip().split()[0] for line in list_path.read_text(encoding='utf-8').splitlines() if line.strip()]


def collect_sequence_records(data_root, video_id, splits=DEFAULT_SPLITS, require_files=True):
    """Collect and sort records for one video id across split folders."""
    data_root = Path(data_root)
    records = []
    seen = {}
    for split in splits:
        split_root = data_root / split
        for dataset_index, sample_name in enumerate(_read_split_names(split_root, split)):
            parsed = parse_sample_name(sample_name)
            if parsed.video_id != video_id:
                continue
            if parsed.sample_name in seen:
                raise ValueError(
                    f'Duplicate sample name {parsed.sample_name!r} in {split}; '
                    f'previously found in {seen[parsed.sample_name]}')
            seen[parsed.sample_name] = split
            csi_path = split_root / 'csi' / f'{parsed.sample_name}.mat'
            keypoint_path = split_root / 'keypoint' / f'{parsed.sample_name}.npy'
            if require_files:
                if not csi_path.exists():
                    raise FileNotFoundError(f'Missing CSI file: {csi_path}')
                if not keypoint_path.exists():
                    raise FileNotFoundError(f'Missing keypoint file: {keypoint_path}')
            records.append(SequenceRecord(
                sample_name=parsed.sample_name,
                video_id=parsed.video_id,
                frame_id=parsed.frame_id,
                split=split,
                dataset_index=dataset_index,
                csi_path=csi_path,
                keypoint_path=keypoint_path))
    return sorted(records, key=lambda record: (record.frame_id, record.split, record.sample_name))


def _contiguous_runs(records):
    if not records:
        return []
    runs = []
    current = [records[0]]
    for record in records[1:]:
        if record.frame_id == current[-1].frame_id + 1:
            current.append(record)
        else:
            runs.append(current)
            current = [record]
    runs.append(current)
    return runs


def select_contiguous_segment(records, segment='longest', start_frame=None, end_frame=None,
                              max_frames=None, stride=1):
    """Select a renderable frame segment.

    Explicit frame bounds are applied before the contiguous-run policy. Stride
    and max_frames are applied last so users can downsample long clips.
    """
    if stride < 1:
        raise ValueError('stride must be >= 1')
    selected = list(records)
    if start_frame is not None:
        selected = [record for record in selected if record.frame_id >= start_frame]
    if end_frame is not None:
        selected = [record for record in selected if record.frame_id <= end_frame]
    selected = sorted(selected, key=lambda record: (record.frame_id, record.split, record.sample_name))
    if not selected:
        return []

    if segment == 'longest':
        runs = _contiguous_runs(selected)
        selected = max(runs, key=lambda run: (len(run), -run[0].frame_id))
    elif segment == 'first':
        selected = _contiguous_runs(selected)[0]
    elif segment == 'all':
        pass
    else:
        raise ValueError(f'Unsupported segment policy: {segment!r}')

    selected = selected[::stride]
    if max_frames is not None and max_frames > 0:
        selected = selected[:max_frames]
    return selected


def summarize_runs(records):
    runs = _contiguous_runs(sorted(records, key=lambda record: record.frame_id))
    return [
        {
            'start_frame': run[0].frame_id,
            'end_frame': run[-1].frame_id,
            'length': len(run),
            'splits': sorted({record.split for record in run}),
        }
        for run in runs
    ]


def resolve_source_video_path(video_id, source_video=None, source_video_root=None,
                              filenames=DEFAULT_SOURCE_VIDEO_FILENAMES):
    """Resolve original video path.

    Supports either an explicit file path or a dataset-style folder:
    ``source_video_root / video_id / output.mkv``.
    """
    if source_video:
        source_video = Path(source_video)
        if source_video.is_dir():
            for filename in filenames:
                candidate = source_video / filename
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f'No source video found in directory: {source_video}')
        if not source_video.exists():
            raise FileNotFoundError(f'Missing source video: {source_video}')
        return source_video

    if source_video_root is None:
        return None
    video_dir = Path(source_video_root) / video_id
    for filename in filenames:
        candidate = video_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f'No source video found for {video_id} under {source_video_root}. '
        f'Expected one of: {", ".join(filenames)}')


def resolve_time_list_path(video_id, source_video_path=None, source_video_root=None, time_list=None):
    if time_list:
        time_list = Path(time_list)
        if not time_list.exists():
            raise FileNotFoundError(f'Missing time list: {time_list}')
        return time_list
    if source_video_path:
        candidate = Path(source_video_path).parent / 'time_list.txt'
        if candidate.exists():
            return candidate
    if source_video_root:
        candidate = Path(source_video_root) / video_id / 'time_list.txt'
        if candidate.exists():
            return candidate
    return None


def _extract_frame_id_token(token, video_id=None):
    token = token.strip()
    if not token:
        return None
    try:
        parsed = parse_sample_name(token)
        if video_id is None or parsed.video_id == video_id:
            return parsed.frame_id
    except ValueError:
        pass
    if re.fullmatch(r'-?\d+', token):
        return int(token)
    match = re.search(r'(\d+)$', token)
    if match:
        return int(match.group(1))
    return None


def load_time_index_map(time_list_path, video_id=None):
    """Load mapping from dataset frame id to source video frame index.

    The parser accepts common lightweight formats:
    ``S52_40_10 0``, ``10 0``, ``S52_40_10,0``, or timestamp rows such as
    ``0_2023-04-08 13:21:21.195959``. Timestamp rows map source frame indices
    to themselves, which matches datasets where sample frame ids reference the
    original video frame number.
    """
    if time_list_path is None:
        return {}
    time_list_path = Path(time_list_path)
    mapping = {}
    for source_index, line in enumerate(time_list_path.read_text(encoding='utf-8').splitlines()):
        clean = line.strip()
        if not clean or clean.startswith('#'):
            continue
        timestamp_match = re.match(r'^(\d+)_\d{4}-\d{2}-\d{2}(?:\s|$)', clean)
        if timestamp_match:
            frame_id = int(timestamp_match.group(1))
            mapping[frame_id] = frame_id
            continue
        tokens = [token for token in re.split(r'[\s,;]+', clean) if token]
        if not tokens:
            continue
        frame_id = _extract_frame_id_token(tokens[0], video_id=video_id)
        if frame_id is None:
            continue
        if len(tokens) >= 2 and re.fullmatch(r'-?\d+', tokens[1]):
            video_index = int(tokens[1])
        else:
            video_index = source_index
        mapping[frame_id] = video_index
    return mapping


def resolve_source_frame_index(frame_id, frame_offset=0, time_index_map=None):
    if time_index_map and frame_id in time_index_map:
        return int(time_index_map[frame_id]) + int(frame_offset)
    return int(frame_id) + int(frame_offset)


def _load_qualitative_module():
    script_path = Path(__file__).resolve().with_name('render_qualitative_figure.py')
    spec = importlib.util.spec_from_file_location('render_qualitative_figure_for_video', script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build_split_dataset(config_path, data_root, split, Config, build_dataset):
    cfg = Config.fromfile(str(config_path))
    vis_pipeline = [
        dict(type='opera.DefaultFormatBundle', extra_keys=['gt_keypoints']),
        dict(type='mmdet.Collect', keys=['img', 'gt_keypoints'], meta_keys=['img_name']),
    ]
    cfg.data.test.pipeline = vis_pipeline
    cfg.data.test.dataset_root = str(Path(data_root) / split)
    cfg.data.test.mode = _split_mode(split)
    return build_dataset(cfg.data.test)


def build_video_model_assets(target_asset, baseline_asset=None):
    if baseline_asset is None:
        return [target_asset]
    return [baseline_asset, target_asset]


def reorder_gt_by_previous_frame(current_gt, previous_gt, np):
    """Reorder current GT people to keep visual identities stable over time."""
    current_gt = np.asarray(current_gt)
    if previous_gt is None or len(previous_gt) == 0 or len(current_gt) == 0:
        return current_gt
    previous_gt = np.asarray(previous_gt)
    if len(previous_gt) != len(current_gt):
        return current_gt
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return current_gt

    cost_matrix = np.zeros((len(previous_gt), len(current_gt)), dtype=float)
    for prev_idx in range(len(previous_gt)):
        for curr_idx in range(len(current_gt)):
            cost_matrix[prev_idx, curr_idx] = np.mean(
                np.linalg.norm(previous_gt[prev_idx] - current_gt[curr_idx], axis=1))
    previous_indices, current_indices = linear_sum_assignment(cost_matrix)
    reordered = np.empty_like(current_gt)
    used_current = set()
    for prev_idx, curr_idx in zip(previous_indices, current_indices):
        reordered[prev_idx] = current_gt[curr_idx]
        used_current.add(curr_idx)
    remaining_current = [idx for idx in range(len(current_gt)) if idx not in used_current]
    for idx in range(len(current_gt)):
        if idx in previous_indices:
            continue
        if remaining_current:
            reordered[idx] = current_gt[remaining_current.pop(0)]
    return reordered


def precompute_global_bounds(frame_summaries, min_range=0.12, margin_scale=0.04):
    pose_sets = []
    for summary in frame_summaries:
        gt_keypoints = summary.get('gt_keypoints')
        if gt_keypoints is not None:
            pose_sets.append(gt_keypoints)
        for output in summary.get('model_outputs', []):
            display = output.get('display')
            if display:
                pose_sets.append(display.get('matched_poses', []))
    if not pose_sets:
        return None
    qualitative = _load_qualitative_module()
    return qualitative.compute_shared_pose_bounds(
        pose_sets,
        min_range=min_range,
        margin_scale=margin_scale)


def _prepare_video_runtime(model_assets, data_root, splits, device=None):
    qualitative = _load_qualitative_module()
    np, torch, plt, Line2D, Config, linear_sum_assignment, init_detector, build_dataset = (
        qualitative._lazy_runtime_imports())
    device = qualitative._normalize_device(device, torch)
    model_assets = list(model_assets)
    if not model_assets:
        raise ValueError('At least one model asset is required.')
    models = {}
    for asset in model_assets:
        model = init_detector(str(asset['config']), str(asset['checkpoint']), device=device)
        model.eval()
        models[asset['model_id']] = model
    datasets = {
        split: _build_split_dataset(model_assets[0]['config'], data_root, split, Config, build_dataset)
        for split in splits
    }
    return {
        'qualitative': qualitative,
        'np': np,
        'torch': torch,
        'plt': plt,
        'Line2D': Line2D,
        'linear_sum_assignment': linear_sum_assignment,
        'models': models,
        'datasets': datasets,
        'data_root': Path(data_root),
        'device': device,
    }


def _load_rgb_frame_from_video(video_capture, frame_id, frame_offset=0, time_index_map=None):
    if video_capture is None:
        return None
    import cv2

    source_frame_index = resolve_source_frame_index(
        frame_id=frame_id,
        frame_offset=frame_offset,
        time_index_map=time_index_map)
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, source_frame_index))
    ok, frame_bgr = video_capture.read()
    if not ok:
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _load_rgb_frame_from_dir(frame_dir, pattern, record):
    if frame_dir is None:
        return None
    frame_dir = Path(frame_dir)
    filename = pattern.format(
        sample_name=record.sample_name,
        video_id=record.video_id,
        frame_id=record.frame_id,
        frame=record.frame_id)
    frame_path = frame_dir / filename
    if not frame_path.exists():
        return None
    from PIL import Image

    return Image.open(frame_path).convert('RGB')


def _finalize_input_wave(wave, np, max_points=240, smooth_window=9, scale=False):
    wave = np.asarray(wave, dtype=np.float64).reshape(-1)
    if wave.size == 0:
        return wave
    if max_points is not None and max_points > 0 and wave.size > max_points:
        sample_idx = np.linspace(0, wave.size - 1, max_points).astype(np.int64)
        wave = wave[sample_idx]
    wave = wave - float(wave.mean())
    if scale:
        std = float(wave.std())
        if std > 1e-6:
            wave = wave / std
    if smooth_window and smooth_window > 1 and wave.size >= smooth_window:
        if smooth_window % 2 == 0:
            smooth_window += 1
        kernel = np.ones(smooth_window, dtype=np.float64) / float(smooth_window)
        wave = np.convolve(wave, kernel, mode='same')
        wave = wave - float(wave.mean())
    return wave.astype(np.float32)


def _input_wave_from_dataset_item(data, np, max_points=240, smooth_window=9, scale=False):
    """Convert the model-ready CSI tensor into a compact waveform for display."""
    img = data['img'].data
    if hasattr(img, 'detach'):
        arr = img.detach().cpu().float().numpy()
    else:
        arr = np.asarray(img, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)

    # Current WiFi samples are usually (3, 3, 20, 60).  Average antenna-pair
    # dimensions and flatten the time/subcarrier grid so the plot looks like a
    # moving CSI signal rather than a static scalar metric.
    if arr.ndim >= 4:
        arr = arr.mean(axis=tuple(range(arr.ndim - 2)))
    elif arr.ndim > 1:
        arr = arr.mean(axis=tuple(range(arr.ndim - 1)))
    return _finalize_input_wave(arr, np, max_points=max_points, smooth_window=smooth_window, scale=scale)


def _raw_amp_wave_from_record(runtime, record, rx=0, tx=0, link_mode='single',
                              max_points=240, smooth_window=9, scale=False):
    """Load raw CSI amplitude from one Rx/Tx link for a more literal radio trace."""
    import h5py

    np = runtime['np']
    csi_path = runtime['data_root'] / record.split / 'csi' / f'{record.sample_name}.mat'
    with h5py.File(csi_path, 'r') as f:
        raw = f['csi_out'][()]
    csi = raw['real'] + raw['imag'] * 1j
    csi = np.asarray(csi)
    if csi.ndim < 4:
        return _finalize_input_wave(
            np.abs(csi),
            np,
            max_points=max_points,
            smooth_window=smooth_window,
            scale=scale)

    # Stored WiFiPose files are (time, subcarrier, rx, tx).  Clamp indices so
    # video rendering keeps running even if a dataset has fewer links.
    rx = min(max(int(rx), 0), csi.shape[2] - 1)
    tx = min(max(int(tx), 0), csi.shape[3] - 1)

    if link_mode == 'all-tx':
        waves = [
            _finalize_input_wave(
                np.abs(csi[:, :, rx, tx_idx]),
                np,
                max_points=max_points,
                smooth_window=smooth_window,
                scale=scale)
            for tx_idx in range(csi.shape[3])
        ]
        return np.stack(waves, axis=0)
    if link_mode == 'all-rx':
        waves = [
            _finalize_input_wave(
                np.abs(csi[:, :, rx_idx, tx]),
                np,
                max_points=max_points,
                smooth_window=smooth_window,
                scale=scale)
            for rx_idx in range(csi.shape[2])
        ]
        return np.stack(waves, axis=0)

    link_amp = np.abs(csi[:, :, rx, tx])
    return _finalize_input_wave(link_amp, np, max_points=max_points, smooth_window=smooth_window, scale=scale)


def _predict_record(runtime, record, model_id, score_thr):
    qualitative = runtime['qualitative']
    torch = runtime['torch']
    data = runtime['datasets'][record.split][record.dataset_index]
    gt_keypoints = data['gt_keypoints'].data.numpy()
    img_tensor = data['img'].data.to(runtime['device']).unsqueeze(0)
    img_metas = [{'img_name': data['img_metas'].data['img_name']}]
    with torch.no_grad():
        result = runtime['models'][model_id].simple_test(img_tensor, img_metas, rescale=False)
    pred_keypoints = qualitative._predict_from_result(result, score_thr=score_thr)
    matches = qualitative._match_predictions(
        pred_keypoints,
        gt_keypoints,
        runtime['linear_sum_assignment'],
        runtime['np'])
    metrics = qualitative.summarize_prediction_quality(
        pred_keypoints,
        gt_keypoints,
        matches,
        runtime['np'])
    return gt_keypoints, pred_keypoints, metrics


def _load_input_wave_for_record(runtime, record, source='preprocessed', rx=0, tx=0,
                                link_mode='single', max_points=240, smooth_window=9, scale=False):
    if source == 'raw-amp':
        return _raw_amp_wave_from_record(
            runtime,
            record,
            rx=rx,
            tx=tx,
            link_mode=link_mode,
            max_points=max_points,
            smooth_window=smooth_window,
            scale=scale)
    data = runtime['datasets'][record.split][record.dataset_index]
    return _input_wave_from_dataset_item(
        data,
        runtime['np'],
        max_points=max_points,
        smooth_window=smooth_window,
        scale=scale)


def _predict_models_for_record(runtime, record, model_assets, score_thr, previous_gt_keypoints=None):
    gt_keypoints = None
    outputs = []
    for asset in model_assets:
        current_gt, pred_keypoints, metrics = _predict_record(
            runtime,
            record,
            asset['model_id'],
            score_thr=score_thr)
        if gt_keypoints is None:
            gt_keypoints = reorder_gt_by_previous_frame(
                current_gt,
                previous_gt_keypoints,
                runtime['np'])
        if gt_keypoints is not current_gt:
            matches = runtime['qualitative']._match_predictions(
                pred_keypoints,
                gt_keypoints,
                runtime['linear_sum_assignment'],
                runtime['np'])
            metrics = runtime['qualitative'].summarize_prediction_quality(
                pred_keypoints,
                gt_keypoints,
                matches,
                runtime['np'])
        outputs.append({
            'asset': asset,
            'pred_keypoints': pred_keypoints,
            'metrics': metrics,
        })
    return gt_keypoints, outputs


def _figure_to_rgb_array(fig, np):
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    if hasattr(fig.canvas, 'buffer_rgba'):
        rgba = np.asarray(fig.canvas.buffer_rgba())
        return rgba[:, :, :3].copy()
    if hasattr(fig.canvas, 'tostring_rgb'):
        return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(height, width, 4)
    return argb[:, :, [1, 2, 3]].copy()


def _normalize_video_frame(frame_rgb):
    """Return a contiguous uint8 RGB frame for video encoders."""
    import numpy as np

    frame = np.asarray(frame_rgb)
    if frame.ndim != 3:
        raise ValueError(f'Expected HxWxC frame, got shape {frame.shape}')
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    if frame.shape[2] != 3:
        raise ValueError(f'Expected RGB/RGBA frame, got shape {frame.shape}')
    if frame.dtype != np.uint8:
        if frame.max(initial=0) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _prepare_model_displays(runtime, gt_keypoints, model_outputs, show_unmatched=False, match_quality_thr_mm=200.0):
    qualitative = runtime['qualitative']
    gt_labels = [f'P{i + 1}' for i in range(len(gt_keypoints))]
    gt_colors = qualitative._panel_colors(len(gt_keypoints))
    displays = []
    for output in model_outputs:
        display = qualitative.prepare_display_predictions(
            pred_keypoints=output['pred_keypoints'],
            match_details=output['metrics']['match_details'],
            gt_colors=gt_colors,
            gt_labels=gt_labels,
            show_unmatched=show_unmatched,
            match_quality_thr_mm=match_quality_thr_mm)
        displays.append(display)
    return gt_labels, gt_colors, displays


def precompute_input_wave_ylim(frame_summaries, np, percentile=98.0, min_ylim=0.5):
    values = []
    for summary in frame_summaries:
        wave = summary.get('input_wave')
        if wave is not None and len(wave) > 0:
            values.append(np.abs(np.asarray(wave, dtype=np.float32).reshape(-1)))
    if not values:
        return min_ylim
    values = np.concatenate(values)
    if values.size == 0:
        return min_ylim
    return max(min_ylim, float(np.percentile(values, percentile)) * 1.25)


def build_input_activity_trace(frame_summaries, np):
    activity = []
    for summary in frame_summaries:
        wave = summary.get('input_wave')
        if wave is None or len(wave) == 0:
            activity.append(0.0)
            continue
        wave = np.asarray(wave, dtype=np.float32)
        activity.append(float(np.sqrt(np.mean(wave * wave))))
    activity = np.asarray(activity, dtype=np.float32)
    if activity.size == 0:
        return activity
    activity = activity - float(activity.min())
    max_value = float(activity.max())
    if max_value > 1e-6:
        activity = activity / max_value
    return activity


def _plot_input_wave(ax, input_wave, sample_name, source='preprocessed', rx=0, tx=0,
                     link_mode='single', ylim=None, activity_trace=None,
                     activity_index=None):
    source_label = 'Model-ready preprocessed CSI'
    if source == 'raw-amp':
        if link_mode == 'all-tx':
            source_label = f'Raw CSI amplitude, all Tx at Rx{rx + 1}'
        elif link_mode == 'all-rx':
            source_label = f'Raw CSI amplitude, all Rx at Tx{tx + 1}'
        else:
            source_label = f'Raw CSI amplitude link Rx{rx + 1}-Tx{tx + 1}'
    ax.set_title(f'{source_label} | {sample_name}', fontsize=12, pad=6)
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('#c8d7e3')
        spine.set_linewidth(0.8)
    ax.tick_params(axis='both', colors='#61778a', labelsize=7, length=2)
    ax.grid(True, color='#dce8f2', alpha=0.75, linewidth=0.6)

    if input_wave is None or len(input_wave) == 0:
        ax.text(0.5, 0.5, 'CSI unavailable', color='#34495e',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    import numpy as np

    input_wave = np.asarray(input_wave)
    waves = input_wave if input_wave.ndim == 2 else input_wave.reshape(1, -1)
    colors = ['#0077b6', '#f77f00', '#2a9d8f', '#9d4edd', '#d62828']
    labels = []
    if source == 'raw-amp' and link_mode == 'all-tx':
        labels = [f'Tx{i + 1}' for i in range(len(waves))]
    elif source == 'raw-amp' and link_mode == 'all-rx':
        labels = [f'Rx{i + 1}' for i in range(len(waves))]
    for wave_idx, wave in enumerate(waves):
        x = range(len(wave))
        color = colors[wave_idx % len(colors)]
        label = labels[wave_idx] if wave_idx < len(labels) else None
        ax.plot(x, wave, color=color, linewidth=1.7, alpha=0.9, label=label)
        if wave_idx == 0:
            ax.fill_between(x, wave, 0, color='#48cae4', alpha=0.12)
    ax.axhline(0, color='#5f6c78', linewidth=0.7, alpha=0.55)
    ax.set_xlim(0, max(1, waves.shape[-1] - 1))
    if ylim is None:
        ylim = max(0.5, float(max(abs(waves.min()), abs(waves.max()))) * 1.15)
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel('temporal CSI trace, centered and lightly smoothed', color='#61778a', fontsize=8)
    ax.set_ylabel('CSI', color='#61778a', fontsize=8)
    if labels:
        ax.legend(loc='upper left', fontsize=7, frameon=False, ncol=min(3, len(labels)))

    if activity_trace is not None and len(activity_trace) > 1:
        activity_ax = ax.twinx()
        activity_x = range(len(activity_trace))
        activity_ax.plot(activity_x, activity_trace, color='#ff7a00', linewidth=1.2, alpha=0.82)
        if activity_index is not None:
            activity_index = min(max(int(activity_index), 0), len(activity_trace) - 1)
            activity_ax.scatter(
                [activity_index],
                [activity_trace[activity_index]],
                s=22,
                color='#d62828',
                zorder=4)
        activity_ax.set_ylim(-0.05, 1.05)
        activity_ax.set_yticks([])
        activity_ax.spines['right'].set_visible(False)
        activity_ax.text(
            0.995,
            0.90,
            'orange: CSI activity over clip',
            ha='right',
            va='top',
            color='#a65d00',
            fontsize=7,
            transform=activity_ax.transAxes)


def _render_frame(runtime, record, model_outputs, gt_keypoints,
                  rgb_frame=None, show_unmatched=False, match_quality_thr_mm=200.0,
                  bounds=None, prepared_displays=None, input_wave=None,
                  input_wave_source='preprocessed', input_wave_rx=0, input_wave_tx=0,
                  input_wave_link_mode='single', input_wave_ylim=None,
                  input_activity_trace=None, input_activity_index=None):
    qualitative = runtime['qualitative']
    np = runtime['np']
    plt = runtime['plt']
    Line2D = runtime['Line2D']

    has_rgb = rgb_frame is not None
    has_input_wave = input_wave is not None
    model_count = len(model_outputs)
    grid_cols = (1 if has_rgb else 0) + 1 + model_count
    fig_height = 5.9 if has_input_wave else 4.8
    fig = plt.figure(figsize=(3.6 * grid_cols, fig_height), facecolor='white')
    if has_input_wave:
        grid = fig.add_gridspec(2, grid_cols, height_ratios=[0.8, 4.0])

        def add_panel(column, projection=None):
            return fig.add_subplot(grid[1, column], projection=projection)

        ax_wave = fig.add_subplot(grid[0, :])
        _plot_input_wave(
            ax_wave,
            input_wave,
            record.sample_name,
            source=input_wave_source,
            rx=input_wave_rx,
            tx=input_wave_tx,
            link_mode=input_wave_link_mode,
            ylim=input_wave_ylim,
            activity_trace=input_activity_trace,
            activity_index=input_activity_index)
    else:
        def add_panel(column, projection=None):
            return fig.add_subplot(1, grid_cols, column + 1, projection=projection)
    gt_labels, gt_colors, displays = _prepare_model_displays(
        runtime,
        gt_keypoints,
        model_outputs,
        show_unmatched=show_unmatched,
        match_quality_thr_mm=match_quality_thr_mm)
    if prepared_displays is not None:
        displays = prepared_displays
    if bounds is None:
        bounds = qualitative.compute_shared_pose_bounds(
            [gt_keypoints] + [display['matched_poses'] for display in displays],
            min_range=0.12,
            margin_scale=0.04)

    col = 0
    if has_rgb:
        ax_rgb = add_panel(col)
        ax_rgb.imshow(rgb_frame)
        ax_rgb.set_title(f'Original Frame\n{record.video_id} #{record.frame_id}', fontsize=13)
        ax_rgb.axis('off')
        col += 1

    ax_gt = add_panel(col, projection='3d')
    qualitative._plot_pose_set(ax_gt, gt_keypoints, gt_colors, gt_labels, 'Ground Truth 3D Pose', Line2D, bounds=bounds)
    col += 1

    footer_parts = [f'{record.sample_name} | split={record.split}']
    for output, display in zip(model_outputs, displays):
        asset = output['asset']
        metrics = output['metrics']
        ax_pred = add_panel(col, projection='3d')
        qualitative._plot_pose_set(
            ax_pred,
            display['poses'],
            display['colors'],
            display['labels'],
            f'{asset["model_id"]}: {asset.get("display_name", asset["model_id"])}',
            Line2D,
            bounds=bounds)
        error_text = 'n/a' if metrics['matched_error_mm'] is None else f'{metrics["matched_error_mm"]:.1f} mm'
        footer_parts.append(
            f'{asset["model_id"]} M{metrics["matched_count"]}/{len(gt_keypoints)} '
            f'FP{metrics["false_positives"]} E{error_text}')
        col += 1
    footer = '\n'.join(footer_parts)
    fig.text(0.5, 0.02, footer, ha='center', va='bottom', fontsize=11)
    fig.suptitle('No camera. No wearable. Just WiFi signals.', fontsize=16, fontweight='bold', y=0.98)
    if has_input_wave:
        fig.subplots_adjust(left=0.02, right=0.99, bottom=0.10, top=0.90, wspace=0.02, hspace=0.22)
    else:
        fig.subplots_adjust(left=0.02, right=0.99, bottom=0.12, top=0.86, wspace=0.02)
    frame = _figure_to_rgb_array(fig, np)
    plt.close(fig)
    return frame


class _ImageioWriter:
    def __init__(self, output_path, fps):
        import imageio.v2 as imageio

        self.writer = imageio.get_writer(
            str(output_path),
            format='FFMPEG',
            mode='I',
            fps=fps,
            codec='libx264',
            pixelformat='yuv420p',
            macro_block_size=16,
            ffmpeg_log_level='warning',
            output_params=['-movflags', '+faststart'])

    def append(self, frame_rgb):
        self.writer.append_data(_normalize_video_frame(frame_rgb))

    def close(self):
        self.writer.close()


class _OpenCvWriter:
    def __init__(self, output_path, fps, first_frame):
        import cv2

        self.cv2 = cv2
        first_frame = _normalize_video_frame(first_frame)
        height, width = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f'OpenCV could not open video writer: {output_path}')

    def append(self, frame_rgb):
        frame_bgr = self.cv2.cvtColor(_normalize_video_frame(frame_rgb), self.cv2.COLOR_RGB2BGR)
        self.writer.write(frame_bgr)

    def close(self):
        self.writer.release()


def _open_video_writer(output_path, fps, first_frame):
    try:
        writer = _ImageioWriter(output_path, fps)
        writer.append(first_frame)
        return writer
    except Exception:
        writer = _OpenCvWriter(output_path, fps, first_frame)
        writer.append(first_frame)
        return writer


def render_presentation_video(records, model_asset, output_path, data_root=DEFAULT_DATA_ROOT,
                              splits=DEFAULT_SPLITS, fps=12, score_thr=0.2, device=None,
                              source_video=None, source_frame_offset=0, source_time_list=None,
                              source_frame_dir=None,
                              source_frame_pattern='{sample_name}.jpg', show_unmatched=False,
                              match_quality_thr_mm=200.0, keep_frames_dir=None, model_assets=None,
                              use_global_bounds=True, show_input_wave=False,
                              input_wave_source='preprocessed', input_wave_rx=0, input_wave_tx=0,
                              input_wave_link_mode='single', input_wave_max_points=240,
                              input_wave_smooth=9):
    if not records:
        raise ValueError('No records selected for rendering.')
    if model_assets is None:
        model_assets = [model_asset]
    else:
        model_assets = list(model_assets)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runtime = _prepare_video_runtime(model_assets, data_root, splits, device=device)

    video_capture = None
    time_index_map = load_time_index_map(source_time_list, video_id=records[0].video_id)
    if source_video:
        import cv2

        video_capture = cv2.VideoCapture(str(source_video))
        if not video_capture.isOpened():
            raise RuntimeError(f'Cannot open source video: {source_video}')

    if keep_frames_dir:
        Path(keep_frames_dir).mkdir(parents=True, exist_ok=True)

    writer = None
    try:
        frame_summaries = []
        previous_gt_keypoints = None
        for record in records:
            gt_keypoints, model_outputs = _predict_models_for_record(
                runtime,
                record,
                model_assets,
                score_thr=score_thr,
                previous_gt_keypoints=previous_gt_keypoints)
            _, _, displays = _prepare_model_displays(
                runtime,
                gt_keypoints,
                model_outputs,
                show_unmatched=show_unmatched,
                match_quality_thr_mm=match_quality_thr_mm)
            for output, display in zip(model_outputs, displays):
                output['display'] = display
            input_wave = (
                _load_input_wave_for_record(
                    runtime,
                    record,
                    source=input_wave_source,
                    rx=input_wave_rx,
                    tx=input_wave_tx,
                    link_mode=input_wave_link_mode,
                    max_points=input_wave_max_points,
                    smooth_window=input_wave_smooth,
                    scale=False)
                if show_input_wave else None)
            frame_summaries.append({
                'record': record,
                'gt_keypoints': gt_keypoints,
                'model_outputs': model_outputs,
                'displays': displays,
                'input_wave': input_wave,
            })
            previous_gt_keypoints = gt_keypoints
        global_bounds = precompute_global_bounds(frame_summaries) if use_global_bounds else None
        input_wave_ylim = (
            precompute_input_wave_ylim(frame_summaries, runtime['np'])
            if show_input_wave else None)
        input_activity_trace = (
            build_input_activity_trace(frame_summaries, runtime['np'])
            if show_input_wave else None)

        for frame_index, record in enumerate(records):
            summary = frame_summaries[frame_index]
            rgb_frame = (
                _load_rgb_frame_from_dir(source_frame_dir, source_frame_pattern, record)
                or _load_rgb_frame_from_video(
                    video_capture,
                    record.frame_id,
                    source_frame_offset,
                    time_index_map=time_index_map)
            )
            frame = _render_frame(
                runtime=runtime,
                record=record,
                model_outputs=summary['model_outputs'],
                gt_keypoints=summary['gt_keypoints'],
                rgb_frame=rgb_frame,
                show_unmatched=show_unmatched,
                match_quality_thr_mm=match_quality_thr_mm,
                bounds=global_bounds,
                prepared_displays=summary['displays'],
                input_wave=summary.get('input_wave'),
                input_wave_source=input_wave_source,
                input_wave_rx=input_wave_rx,
                input_wave_tx=input_wave_tx,
                input_wave_link_mode=input_wave_link_mode,
                input_wave_ylim=input_wave_ylim,
                input_activity_trace=input_activity_trace,
                input_activity_index=frame_index)
            if keep_frames_dir:
                frame_path = Path(keep_frames_dir) / f'{frame_index:05d}_{record.sample_name}.png'
                runtime['plt'].imsave(frame_path, frame)
            if writer is None:
                writer = _open_video_writer(output_path, fps, frame)
            else:
                writer.append(frame)
            print(f'[{frame_index + 1:04d}/{len(records):04d}] rendered {record.sample_name}')
    finally:
        if writer is not None:
            writer.close()
        if video_capture is not None:
            video_capture.release()
    return output_path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Render a presentation video from merged WiFi train/test CSI samples.')
    parser.add_argument('--video-id', default='S11_01', help='Video/sample prefix, e.g. S11_01 or S23_12.')
    parser.add_argument('--data-root', default=str(DEFAULT_DATA_ROOT))
    parser.add_argument('--splits', nargs='+', default=DEFAULT_SPLITS)
    parser.add_argument('--paper-dir', default=str(DEFAULT_PAPER_DIR))
    parser.add_argument('--experiment-log', default=str(DEFAULT_EXPERIMENT_LOG))
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--config', default=None, help='Optional model config override.')
    parser.add_argument('--checkpoint', default=None, help='Optional checkpoint override.')
    parser.add_argument('--baseline-model', default=None, help='Optional baseline model id, e.g. M0.')
    parser.add_argument('--baseline-config', default=None, help='Optional baseline config override.')
    parser.add_argument('--baseline-checkpoint', default=None, help='Optional baseline checkpoint override.')
    parser.add_argument('--output', default=None)
    parser.add_argument('--segment', choices=['longest', 'first', 'all'], default='longest')
    parser.add_argument('--start-frame', type=int, default=None)
    parser.add_argument('--end-frame', type=int, default=None)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='Prediction score threshold for display. Default 0.0 mirrors evaluation, which matches over all queries.')
    parser.add_argument('--device', default=None)
    parser.add_argument('--source-video', default=None, help='Optional original RGB video path.')
    parser.add_argument(
        '--source-video-root',
        default=None,
        help='Optional root containing per-video folders, e.g. ROOT/S52_40/output.mkv.')
    parser.add_argument(
        '--source-time-list',
        default=None,
        help='Optional time_list.txt. If omitted, the script uses SOURCE_VIDEO_DIR/time_list.txt when present.')
    parser.add_argument(
        '--source-frame-offset',
        type=int,
        default=0,
        help='Offset added to resolved source frame index. Use -1 for 1-based video labels.')
    parser.add_argument('--source-frame-dir', default=None, help='Optional directory with RGB frames.')
    parser.add_argument('--source-frame-pattern', default='{sample_name}.jpg')
    parser.add_argument('--show-unmatched', action='store_true')
    parser.add_argument('--match-quality-thr-mm', type=float, default=200.0)
    parser.add_argument('--keep-frames-dir', default=None)
    parser.add_argument(
        '--show-input-wave',
        action='store_true',
        help='Add an animated WiFi CSI input waveform panel for each rendered sample.')
    parser.add_argument(
        '--input-wave-source',
        choices=['preprocessed', 'raw-amp'],
        default='preprocessed',
        help='Waveform source. preprocessed shows the model-ready CSI tensor; raw-amp shows one raw Rx/Tx amplitude link.')
    parser.add_argument(
        '--input-wave-rx',
        type=int,
        default=0,
        help='Raw CSI receiver index for --input-wave-source raw-amp. Zero-based.')
    parser.add_argument(
        '--input-wave-tx',
        type=int,
        default=0,
        help='Raw CSI transmitter index for --input-wave-source raw-amp. Zero-based.')
    parser.add_argument(
        '--input-wave-link-mode',
        choices=['single', 'all-tx', 'all-rx'],
        default='single',
        help='For raw-amp only: show one Rx/Tx link, all transmitters for one Rx, or all receivers for one Tx.')
    parser.add_argument(
        '--input-wave-max-points',
        type=int,
        default=240,
        help='Maximum points shown in the animated waveform.')
    parser.add_argument(
        '--input-wave-smooth',
        type=int,
        default=9,
        help='Moving-average smoothing window for the displayed waveform. Use 1 to disable.')
    parser.add_argument(
        '--per-frame-bounds',
        action='store_true',
        help='Use per-frame 3D camera bounds. Default uses global bounds to reduce video jitter.')
    parser.add_argument('--dry-run', action='store_true', help='Only print selected frame records; do not render.')
    return parser


def main(argv: Optional[Sequence[str]] = None):
    args = build_arg_parser().parse_args(argv)
    records = collect_sequence_records(args.data_root, args.video_id, args.splits)
    selected = select_contiguous_segment(
        records,
        segment=args.segment,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        max_frames=args.max_frames,
        stride=args.stride)
    print(f'Found {len(records)} records for {args.video_id}. Runs:')
    for run in summarize_runs(records):
        print(
            f'  {run["start_frame"]}-{run["end_frame"]} '
            f'({run["length"]} frames, splits={",".join(run["splits"])})')
    print(f'Selected {len(selected)} frames.')
    if selected:
        print(f'  first: {selected[0].sample_name}')
        print(f'  last : {selected[-1].sample_name}')
    if args.dry_run:
        return 0

    qualitative = _load_qualitative_module()
    specs = qualitative.load_experiment_specs(args.experiment_log)
    model_asset = qualitative.resolve_model_assets(
        PROJECT_ROOT,
        args.paper_dir,
        args.model,
        specs,
        config_override=args.config,
        checkpoint_override=args.checkpoint)
    baseline_asset = None
    if args.baseline_model:
        baseline_asset = qualitative.resolve_model_assets(
            PROJECT_ROOT,
            args.paper_dir,
            args.baseline_model,
            specs,
            config_override=args.baseline_config,
            checkpoint_override=args.baseline_checkpoint)
    model_assets = build_video_model_assets(
        target_asset=model_asset,
        baseline_asset=baseline_asset)
    output_path = Path(args.output) if args.output else (
        DEFAULT_OUTPUT_DIR / f'presentation_{args.video_id}_{args.model}.mp4')
    source_video = resolve_source_video_path(
        video_id=args.video_id,
        source_video=args.source_video,
        source_video_root=args.source_video_root)
    source_time_list = resolve_time_list_path(
        video_id=args.video_id,
        source_video_path=source_video,
        source_video_root=args.source_video_root,
        time_list=args.source_time_list)
    if source_video:
        print(f'Source video: {source_video}')
    if source_time_list:
        print(f'Source time list: {source_time_list}')
    render_presentation_video(
        records=selected,
        model_asset=model_asset,
        output_path=output_path,
        data_root=args.data_root,
        splits=args.splits,
        fps=args.fps,
        score_thr=args.score_thr,
        device=args.device,
        source_video=source_video,
        source_frame_offset=args.source_frame_offset,
        source_time_list=source_time_list,
        source_frame_dir=args.source_frame_dir,
        source_frame_pattern=args.source_frame_pattern,
        show_unmatched=args.show_unmatched,
        match_quality_thr_mm=args.match_quality_thr_mm,
        keep_frames_dir=args.keep_frames_dir,
        model_assets=model_assets,
        use_global_bounds=not args.per_frame_bounds,
        show_input_wave=args.show_input_wave,
        input_wave_source=args.input_wave_source,
        input_wave_rx=args.input_wave_rx,
        input_wave_tx=args.input_wave_tx,
        input_wave_link_mode=args.input_wave_link_mode,
        input_wave_max_points=args.input_wave_max_points,
        input_wave_smooth=args.input_wave_smooth)
    print(f'Video written to: {output_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
