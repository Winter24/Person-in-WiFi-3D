import argparse
import json
import os
import sys

import matplotlib
import numpy as np
import torch


matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export computational traces from a trained spectral tokenizer.')
    parser.add_argument('config', help='Path to model config file.')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Optional checkpoint used to load trained weights before exporting panels.')
    parser.add_argument(
        '--split',
        choices=['train', 'val', 'test'],
        default='test',
        help='Dataset split to sample from.')
    parser.add_argument(
        '--sample-index',
        type=int,
        default=0,
        help='Sample index inside the chosen split.')
    parser.add_argument(
        '--device',
        default='cpu',
        help='Device for the backbone forward pass, e.g. cpu or cuda:0.')
    parser.add_argument(
        '--output-dir',
        default=os.path.join(
            PROJECT_ROOT, 'docs', 'presentation', 'Figures', 'spectral_debug_panels'),
        help='Directory where panel PNGs and metadata will be written.')
    parser.add_argument(
        '--prefix',
        default='',
        help='Optional filename prefix for all exported artifacts.')
    parser.add_argument(
        '--spatial-index',
        default='auto',
        help='Spatial group index 0..8, or "auto" to select the highest-energy group.')
    parser.add_argument(
        '--feature-index',
        default='auto',
        help='Feature dimension index, or "auto" to select the most dynamic channel.')
    parser.add_argument(
        '--token-source',
        choices=['x_token', 'residual_correction_grid'],
        default='residual_correction_grid',
        help='Source tensor for the final activation heatmap.')
    parser.add_argument(
        '--topk-features',
        type=int,
        default=64,
        help='Number of most dynamic feature channels to show in the token-map heatmap.')
    parser.add_argument(
        '--dpi',
        type=int,
        default=220,
        help='DPI used for exported PNG files.')
    return parser.parse_args()


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def import_project_runtime():
    try:
        from mmcv import Config
        from mmcv.runner import load_checkpoint
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'Missing mmcv in the current Python environment. '
            'Activate the project training/inference environment before running this script.') from exc

    try:
        from opera.datasets.wifi_pose import WifiPoseDataset
        from opera.models import build_model
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'Failed to import project modules. Run this script from the repository environment.') from exc

    return Config, load_checkpoint, WifiPoseDataset, build_model


def resolve_dataset_spec(cfg, split):
    data_cfg = cfg.data[split]
    dataset_root = data_cfg['dataset_root']
    if not os.path.isabs(dataset_root):
        dataset_root = os.path.join(PROJECT_ROOT, dataset_root)
    dataset_mode = data_cfg['mode']
    return dataset_root, dataset_mode


def load_sample(cfg, split, sample_index, dataset_cls):
    dataset_root, dataset_mode = resolve_dataset_spec(cfg, split)
    dataset = dataset_cls(dataset_root=dataset_root, pipeline=[], mode=dataset_mode)

    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            f'sample-index={sample_index} is outside dataset range [0, {len(dataset) - 1}].')

    sample = dataset[sample_index]
    return sample, dataset_root, dataset_mode


def build_backbone_from_config(cfg, checkpoint, device, build_model_fn, load_checkpoint_fn):
    model = build_model_fn(cfg.model, test_cfg=cfg.get('test_cfg'))
    if checkpoint:
        load_checkpoint_fn(model, checkpoint, map_location='cpu')
    model.to(device)
    model.eval()

    backbone = model.backbone
    backbone.to(device)
    backbone.eval()
    return backbone


def choose_spatial_index(debug, num_spatial, requested):
    if requested != 'auto':
        index = int(requested)
        if index < 0 or index >= num_spatial:
            raise ValueError(f'spatial-index must be in [0, {num_spatial - 1}].')
        return index

    descriptor = debug['spectral_descriptor']
    batch_size = descriptor.shape[0] // num_spatial
    descriptor = descriptor.view(batch_size, num_spatial, -1)
    descriptor_energy = descriptor[0].sum(dim=-1)
    return int(torch.argmax(descriptor_energy).item())


def choose_feature_index(debug, num_spatial, seq_len, requested, spatial_index):
    x_time = debug['temporal_features']
    batch_size = x_time.shape[0] // num_spatial
    x_time = x_time.view(batch_size, num_spatial, seq_len, -1)[0, spatial_index]

    if requested != 'auto':
        index = int(requested)
        if index < 0 or index >= x_time.shape[-1]:
            raise ValueError(f'feature-index must be in [0, {x_time.shape[-1] - 1}].')
        return index

    feature_dynamics = x_time.std(dim=0)
    return int(torch.argmax(feature_dynamics).item())


def choose_topk_features(token_tensor, topk_features):
    feature_dynamics = token_tensor.std(axis=0)
    order = np.argsort(feature_dynamics)[::-1]
    topk = min(topk_features, token_tensor.shape[1])
    return order[:topk]


def make_output_base(output_dir, prefix, name):
    filename = f'{prefix}_{name}' if prefix else name
    return os.path.join(output_dir, filename)


def save_figure(fig, output_base, dpi):
    paths = {
        'png': f'{output_base}.png',
        'pdf': f'{output_base}.pdf',
    }
    fig.savefig(paths['png'], dpi=dpi, bbox_inches='tight')
    fig.savefig(paths['pdf'], bbox_inches='tight')
    plt.close(fig)
    return paths


def robust_limits(array, lower_q=1, upper_q=99):
    lower = np.percentile(array, lower_q)
    upper = np.percentile(array, upper_q)
    if np.isclose(lower, upper):
        lower = float(np.min(array))
        upper = float(np.max(array))
    if np.isclose(lower, upper):
        lower -= 1.0
        upper += 1.0
    return lower, upper


def save_line_plot(values, title, xlabel, ylabel, output_base, dpi, color):
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.plot(np.arange(len(values)), values, linewidth=2.0, color=color)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, output_base, dpi)


def save_heatmap(image, title, xlabel, ylabel, output_base, dpi, cmap,
                 vmin=None, vmax=None, highlight_row=None):
    if vmin is None or vmax is None:
        vmin, vmax = robust_limits(image)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    im = ax.imshow(
        image,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax)
    if highlight_row is not None:
        ax.axhline(highlight_row, color='white', linestyle='--', linewidth=1.0, alpha=0.9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return save_figure(fig, output_base, dpi)


def save_gate_trajectories(gate_image, selected_index, output_base, dpi):
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    x = np.arange(gate_image.shape[1])
    for idx in range(gate_image.shape[0]):
        color = '#b0bec5'
        alpha = 0.35
        linewidth = 1.2
        if idx == selected_index:
            color = '#101820'
            alpha = 0.95
            linewidth = 2.6
        ax.plot(x, gate_image[idx], color=color, alpha=alpha, linewidth=linewidth)
    ax.plot(x, gate_image.mean(axis=0), color='#ff6f00', linewidth=2.0, label='mean gate')
    ax.set_ylim(0.0, 1.0)
    ax.set_title('Temporal gate trajectories', fontsize=11)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Gate weight')
    ax.grid(alpha=0.2)
    ax.legend(loc='upper right')
    fig.tight_layout()
    return save_figure(fig, output_base, dpi)


def save_diagnostic_triptych(descriptor_image,
                        gate_image,
                        correction_image,
                        sample_name,
                        spatial_index,
                        output_base,
                        dpi):
    descriptor_vmin, descriptor_vmax = robust_limits(
        descriptor_image, lower_q=2, upper_q=99)
    correction_vmin, correction_vmax = robust_limits(
        correction_image, lower_q=2, upper_q=98)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

    im0 = axes[0].imshow(
        descriptor_image,
        aspect='auto',
        origin='lower',
        cmap='magma',
        vmin=descriptor_vmin,
        vmax=descriptor_vmax)
    axes[0].axhline(spatial_index, color='white', linestyle='--', linewidth=1.0, alpha=0.9)
    axes[0].set_title('Projected-feature spectral descriptor', fontsize=12)
    axes[0].set_xlabel('Frequency bin')
    axes[0].set_ylabel('Spatial link')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        gate_image,
        aspect='auto',
        origin='lower',
        cmap='plasma',
        vmin=0.0,
        vmax=1.0)
    axes[1].axhline(spatial_index, color='white', linestyle='--', linewidth=1.0, alpha=0.9)
    axes[1].set_title('Learned per-link temporal gate', fontsize=12)
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Spatial link')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(
        correction_image,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        vmin=correction_vmin,
        vmax=correction_vmax)
    axes[2].set_title('Residual correction activation', fontsize=12)
    axes[2].set_xlabel('Timestep')
    axes[2].set_ylabel('Top dynamic feature channels')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f'Spectral tokenizer trace | sample={sample_name} | spatial link={spatial_index}',
        fontsize=13)
    fig.tight_layout()
    return save_figure(fig, output_base, dpi)


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    Config, load_checkpoint_fn, dataset_cls, build_model_fn = import_project_runtime()
    cfg = Config.fromfile(args.config)
    backbone = build_backbone_from_config(
        cfg, args.checkpoint, args.device, build_model_fn, load_checkpoint_fn)

    if getattr(backbone, 'mode', None) != 'spectral':
        raise RuntimeError(
            'This script expects a backbone using spectral tokenizer mode. '
            f'Current backbone mode is {getattr(backbone, "mode", None)!r}.')

    sample, dataset_root, dataset_mode = load_sample(
        cfg, args.split, args.sample_index, dataset_cls)
    img = sample['img'].unsqueeze(0).to(args.device)
    sample_name = str(sample.get('img_name', args.sample_index))

    with torch.no_grad():
        backbone_input = img.reshape(img.shape[0], -1, img.shape[-1])
        _, debug = backbone(backbone_input, return_debug=True)

    spatial_index = choose_spatial_index(debug, backbone.num_spatial, args.spatial_index)
    feature_index = choose_feature_index(
        debug, backbone.num_spatial, backbone.seq_len, args.feature_index, spatial_index)

    x_time = debug['temporal_features'].view(
        img.shape[0], backbone.num_spatial, backbone.seq_len, backbone.embed_dims)
    x_fft_magnitude = debug['x_fft_magnitude'].view(
        img.shape[0],
        backbone.num_spatial,
        backbone.seq_len // 2 + 1,
        backbone.embed_dims)
    spectral_descriptor = debug['spectral_descriptor'].view(
        img.shape[0], backbone.num_spatial, backbone.seq_len // 2 + 1)
    gate = debug['temporal_gate'].view(
        img.shape[0], backbone.num_spatial, backbone.seq_len)

    if args.token_source == 'x_token':
        token_map = debug['x_token'].view(
            img.shape[0], backbone.num_spatial, backbone.seq_len, backbone.embed_dims)
    else:
        token_map = debug['residual_correction_grid'].view(
            img.shape[0], backbone.num_spatial, backbone.seq_len, backbone.embed_dims)

    time_values = tensor_to_numpy(x_time[0, spatial_index, :, feature_index])
    freq_values = tensor_to_numpy(
        x_fft_magnitude[0, spatial_index, :, feature_index])
    descriptor_image = tensor_to_numpy(spectral_descriptor[0])
    gate_image = tensor_to_numpy(gate[0])

    token_block = tensor_to_numpy(token_map[0, spatial_index])
    top_feature_indices = choose_topk_features(token_block, args.topk_features)
    token_image = token_block[:, top_feature_indices].transpose(1, 0)

    time_base = make_output_base(output_dir, args.prefix, 'projected_temporal_feature')
    freq_base = make_output_base(output_dir, args.prefix, 'projected_feature_spectrum')
    descriptor_base = make_output_base(
        output_dir, args.prefix, 'projected_feature_spectral_descriptor')
    gate_base = make_output_base(output_dir, args.prefix, 'per_link_temporal_gate')
    gate_curve_base = make_output_base(
        output_dir, args.prefix, 'per_link_temporal_gate_trajectories')
    correction_base = make_output_base(
        output_dir, args.prefix, 'residual_correction_activation')
    triptych_base = make_output_base(
        output_dir, args.prefix, 'spectral_tokenizer_trace')

    time_paths = save_line_plot(
        time_values,
        title='Projected temporal feature',
        xlabel='Timestep',
        ylabel='Activation',
        output_base=time_base,
        dpi=args.dpi,
        color='#1278c8')
    frequency_paths = save_line_plot(
        freq_values,
        title='Projected-feature spectrum |RFFT|',
        xlabel='Frequency bin',
        ylabel='Magnitude',
        output_base=freq_base,
        dpi=args.dpi,
        color='#ef6c00')
    descriptor_paths = save_heatmap(
        descriptor_image,
        title='Projected-feature spectral descriptor',
        xlabel='Frequency bin',
        ylabel='Spatial link',
        output_base=descriptor_base,
        dpi=args.dpi,
        cmap='magma',
        highlight_row=spatial_index)
    gate_paths = save_heatmap(
        gate_image,
        title='Learned per-link temporal gate',
        xlabel='Timestep',
        ylabel='Spatial link',
        output_base=gate_base,
        dpi=args.dpi,
        cmap='plasma',
        vmin=0.0,
        vmax=1.0,
        highlight_row=spatial_index)
    gate_curve_paths = save_gate_trajectories(
        gate_image,
        selected_index=spatial_index,
        output_base=gate_curve_base,
        dpi=args.dpi)
    correction_paths = save_heatmap(
        token_image,
        title='Residual correction activation',
        xlabel='Timestep',
        ylabel='Top dynamic feature channels',
        output_base=correction_base,
        dpi=args.dpi,
        cmap='viridis')
    triptych_paths = save_diagnostic_triptych(
        descriptor_image,
        gate_image,
        token_image,
        sample_name=sample_name,
        spatial_index=spatial_index,
        output_base=triptych_base,
        dpi=args.dpi)

    metadata = {
        'config': os.path.abspath(args.config),
        'checkpoint': os.path.abspath(args.checkpoint) if args.checkpoint else None,
        'split': args.split,
        'dataset_root': os.path.abspath(dataset_root),
        'dataset_mode': dataset_mode,
        'sample_index': args.sample_index,
        'sample_name': sample_name,
        'device': args.device,
        'spatial_index': spatial_index,
        'feature_index': feature_index,
        'token_source': args.token_source,
        'top_feature_indices': top_feature_indices.tolist(),
        'artifacts': {
            'projected_temporal_feature': time_paths,
            'projected_feature_spectrum': frequency_paths,
            'projected_feature_spectral_descriptor': descriptor_paths,
            'per_link_temporal_gate': gate_paths,
            'per_link_temporal_gate_trajectories': gate_curve_paths,
            'residual_correction_activation': correction_paths,
            'spectral_tokenizer_trace': triptych_paths,
        },
        'tensor_shapes': {
            key: list(value.shape)
            for key, value in debug.items()
        }
    }

    metadata_path = os.path.join(
        output_dir,
        f'{args.prefix}_debug_metadata.json' if args.prefix else 'debug_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print('Export completed:')
    print(f'  Sample: {sample_name}')
    print(f'  Spatial index: {spatial_index}')
    print(f'  Feature index: {feature_index}')
    print(f'  Token source: {args.token_source}')
    print(f'  Output directory: {output_dir}')


if __name__ == '__main__':
    main()
