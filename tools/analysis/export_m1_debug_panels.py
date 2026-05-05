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
        description='Export real M1 debug panels from the spectral WiFi adapter.')
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
            PROJECT_ROOT, 'docs', 'presentation', 'Figures', 'm1_debug_panels'),
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
        choices=['x_token', 'x_enhanced_grid'],
        default='x_token',
        help='Source tensor for the final token-map heatmap.')
    parser.add_argument(
        '--dpi',
        type=int,
        default=200,
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

    doppler = debug['doppler_profile']
    batch_size = doppler.shape[0] // num_spatial
    doppler = doppler.view(batch_size, num_spatial, -1)
    motion_energy = doppler[0].sum(dim=-1)
    return int(torch.argmax(motion_energy).item())


def choose_feature_index(debug, num_spatial, seq_len, requested, spatial_index):
    x_time = debug['x_time']
    batch_size = x_time.shape[0] // num_spatial
    x_time = x_time.view(batch_size, num_spatial, seq_len, -1)[0, spatial_index]

    if requested != 'auto':
        index = int(requested)
        if index < 0 or index >= x_time.shape[-1]:
            raise ValueError(f'feature-index must be in [0, {x_time.shape[-1] - 1}].')
        return index

    feature_dynamics = x_time.std(dim=0)
    return int(torch.argmax(feature_dynamics).item())


def make_output_path(output_dir, prefix, name):
    filename = f'{prefix}_{name}.png' if prefix else f'{name}.png'
    return os.path.join(output_dir, filename)


def save_line_plot(values, title, xlabel, ylabel, output_path, dpi, color):
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.plot(np.arange(len(values)), values, linewidth=2.0, color=color)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def robust_limits(array):
    lower = np.percentile(array, 1)
    upper = np.percentile(array, 99)
    if np.isclose(lower, upper):
        lower = float(np.min(array))
        upper = float(np.max(array))
    if np.isclose(lower, upper):
        lower -= 1.0
        upper += 1.0
    return lower, upper


def save_heatmap(image, title, xlabel, ylabel, output_path, dpi, cmap):
    vmin, vmax = robust_limits(image)
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    im = ax.imshow(
        image,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def save_combined_figure(time_values,
                         freq_values,
                         doppler_image,
                         token_image,
                         output_path,
                         dpi):
    time_x = np.arange(len(time_values))
    freq_x = np.arange(len(freq_values))
    doppler_vmin, doppler_vmax = robust_limits(doppler_image)
    token_vmin, token_vmax = robust_limits(token_image)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))

    axes[0, 0].plot(time_x, time_values, linewidth=2.0, color='#1278c8')
    axes[0, 0].set_title('Time-domain signal', fontsize=11)
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Activation')
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(freq_x, freq_values, linewidth=2.0, color='#ef6c00')
    axes[0, 1].set_title('Frequency spectrum |RFFT|', fontsize=11)
    axes[0, 1].set_xlabel('Frequency bin')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].grid(alpha=0.25)

    doppler_im = axes[1, 0].imshow(
        doppler_image,
        aspect='auto',
        origin='lower',
        cmap='magma',
        vmin=doppler_vmin,
        vmax=doppler_vmax)
    axes[1, 0].set_title('Doppler magnitude profile', fontsize=11)
    axes[1, 0].set_xlabel('Frequency bin')
    axes[1, 0].set_ylabel('Spatial group')
    fig.colorbar(doppler_im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    token_im = axes[1, 1].imshow(
        token_image,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        vmin=token_vmin,
        vmax=token_vmax)
    axes[1, 1].set_title('Motion-aware token map', fontsize=11)
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylabel('Feature dim')
    fig.colorbar(token_im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


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
            'This script expects a spectral M1-style adapter. '
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

    x_time = debug['x_time'].view(
        img.shape[0], backbone.num_spatial, backbone.seq_len, backbone.embed_dims)
    x_fft_mag = debug['x_fft_mag'].view(
        img.shape[0],
        backbone.num_spatial,
        backbone.seq_len // 2 + 1,
        backbone.embed_dims)
    doppler_profile = debug['doppler_profile'].view(
        img.shape[0], backbone.num_spatial, backbone.seq_len // 2 + 1)

    if args.token_source == 'x_token':
        token_map = debug['x_token'].view(
            img.shape[0], backbone.num_spatial, backbone.seq_len, backbone.embed_dims)
    else:
        token_map = debug['x_enhanced_grid'].view(
            img.shape[0], backbone.num_spatial, backbone.seq_len, backbone.embed_dims)

    time_values = tensor_to_numpy(x_time[0, spatial_index, :, feature_index])
    freq_values = tensor_to_numpy(x_fft_mag[0, spatial_index, :, feature_index])
    doppler_image = tensor_to_numpy(doppler_profile[0])
    token_image = tensor_to_numpy(token_map[0, spatial_index].transpose(0, 1))

    time_path = make_output_path(output_dir, args.prefix, 'time_domain_signal')
    freq_path = make_output_path(output_dir, args.prefix, 'frequency_spectrum')
    doppler_path = make_output_path(output_dir, args.prefix, 'doppler_profile')
    token_path = make_output_path(output_dir, args.prefix, 'motion_aware_token_map')
    combined_path = make_output_path(output_dir, args.prefix, 'm1_debug_panels_combined')

    save_line_plot(
        time_values,
        title='Time-domain signal',
        xlabel='Timestep',
        ylabel='Activation',
        output_path=time_path,
        dpi=args.dpi,
        color='#1278c8')
    save_line_plot(
        freq_values,
        title='Frequency spectrum |RFFT|',
        xlabel='Frequency bin',
        ylabel='Magnitude',
        output_path=freq_path,
        dpi=args.dpi,
        color='#ef6c00')
    save_heatmap(
        doppler_image,
        title='Doppler magnitude profile',
        xlabel='Frequency bin',
        ylabel='Spatial group',
        output_path=doppler_path,
        dpi=args.dpi,
        cmap='magma')
    save_heatmap(
        token_image,
        title='Motion-aware token map',
        xlabel='Timestep',
        ylabel='Feature dim',
        output_path=token_path,
        dpi=args.dpi,
        cmap='viridis')
    save_combined_figure(
        time_values,
        freq_values,
        doppler_image,
        token_image,
        output_path=combined_path,
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
        'artifacts': {
            'time_domain_signal': time_path,
            'frequency_spectrum': freq_path,
            'doppler_profile': doppler_path,
            'motion_aware_token_map': token_path,
            'combined': combined_path
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
    print(f'  Output directory: {output_dir}')


if __name__ == '__main__':
    main()
