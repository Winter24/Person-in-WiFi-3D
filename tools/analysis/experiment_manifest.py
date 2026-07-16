#!/usr/bin/env python
"""Build a machine-readable experiment manifest for paper experiments."""

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


FLOAT_FIELDS = {
    'mpjpe',
    'mpjpeh',
    'mpjpev',
    'mpjped',
    'mpjpe_1p',
    'mpjpe_2p',
    'mpjpe_3p',
    'latency_ms',
    'fps',
    'params_m',
    'peak_memory_allocated_mb',
    'flops_g',
}
INT_FIELDS = {'count_1p', 'count_2p', 'count_3p'}

SHARED_TRAINING = {
    'max_epochs': 20,
    'samples_per_gpu': 32,
    'optimizer': 'AdamW',
    'lr': 2e-5,
    'weight_decay': 1e-4,
    'seed': 42,
    'deterministic': True,
    'precision': 'FP32',
}


def _maybe_float(value):
    if value in (None, '', 'n/a'):
        return None
    return float(value)


def _maybe_int(value):
    if value in (None, '', 'n/a'):
        return None
    return int(float(value))


def _read_experiment_rows(experiment_log):
    with Path(experiment_log).open('r', newline='', encoding='utf-8') as file_obj:
        return list(csv.DictReader(file_obj))


def _find_launch_log(launch_log_dir, experiment_id):
    launch_log_dir = Path(launch_log_dir)
    if not launch_log_dir.exists():
        return None
    candidates = [
        launch_log_dir / f'{experiment_id}.log',
        launch_log_dir / f'{experiment_id}_launch.log',
        launch_log_dir / f'{experiment_id}.txt',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(launch_log_dir.glob(f'*{experiment_id}*'))
    return matches[0] if matches else None


def _extract_float(patterns, text):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _extract_int(patterns, text):
    value = _extract_float(patterns, text)
    return int(value) if value is not None else None


def parse_launch_log(path):
    if path is None or not Path(path).exists():
        return {
            'config': None,
            'samples_per_gpu': None,
            'optimizer': None,
            'lr': None,
            'weight_decay': None,
            'seed': None,
            'deterministic': None,
            'max_epochs': None,
            'grad_clip_max_norm': None,
        }
    text = Path(path).read_text(encoding='utf-8', errors='ignore')
    config_match = re.search(
        r'(?:Config|config)\s*[:=]\s*([^\s,]+\.py)', text)
    optimizer_match = re.search(
        r'optimizer\s*=\s*dict\(type=["\']?([^,"\')]+)', text)
    deterministic_match = re.search(
        r'deterministic\s*=\s*(True|False)', text, flags=re.IGNORECASE)
    return {
        'config': config_match.group(1) if config_match else None,
        'samples_per_gpu': _extract_int(
            [r'samples_per_gpu\s*[=:]\s*(\d+)'], text),
        'optimizer': optimizer_match.group(1) if optimizer_match else None,
        'lr': _extract_float([r'\blr\s*[=:]\s*([0-9.eE+-]+)'], text),
        'weight_decay': _extract_float(
            [r'weight_decay\s*[=:]\s*([0-9.eE+-]+)'], text),
        'seed': _extract_int([r'\bseed\s*[=:]\s*(\d+)'], text),
        'deterministic': (
            deterministic_match.group(1).lower() == 'true'
            if deterministic_match else None),
        'max_epochs': _extract_int(
            [r'max_epochs\s*[=:]\s*(\d+)', r'total_epochs\s*[=:]\s*(\d+)'],
            text),
        'grad_clip_max_norm': _extract_float(
            [r'max_norm\s*[=:]\s*([0-9.eE+-]+)'], text),
    }


def _metrics_from_row(row):
    metrics = {}
    for key, value in row.items():
        if key in FLOAT_FIELDS:
            metrics[key] = _maybe_float(value)
        elif key in INT_FIELDS:
            metrics[key] = _maybe_int(value)
    return metrics


def build_manifest(experiment_log,
                    launch_log_dir,
                    commit=None,
                    dataset_signature=None,
                    hardware='NVIDIA RTX 6000 Ada Generation',
                   benchmark_protocol='single-GPU inference benchmark',
                   evaluator='WifiPoseDataset.evaluate'):
    rows = _read_experiment_rows(experiment_log)
    experiments = []
    for row in rows:
        experiment_id = row['experiment_id']
        experiments.append({
            'experiment_id': experiment_id,
            'config': row.get('config'),
            'checkpoint': row.get('checkpoint'),
            'commit': commit,
            'dataset': {
                'split': 'test',
                'ordered_sample_sha256': dataset_signature,
            },
            'training': dict(SHARED_TRAINING),
            'metrics': _metrics_from_row(row),
            'evaluator': evaluator,
            'benchmark_protocol': benchmark_protocol,
            'hardware': hardware,
        })
    return {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source_experiment_log': str(experiment_log),
        'source_launch_log_dir': str(launch_log_dir),
        'experiments': experiments,
    }


def write_manifest(manifest, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding='utf-8')


def _latex_escape(value):
    text = 'n/a' if value is None else str(value)
    return (text.replace('\\', r'\textbackslash{}')
                .replace('_', r'\_')
                .replace('&', r'\&')
                .replace('%', r'\%'))


def _compact_path(value):
    if value is None:
        return None
    path = str(value).replace('\\', '/')
    for prefix in (
            'work_dirs/full_alation_20e/',
            'configs/wifi/'):
        if path.startswith(prefix):
            return path[len(prefix):]
    parts = path.split('/')
    return '/'.join(parts[-2:]) if len(parts) > 1 else path


def write_manifest_table(manifest, output_path):
    records = manifest['experiments']
    lines = [
        r'\begin{tabular}{lll}',
        r'\toprule',
        r'Model & Config & Checkpoint \\',
        r'\midrule',
    ]
    for record in records:
        lines.append(
            ' & '.join([
                _latex_escape(record['experiment_id']),
                _latex_escape(_compact_path(record['config'])),
                _latex_escape(_compact_path(record['checkpoint'])),
            ]) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}', ''])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines), encoding='utf-8')


def current_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export an experiment manifest.')
    parser.add_argument(
        '--experiment-log',
        default='paper_assets/logs/full_alation_20e/experiment_log.csv')
    parser.add_argument(
        '--launch-log-dir',
        default='paper_assets/logs/full_alation_20e/launch_logs')
    parser.add_argument(
        '--out-json',
        default='paper_assets/logs/full_alation_20e/experiment_manifest.json')
    parser.add_argument(
        '--out-tex',
        default=None,
        help='Optional LaTeX ledger path. The manuscript uses no ledger table.')
    parser.add_argument(
        '--dataset-signature',
        default='0b7c80f1190f2e4cea4f65364a5004b2beb1bcb1bd9d46940f7135d9a13280f2')
    parser.add_argument(
        '--hardware', default='NVIDIA RTX 6000 Ada Generation')
    parser.add_argument('--commit', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build_manifest(
        experiment_log=Path(args.experiment_log),
        launch_log_dir=Path(args.launch_log_dir),
        commit=args.commit or current_commit(),
        dataset_signature=args.dataset_signature,
        hardware=args.hardware,
    )
    write_manifest(manifest, args.out_json)
    print(f'Wrote manifest to {args.out_json}')
    if args.out_tex:
        write_manifest_table(manifest, args.out_tex)
        print(f'Wrote LaTeX manifest table to {args.out_tex}')
    print(f'Wrote LaTeX manifest table to {args.out_tex}')


if __name__ == '__main__':
    main()
