#!/usr/bin/env python
"""Run postprocess for Colab paper runs stored on Google Drive."""

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAPER_DIR = '/content/drive/MyDrive/RESFES2026/Test'
DEFAULT_OUTPUT_DIR = DEFAULT_PAPER_DIR
DEFAULT_EVAL_WORK_ROOT = REPO_ROOT / 'work_dirs' / 'paper_eval_colab'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Postprocess Colab paper work dirs using folder names as experiment ids.')
    parser.add_argument(
        '--paper-dir',
        type=Path,
        default=Path(DEFAULT_PAPER_DIR),
        help=f'Root folder containing one subdirectory per model run. Default: {DEFAULT_PAPER_DIR}')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f'Directory for eval/benchmark JSON and experiment_log.csv. Default: {DEFAULT_OUTPUT_DIR}')
    parser.add_argument(
        '--eval-work-root',
        type=Path,
        default=DEFAULT_EVAL_WORK_ROOT,
        help=f'Working directory root for tools/test.py outputs. Default: {DEFAULT_EVAL_WORK_ROOT}')
    parser.add_argument('--benchmark-device', default='cuda:0')
    parser.add_argument('--benchmark-times', default='100')
    parser.add_argument('--benchmark-warmup', default='10')
    return parser.parse_args(argv)


def discover_run_dirs(paper_dir):
    return sorted(
        [path for path in paper_dir.iterdir() if path.is_dir()],
        key=lambda path: path.name)


def resolve_config(run_dir):
    config_paths = sorted(run_dir.glob('*.py'))
    if not config_paths:
        raise FileNotFoundError(f'No config dump found in {run_dir}')
    return config_paths[0]


def resolve_checkpoint(run_dir):
    latest_path = run_dir / 'latest.pth'
    if latest_path.exists():
        return latest_path

    epoch_paths = sorted(run_dir.glob('epoch_*.pth'))
    if not epoch_paths:
        raise FileNotFoundError(
            f'No checkpoint found in {run_dir} (expected latest.pth or epoch_*.pth)')
    return epoch_paths[-1]


def run_command(command):
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def run_one(args, run_dir):
    experiment_id = run_dir.name
    config_path = resolve_config(run_dir)
    checkpoint_path = resolve_checkpoint(run_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_work_dir = args.eval_work_root / experiment_id
    eval_work_dir.mkdir(parents=True, exist_ok=True)

    eval_json = args.output_dir / f'{experiment_id}_eval.json'
    benchmark_json = args.output_dir / f'{experiment_id}_benchmark.json'
    csv_path = args.output_dir / 'experiment_log.csv'

    run_command([
        'python',
        'tools/test.py',
        str(config_path),
        str(checkpoint_path),
        '--eval',
        'mpjpe',
        '--work-dir',
        str(eval_work_dir),
        '--metrics-out',
        str(eval_json),
    ])

    run_command([
        'python',
        'tools/analysis/benchmark.py',
        str(config_path),
        '--checkpoint',
        str(checkpoint_path),
        '--device',
        args.benchmark_device,
        '--times',
        str(args.benchmark_times),
        '--warmup',
        str(args.benchmark_warmup),
        '--out',
        str(benchmark_json),
    ])

    run_command([
        'python',
        'tools/analysis/append_experiment_log.py',
        '--experiment-id',
        experiment_id,
        '--config',
        str(config_path),
        '--checkpoint',
        str(checkpoint_path),
        '--eval-json',
        str(eval_json),
        '--benchmark-json',
        str(benchmark_json),
        '--csv',
        str(csv_path),
        '--notes',
        '',
    ])


def main(argv=None):
    args = parse_args(argv)
    paper_dir = args.paper_dir

    if not paper_dir.exists():
        raise FileNotFoundError(f'Paper directory does not exist: {paper_dir}')

    run_dirs = discover_run_dirs(paper_dir)
    if not run_dirs:
        raise FileNotFoundError(f'No run folders found under: {paper_dir}')

    for run_dir in run_dirs:
        print(f'Processing {run_dir.name} from {run_dir}')
        run_one(args, run_dir)

    print(f'Done. Updated {args.output_dir / "experiment_log.csv"}')


if __name__ == '__main__':
    main()
