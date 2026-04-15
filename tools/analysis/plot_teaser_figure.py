#!/usr/bin/env python
"""Generate the teaser bubble chart for Figure 1.

The chart compares WiFi pose models on:
  - X axis: inference speed (FPS)
  - Y axis: MPJPE in mm (lower is better, axis inverted)
  - Bubble size: parameter count in millions
"""

import argparse
import csv
import json
from pathlib import Path

DISPLAY_NAMES = {
    'M0': 'M0 (Baseline)',
    'M1': 'M1',
    'M2': 'M2',
    'M3': 'M3',
    'M4': 'M4',
    'M5': 'M5 (Ours)',
}

DEFAULT_ALIAS_PATH = Path(__file__).with_name('experiment_id_aliases.json')

PAPER_TITLE = (
    'Fast yet Accurate: Bridging the Gap in WiFi Pose Estimation '
    'via Mamba and Rectified Flow'
)


def get_display_name(experiment_id):
    return DISPLAY_NAMES.get(experiment_id, experiment_id)


def load_experiment_aliases(alias_path=None):
    alias_path = Path(alias_path) if alias_path else DEFAULT_ALIAS_PATH
    if not alias_path.exists():
        return {}
    with alias_path.open('r', encoding='utf-8') as file_obj:
        aliases = json.load(file_obj)
    return {
        str(source_id).strip(): str(target_id).strip()
        for source_id, target_id in aliases.items()
        if str(source_id).strip() and str(target_id).strip()
    }


def canonicalize_experiment_id(experiment_id, aliases=None):
    experiment_id = experiment_id.strip()
    if not experiment_id:
        return ''
    aliases = aliases or {}
    return aliases.get(experiment_id, experiment_id)


def load_experiment_rows(csv_path, runs=None, alias_path=None):
    row_map = {}
    run_order = list(runs) if runs else None
    runs = set(run_order) if run_order else None
    aliases = load_experiment_aliases(alias_path)
    with Path(csv_path).open('r', newline='', encoding='utf-8') as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            source_experiment_id = row.get('experiment_id', '').strip()
            experiment_id = canonicalize_experiment_id(source_experiment_id, aliases)
            if not experiment_id:
                continue
            if runs is not None and experiment_id not in runs:
                continue

            try:
                mpjpe = float(row['mpjpe'])
                fps = float(row['fps'])
                params_m = float(row['params_m'])
            except (KeyError, TypeError, ValueError):
                continue

            row_map[experiment_id] = {
                'experiment_id': experiment_id,
                'source_experiment_id': source_experiment_id,
                'display_name': get_display_name(experiment_id),
                'mpjpe': mpjpe,
                'fps': fps,
                'params_m': params_m,
            }

    rows = list(row_map.values())
    if run_order is not None:
        rows.sort(key=lambda row: run_order.index(row['experiment_id']))
    return rows


def bubble_size_scale(rows, min_size=500, max_size=2800):
    if not rows:
        return {}

    values = [row['params_m'] for row in rows]
    low = min(values)
    high = max(values)
    if high == low:
        mid = (min_size + max_size) / 2.0
        return {row['experiment_id']: mid for row in rows}

    size_map = {}
    for row in rows:
        norm = (row['params_m'] - low) / (high - low)
        size_map[row['experiment_id']] = min_size + norm * (max_size - min_size)
    return size_map


def _style_for_record(experiment_id, highlight):
    if experiment_id == highlight:
        return dict(color='#0f766e', edgecolor='#042f2e', alpha=0.95, linewidth=2.2, zorder=5)
    if experiment_id == 'M0':
        return dict(color='#c2410c', edgecolor='#7c2d12', alpha=0.55, linewidth=1.6, zorder=3)
    return dict(color='#94a3b8', edgecolor='#475569', alpha=0.75, linewidth=1.2, zorder=4)


def annotation_spec(experiment_id, highlight='M5'):
    if experiment_id == 'M0':
        return dict(dx=-3.0, dy=-1.4, ha='right', fontweight='normal')
    if experiment_id == highlight:
        return dict(dx=4.5, dy=-2.4, ha='left', fontweight='bold')
    return dict(dx=3.0, dy=-1.4, ha='left', fontweight='normal')


def _load_pyplot():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            'matplotlib is required to render the teaser figure. '
            'Install it in the paper environment before running this script.'
        ) from exc
    return plt


def plot_teaser_figure(records, output_prefix, xmax=200, highlight='M5'):
    plt = _load_pyplot()
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    sizes = bubble_size_scale(records)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    for record in records:
        style = _style_for_record(record['experiment_id'], highlight)
        label = annotation_spec(record['experiment_id'], highlight=highlight)
        ax.scatter(
            record['fps'],
            record['mpjpe'],
            s=sizes[record['experiment_id']],
            **style)

        ax.text(
            record['fps'] + label['dx'],
            record['mpjpe'] + label['dy'],
            record['display_name'],
            fontsize=10.5,
            fontweight=label['fontweight'],
            ha=label['ha'],
            color=style['edgecolor'])

    mpjpe_values = [record['mpjpe'] for record in records]
    ymin = min(mpjpe_values) - 6
    ymax = max(mpjpe_values) + 8

    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin, ymax)
    ax.invert_yaxis()
    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.25)
    ax.set_axisbelow(True)

    ax.set_xlabel('Inference Speed (FPS)', fontsize=12)
    ax.set_ylabel('MPJPE (mm, lower is better)', fontsize=12)
    ax.set_title(PAPER_TITLE, fontsize=13, fontweight='bold')

    ax.text(
        0.99, 0.04,
        'Bubble size: Parameters (M)',
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=10,
        color='#334155')

    fig.tight_layout()

    for suffix in ('.png', '.pdf', '.svg'):
        save_kwargs = {'bbox_inches': 'tight'}
        if suffix == '.png':
            save_kwargs['dpi'] = 400
        fig.savefig(output_prefix.with_suffix(suffix), **save_kwargs)

    return fig, ax


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Plot the teaser Figure 1 bubble chart.')
    parser.add_argument(
        '--csv',
        default='paper_assets/logs/experiment_log.csv',
        help='path to experiment log CSV')
    parser.add_argument(
        '--out-prefix',
        default='paper_assets/figures/figure1_teaser',
        help='output prefix without extension')
    parser.add_argument(
        '--alias-path',
        default=str(DEFAULT_ALIAS_PATH),
        help='JSON file that maps legacy experiment IDs to canonical IDs')
    parser.add_argument(
        '--runs',
        nargs='+',
        default=['M0', 'M1', 'M2', 'M3', 'M4', 'M5'],
        help='experiment IDs to plot')
    parser.add_argument(
        '--highlight',
        default='M5',
        help='experiment ID to emphasize')
    parser.add_argument(
        '--xmax',
        type=float,
        default=200.0,
        help='upper bound for FPS axis')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = load_experiment_rows(args.csv, runs=args.runs, alias_path=args.alias_path)
    if not rows:
        raise RuntimeError(f'No valid rows found in {args.csv} for runs {args.runs}')
    plot_teaser_figure(
        rows,
        args.out_prefix,
        xmax=args.xmax,
        highlight=args.highlight)
    print(f'Teaser figure written to {args.out_prefix}.[png|pdf|svg]')


if __name__ == '__main__':
    main()
