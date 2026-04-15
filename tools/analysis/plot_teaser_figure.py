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
import math
from pathlib import Path
from xml.sax.saxutils import escape

DISPLAY_NAMES = {
    'M0': 'M0',
    'M1': 'M1',
    'M2': 'M2',
    'M3': 'M3',
    'M4': 'M4',
}

DEFAULT_ALIAS_PATH = Path(__file__).with_name('experiment_id_aliases.json')

PAPER_TITLE = (
    'FlowPose-WiFi Improves the WiFi Pose Accuracy-Efficiency Frontier'
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


def _style_for_record(experiment_id, highlight, best_mpjpe='M3'):
    if experiment_id == highlight:
        return dict(color='#0f766e', edgecolor='#042f2e', alpha=0.95, linewidth=2.2, zorder=5)
    if experiment_id == best_mpjpe:
        return dict(color='#2563eb', edgecolor='#1e3a8a', alpha=0.9, linewidth=2.0, zorder=5)
    if experiment_id == 'M0':
        return dict(color='#c2410c', edgecolor='#7c2d12', alpha=0.55, linewidth=1.6, zorder=3)
    return dict(color='#94a3b8', edgecolor='#475569', alpha=0.75, linewidth=1.2, zorder=4)


def annotation_spec(experiment_id, highlight='M4', best_mpjpe='M3'):
    if experiment_id == 'M0':
        return dict(dx=-3.0, dy=-2.3, ha='right', fontweight='normal')
    if experiment_id == best_mpjpe:
        return dict(dx=3.0, dy=-3.2, ha='left', fontweight='bold')
    if experiment_id == highlight:
        return dict(dx=4.5, dy=-2.4, ha='left', fontweight='bold')
    return dict(dx=3.0, dy=-2.3, ha='left', fontweight='normal')


def add_role_callouts(ax, records, highlight='M4', best_mpjpe='M3'):
    record_map = {record['experiment_id']: record for record in records}
    callouts = {
        'M0': dict(
            text='Baseline',
            xytext=(-34, -40),
            color='#7c2d12',
            edgecolor='#7c2d12'),
        best_mpjpe: dict(
            text='Best MPJPE',
            xytext=(-86, 18),
            color='#1e3a8a',
            edgecolor='#1e3a8a'),
        highlight: dict(
            text='Best Trade-off',
            xytext=(22, 0),
            color='#042f2e',
            edgecolor='#042f2e'),
    }

    for experiment_id, spec in callouts.items():
        record = record_map.get(experiment_id)
        if record is None:
            continue
        ax.annotate(
            spec['text'],
            xy=(record['fps'], record['mpjpe']),
            xytext=spec['xytext'],
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            color=spec['color'],
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=spec['edgecolor'], alpha=0.92),
            arrowprops=dict(arrowstyle='->', color=spec['edgecolor'], lw=1.2))


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


def _fallback_radius_scale(records, min_radius=18.0, max_radius=40.0):
    params = [record['params_m'] for record in records]
    low = min(params)
    high = max(params)
    if math.isclose(low, high):
        mid = (min_radius + max_radius) / 2.0
        return {record['experiment_id']: mid for record in records}

    radii = {}
    for record in records:
        norm = (record['params_m'] - low) / (high - low)
        radii[record['experiment_id']] = min_radius + norm * (max_radius - min_radius)
    return radii


def _write_fallback_svg(records, output_prefix, xmax=200, highlight='M4', best_mpjpe='M3'):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    width = 1600
    height = 980
    plot_left = 160
    plot_right = 1500
    plot_top = 100
    plot_bottom = 820

    mpjpe_values = [record['mpjpe'] for record in records]
    ymin = min(mpjpe_values) - 6
    ymax = max(mpjpe_values) + 8
    radii = _fallback_radius_scale(records)

    def map_x(value):
        return plot_left + (value / xmax) * (plot_right - plot_left)

    def map_y(value):
        span = ymax - ymin
        return plot_bottom - ((value - ymin) / span) * (plot_bottom - plot_top)

    def tick_values(start, end, step):
        current = math.floor(start / step) * step
        ticks = []
        while current <= end:
            if current >= start - 1e-6:
                ticks.append(current)
            current += step
        return ticks

    x_ticks = [0, 50, 100, 150, 200]
    y_ticks = tick_values(ymin, ymax, 5)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    for tick in x_ticks:
        x = map_x(tick)
        parts.append(
            f'<line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_bottom}" '
            'stroke="#94a3b8" stroke-width="2" stroke-dasharray="8 8" opacity="0.25"/>')
        parts.append(
            f'<text x="{x:.2f}" y="{plot_bottom + 38}" text-anchor="middle" '
            'font-family="Arial" font-size="20" fill="#334155">'
            f'{tick}</text>')

    for tick in y_ticks:
        y = map_y(tick)
        parts.append(
            f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" '
            'stroke="#94a3b8" stroke-width="2" stroke-dasharray="8 8" opacity="0.25"/>')
        parts.append(
            f'<text x="{plot_left - 14}" y="{y + 7:.2f}" text-anchor="end" '
            'font-family="Arial" font-size="20" fill="#334155">'
            f'{tick:.0f}</text>')

    parts.append(
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_right - plot_left}" height="{plot_bottom - plot_top}" '
        'fill="none" stroke="#334155" stroke-width="3"/>')
    parts.append(
        '<text x="800" y="46" text-anchor="middle" font-family="Arial" font-size="28" '
        'font-weight="700" fill="#0f172a">'
        f'{escape(PAPER_TITLE)}</text>')
    parts.append(
        '<text x="800" y="930" text-anchor="middle" font-family="Arial" font-size="22" fill="#0f172a">'
        'Inference Speed (FPS)</text>')
    parts.append(
        '<g transform="translate(38,460) rotate(-90)"><text x="0" y="0" text-anchor="middle" '
        'font-family="Arial" font-size="22" fill="#0f172a">MPJPE (mm, lower is better)</text></g>')
    parts.append(
        '<text x="1495" y="795" text-anchor="end" font-family="Arial" font-size="18" fill="#334155">'
        'Bubble size: Parameters (M)</text>')

    for record in records:
        style = _style_for_record(record['experiment_id'], highlight, best_mpjpe=best_mpjpe)
        label = annotation_spec(record['experiment_id'], highlight=highlight, best_mpjpe=best_mpjpe)
        x = map_x(record['fps'])
        y = map_y(record['mpjpe'])
        radius = radii[record['experiment_id']]
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{style["color"]}" '
            f'fill-opacity="{style["alpha"]:.2f}" stroke="{style["edgecolor"]}" '
            f'stroke-width="{style["linewidth"]:.2f}"/>')
        parts.append(
            f'<text x="{x + label["dx"] * 4.5:.2f}" y="{y + label["dy"] * 4.5:.2f}" '
            f'text-anchor="{"end" if label["ha"] == "right" else "start"}" '
            'font-family="Arial" font-size="20" '
            f'font-weight="{"700" if label["fontweight"] == "bold" else "500"}" '
            f'fill="{style["edgecolor"]}">{escape(record["display_name"])}</text>')

    role_specs = {
        'M0': dict(text='Baseline', dx=44, dy=-28),
        best_mpjpe: dict(text='Best MPJPE', dx=-110, dy=24),
        highlight: dict(text='Best Trade-off', dx=40, dy=90),
    }
    for experiment_id, spec in role_specs.items():
        matching = next((record for record in records if record['experiment_id'] == experiment_id), None)
        if matching is None:
            continue
        style = _style_for_record(experiment_id, highlight, best_mpjpe=best_mpjpe)
        x = map_x(matching['fps'])
        y = map_y(matching['mpjpe'])
        box_x = x + spec['dx']
        box_y = y + spec['dy']
        text_width = 9.8 * len(spec['text']) + 24
        parts.append(
            f'<rect x="{box_x:.2f}" y="{box_y - 24:.2f}" width="{text_width:.2f}" height="34" rx="10" ry="10" '
            'fill="#ffffff" fill-opacity="0.92" '
            f'stroke="{style["edgecolor"]}" stroke-width="2"/>')
        parts.append(
            f'<text x="{box_x + text_width / 2:.2f}" y="{box_y:.2f}" text-anchor="middle" '
            'font-family="Arial" font-size="17" font-weight="700" '
            f'fill="{style["edgecolor"]}">{escape(spec["text"])}</text>')

    parts.append('</svg>')
    output_prefix.with_suffix('.svg').write_text('\n'.join(parts) + '\n', encoding='utf-8')

    return output_prefix.with_suffix('.svg')


def plot_teaser_figure(records, output_prefix, xmax=200, highlight='M4', best_mpjpe='M3'):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt = _load_pyplot()
    except RuntimeError:
        svg_path = _write_fallback_svg(
            records,
            output_prefix,
            xmax=xmax,
            highlight=highlight,
            best_mpjpe=best_mpjpe)
        return svg_path, None

    sizes = bubble_size_scale(records)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    for record in records:
        style = _style_for_record(record['experiment_id'], highlight, best_mpjpe=best_mpjpe)
        label = annotation_spec(record['experiment_id'], highlight=highlight, best_mpjpe=best_mpjpe)
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

    add_role_callouts(ax, records, highlight=highlight, best_mpjpe=best_mpjpe)

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
        default=['M0', 'M1', 'M2', 'M3', 'M4'],
        help='experiment IDs to plot')
    parser.add_argument(
        '--highlight',
        default='M4',
        help='experiment ID to emphasize')
    parser.add_argument(
        '--best-mpjpe',
        default='M3',
        help='experiment ID to mark as the best-MPJPE variant')
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
    figure_obj, _ = plot_teaser_figure(
        rows,
        args.out_prefix,
        xmax=args.xmax,
        highlight=args.highlight,
        best_mpjpe=args.best_mpjpe)
    if isinstance(figure_obj, Path):
        print(f'Teaser figure written to {figure_obj} (SVG fallback mode; matplotlib unavailable)')
    else:
        print(f'Teaser figure written to {args.out_prefix}.[png|pdf|svg]')


if __name__ == '__main__':
    main()
