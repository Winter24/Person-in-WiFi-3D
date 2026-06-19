#!/usr/bin/env python
"""Export paper-ready tables and figures for the full 20e ablation.

The script treats ``experiment_log.csv`` as the source of truth for
accuracy, latency, memory, and parameter numbers. It uses the per-model
``*_eval.json`` files only for diagnostic fields that are not stored in
the CSV, such as missed-person count and matched-only MPJPE.
"""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas


MAIN_IDS = ['M0', 'M1', 'M7', 'M8', 'M9', 'M9_RF2']
FLOW_IDS = [
    'M6',
    'M9_no_flow',
    'M9',
    'M9_RF2',
    'T_FW2_20e_RF2',
    'T_FW2_20e_RF4',
]

MAIN_META = {
    'M0': ('Original PETR baseline', 'Linear', 'Transformer', 'PETRHead', 'No'),
    'M1': ('Spectral PETR baseline', 'Spectral', 'Transformer', 'PETRHead', 'No'),
    'M7': ('WiTiDAR with Transformer', 'Spectral', 'Transformer', 'WiTiDAR', '1-step'),
    'M8': ('WiTiDAR with Mamba-1', 'Spectral', 'Mamba-1', 'WiTiDAR', '1-step'),
    'M9': ('Mamba2-CSI WiTiDAR', 'Spectral', 'Mamba2-CSI flattened', 'WiTiDAR', '1-step'),
    'M9_RF2': (
        'Proposed final model',
        'Spectral',
        'Mamba2-CSI flattened',
        'WiTiDAR',
        '2-step',
    ),
}

FLOW_META = {
    'M6': ('Draft Mamba2-CSI', 'No flow'),
    'M9_no_flow': ('M9 20e', 'Flow disabled'),
    'M9': ('M9 20e', '1-step RF'),
    'M9_RF2': ('M9 20e', '2-step RF'),
    'T_FW2_20e_RF2': ('Flow loss weight 2.0', '2-step RF'),
    'T_FW2_20e_RF4': ('Flow loss weight 2.0', '4-step RF'),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate paper-ready ablation tables and figures.')
    parser.add_argument(
        '--experiment-log',
        default='paper_assets/logs/full_alation_20e/experiment_log.csv',
        help='Path to experiment_log.csv.')
    parser.add_argument(
        '--section-log',
        default='paper_assets/logs/full_alation_20e/section_latency_log.csv',
        help='Path to section_latency_log.csv. Used for existence checks.')
    parser.add_argument(
        '--eval-dir',
        default='paper_assets/logs/full_alation_20e',
        help='Directory containing *_eval.json files.')
    parser.add_argument(
        '--out-dir',
        default='paper_assets/figures/full_alation_20e',
        help='Output directory for figures.')
    parser.add_argument(
        '--table-dir',
        default='paper_assets/tables/full_alation_20e',
        help='Output directory for Markdown tables.')
    return parser.parse_args()


def load_experiment_rows(csv_path):
    csv_path = Path(csv_path)
    rows = {}
    with csv_path.open('r', newline='', encoding='utf-8') as file_obj:
        for row in csv.DictReader(file_obj):
            experiment_id = row.get('experiment_id', '').strip()
            if experiment_id:
                rows[experiment_id] = row
    return rows


def load_eval_json(eval_dir, experiment_id):
    path = Path(eval_dir) / f'{experiment_id}_eval.json'
    if not path.exists():
        raise FileNotFoundError(f'Missing eval JSON for {experiment_id}: {path}')
    with path.open('r', encoding='utf-8') as file_obj:
        return json.load(file_obj)


def as_float(row, key):
    value = row.get(key, '')
    if value in ('', None, 'n/a'):
        return None
    return float(value)


def format_float(value, decimals=3):
    if value is None:
        return 'n/a'
    return f'{value:.{decimals}f}'


def format_fps(value):
    if value is None:
        return 'n/a'
    return f'{value:.1f}'


def build_rows(experiment_rows, eval_dir):
    required_ids = sorted((set(MAIN_IDS) | set(FLOW_IDS)) - {'M9_no_flow'})
    missing = [experiment_id for experiment_id in required_ids if experiment_id not in experiment_rows]
    if missing:
        raise KeyError(f'Missing required experiment IDs in experiment log: {missing}')

    rows = dict(experiment_rows)
    if 'M9_no_flow' not in rows:
        eval_row = load_eval_json(eval_dir, 'M9_no_flow')
        rows['M9_no_flow'] = {
            'experiment_id': 'M9_no_flow',
            'mpjpe': eval_row['mpjpe'],
            'mpjpe_1p': eval_row['mpjpe_1p'],
            'mpjpe_2p': eval_row['mpjpe_2p'],
            'mpjpe_3p': eval_row['mpjpe_3p'],
            'latency_ms': 'n/a',
            'fps': 'n/a',
            'params_m': 'n/a',
            'peak_memory_allocated_mb': 'n/a',
        }
    return rows


def matched_only_mpjpe(eval_row):
    matched = int(eval_row['matched_persons'])
    total = int(eval_row['total_gt_persons'])
    missed = int(eval_row['missed_persons'])
    penalty = float(eval_row.get('miss_penalty_mm', 500.0))
    if matched <= 0:
        return None
    return (float(eval_row['mpjpe']) * total - penalty * missed) / matched


def markdown_table(headers, rows):
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(str(value) for value in row) + ' |')
    return '\n'.join(lines) + '\n'


def write_main_table(rows, output_path):
    headers = [
        'ID',
        'Role',
        'Input',
        'Encoder',
        'Head',
        'Flow inference',
        'MPJPE',
        '1P',
        '2P',
        '3P',
        'Latency ms',
        'FPS',
        'Params M',
        'Memory MB',
    ]
    table_rows = []
    for experiment_id in MAIN_IDS:
        row = rows[experiment_id]
        role, input_type, encoder, head, flow = MAIN_META[experiment_id]
        table_rows.append([
            f'`{experiment_id}`',
            role,
            input_type,
            encoder,
            head,
            flow,
            format_float(as_float(row, 'mpjpe')),
            format_float(as_float(row, 'mpjpe_1p')),
            format_float(as_float(row, 'mpjpe_2p')),
            format_float(as_float(row, 'mpjpe_3p')),
            format_float(as_float(row, 'latency_ms')),
            format_fps(as_float(row, 'fps')),
            format_float(as_float(row, 'params_m')),
            format_float(as_float(row, 'peak_memory_allocated_mb'), 2),
        ])

    output_path.write_text(markdown_table(headers, table_rows), encoding='utf-8')


def write_flow_table(rows, eval_dir, output_path):
    headers = [
        'ID',
        'Checkpoint',
        'Inference',
        'MPJPE',
        'Matched-only MPJPE',
        'Missed persons',
        '1P',
        '2P',
        '3P',
        'Latency ms',
        'FPS',
    ]
    table_rows = []
    for experiment_id in FLOW_IDS:
        row = rows[experiment_id]
        eval_row = load_eval_json(eval_dir, experiment_id)
        checkpoint, inference = FLOW_META[experiment_id]
        table_rows.append([
            f'`{experiment_id}`',
            checkpoint,
            inference,
            format_float(float(eval_row['mpjpe'])),
            format_float(matched_only_mpjpe(eval_row)),
            str(eval_row['missed_persons']),
            format_float(float(eval_row['mpjpe_1p'])),
            format_float(float(eval_row['mpjpe_2p'])),
            format_float(float(eval_row['mpjpe_3p'])),
            format_float(as_float(row, 'latency_ms')),
            format_fps(as_float(row, 'fps')),
        ])

    output_path.write_text(markdown_table(headers, table_rows), encoding='utf-8')


def _bar_colors(ids):
    return ['#0f766e' if experiment_id == 'M9_RF2' else '#64748b' for experiment_id in ids]


def _load_font(size, bold=False):
    candidates = [
        'arialbd.ttf' if bold else 'arial.ttf',
        'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf',
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_centered(draw, xy, text, font, fill='#0f172a'):
    x, y = xy
    width, _ = _text_size(draw, text, font)
    draw.text((x - width / 2, y), text, font=font, fill=fill)


def _draw_rotated_label(image, center_xy, text, font, fill='#0f172a'):
    draw = ImageDraw.Draw(image)
    width, height = _text_size(draw, text, font)
    label = Image.new('RGBA', (width + 8, height + 8), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((4, 4), text, font=font, fill=fill)
    rotated = label.rotate(35, expand=True, resample=Image.BICUBIC)
    x = int(center_xy[0] - rotated.width / 2)
    y = int(center_xy[1] - rotated.height / 2)
    image.alpha_composite(rotated, (x, y))


def _draw_bar_panel(draw, image, box, title, ylabel, ids, values, decimals=1):
    left, top, right, bottom = box
    title_font = _load_font(22, bold=True)
    axis_font = _load_font(16)
    label_font = _load_font(15)
    value_font = _load_font(14)

    plot_left = left + 70
    plot_top = top + 55
    plot_right = right - 25
    plot_bottom = bottom - 95
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    max_value = max(values) * 1.12
    if max_value <= 0:
        max_value = 1.0

    _draw_centered(draw, ((left + right) / 2, top), title, title_font)
    draw.text((plot_left, plot_top - 28), ylabel, font=axis_font, fill='#0f172a')

    for tick in range(5):
        y = plot_bottom - tick * plot_height / 4
        value = max_value * tick / 4
        draw.line((plot_left, y, plot_right, y), fill='#e2e8f0', width=1)
        draw.text((plot_left - 62, y - 8), f'{value:.0f}', font=value_font, fill='#475569')

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill='#334155', width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill='#334155', width=2)

    count = len(ids)
    bar_slot = plot_width / count
    bar_width = min(44, bar_slot * 0.58)
    colors = _bar_colors(ids)

    for idx, (experiment_id, value, color) in enumerate(zip(ids, values, colors)):
        cx = plot_left + bar_slot * (idx + 0.5)
        bar_height = value / max_value * plot_height
        x0 = cx - bar_width / 2
        x1 = cx + bar_width / 2
        y0 = plot_bottom - bar_height
        draw.rectangle((x0, y0, x1, plot_bottom), fill=color, outline='#334155', width=2)
        _draw_centered(
            draw,
            (cx, y0 - 22),
            f'{value:.{decimals}f}',
            value_font,
            fill='#0f172a',
        )
        _draw_rotated_label(image, (cx, plot_bottom + 42), experiment_id, label_font)


def _write_pdf_from_png(png_path, pdf_path):
    page_width, page_height = landscape(letter)
    pdf_canvas = canvas.Canvas(str(pdf_path), pagesize=landscape(letter))
    margin = 24
    pdf_canvas.drawImage(
        str(png_path),
        margin,
        margin,
        width=page_width - 2 * margin,
        height=page_height - 2 * margin,
        preserveAspectRatio=True,
        anchor='c',
    )
    pdf_canvas.showPage()
    pdf_canvas.save()


def _save_panel_figure(title, panels, output_base, width=3600, height=1100):
    image = Image.new('RGBA', (width, height), '#ffffff')
    draw = ImageDraw.Draw(image)
    title_font = _load_font(42, bold=True)
    _draw_centered(draw, (width / 2, 35), title, title_font)

    panel_top = 130
    panel_bottom = height - 35
    gap = 55
    panel_width = (width - gap * (len(panels) + 1)) / len(panels)
    for idx, panel in enumerate(panels):
        left = gap + idx * (panel_width + gap)
        right = left + panel_width
        _draw_bar_panel(
            draw,
            image,
            (left, panel_top, right, panel_bottom),
            panel['title'],
            panel['ylabel'],
            panel['ids'],
            panel['values'],
            decimals=panel.get('decimals', 1),
        )

    png_path = Path(f'{output_base}.png')
    pdf_path = Path(f'{output_base}.pdf')
    image.convert('RGB').save(png_path, dpi=(300, 300))
    _write_pdf_from_png(png_path, pdf_path)


def plot_main_ablation(rows, output_base):
    ids = MAIN_IDS
    panels = [
        {
            'title': 'Pose accuracy',
            'ylabel': 'MPJPE (mm)',
            'ids': ids,
            'values': [as_float(rows[experiment_id], 'mpjpe') for experiment_id in ids],
            'decimals': 1,
        },
        {
            'title': 'Inference latency',
            'ylabel': 'Latency (ms)',
            'ids': ids,
            'values': [as_float(rows[experiment_id], 'latency_ms') for experiment_id in ids],
            'decimals': 1,
        },
        {
            'title': 'Model size',
            'ylabel': 'Parameters (M)',
            'ids': ids,
            'values': [as_float(rows[experiment_id], 'params_m') for experiment_id in ids],
            'decimals': 1,
        },
    ]
    _save_panel_figure(
        'Main 20e Ablation: Accuracy-Efficiency Trade-off',
        panels,
        output_base,
        width=3600,
        height=1100,
    )


def plot_flow_ablation(rows, eval_dir, output_base):
    ids = FLOW_IDS
    mpjpe_values = []
    missed_values = []
    for experiment_id in ids:
        eval_row = load_eval_json(eval_dir, experiment_id)
        mpjpe_values.append(float(eval_row['mpjpe']))
        missed_values.append(int(eval_row['missed_persons']))

    panels = [
        {
            'title': 'Pose accuracy',
            'ylabel': 'MPJPE (mm)',
            'ids': ids,
            'values': mpjpe_values,
            'decimals': 1,
        },
        {
            'title': 'Detection/matching penalty',
            'ylabel': 'Missed persons',
            'ids': ids,
            'values': missed_values,
            'decimals': 0,
        },
    ]
    _save_panel_figure(
        'Supplementary Rectified-flow Ablation',
        panels,
        output_base,
        width=2600,
        height=1100,
    )


def main():
    args = parse_args()
    experiment_log = Path(args.experiment_log)
    section_log = Path(args.section_log)
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    table_dir = Path(args.table_dir)

    if not experiment_log.exists():
        raise FileNotFoundError(f'Missing experiment log: {experiment_log}')
    if not section_log.exists():
        raise FileNotFoundError(f'Missing section latency log: {section_log}')
    if not eval_dir.exists():
        raise FileNotFoundError(f'Missing eval directory: {eval_dir}')

    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    experiment_rows = load_experiment_rows(experiment_log)
    rows = build_rows(experiment_rows, eval_dir)

    write_main_table(rows, table_dir / 'main_ablation_table.md')
    write_flow_table(rows, eval_dir, table_dir / 'supplementary_flow_table.md')
    plot_main_ablation(rows, out_dir / 'fig_main_ablation_mpjpe_latency_params')
    plot_flow_ablation(rows, eval_dir, out_dir / 'fig_flow_solver_ablation')

    print(f'Wrote tables to {table_dir}')
    print(f'Wrote figures to {out_dir}')


if __name__ == '__main__':
    main()
