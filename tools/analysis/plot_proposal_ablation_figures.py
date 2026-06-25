#!/usr/bin/env python
"""Generate proposal-aligned ablation figures for the ResFes paper."""

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analysis.model_palette import MODEL_LINESTYLES, MODEL_MARKERS, get_model_color

FIGURE_DIR = ROOT / 'paper_assets' / 'manuscript_latex' / 'resfes2026_witidar' / 'figures'

VARIANTS = [
    {
        'id': 'M0',
        'architecture': 'Person-in-WiFi 3D',
        'mpjpe': 169.34,
        'fps': 45.69,
        'params_m': 13.13,
        'memory_mb': 155.60,
    },
    {
        'id': 'M1',
        'architecture': 'M0 + Spectral Tokenizer',
        'mpjpe': 164.60,
        'fps': 49.36,
        'params_m': 13.20,
        'memory_mb': 155.86,
    },
    {
        'id': 'M2',
        'architecture': 'M1 + WiMamba',
        'mpjpe': 169.41,
        'fps': 51.48,
        'params_m': 11.97,
        'memory_mb': 151.16,
    },
    {
        'id': 'M3',
        'architecture': 'M1 + Rectified Flow',
        'mpjpe': 151.99,
        'fps': 138.79,
        'params_m': 7.06,
        'memory_mb': 38.49,
    },
    {
        'id': 'M4',
        'architecture': 'M2 + Rectified Flow',
        'mpjpe': 159.00,
        'fps': 159.85,
        'params_m': 5.83,
        'memory_mb': 27.02,
    },
]


def configure_matplotlib():
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.labelsize': 9.5,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8.2,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def save_figure(fig, stem):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_DIR / f'{stem}.pdf'
    png_path = FIGURE_DIR / f'{stem}.png'
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    return pdf_path, png_path


def draw_main_ablation():
    ids = [item['id'] for item in VARIANTS]
    x = np.arange(len(ids))
    colors = [get_model_color(model_id) for model_id in ids]
    metrics = [
        ('mpjpe', 'MPJPE (mm)'),
        ('fps', 'FPS'),
        ('params_m', 'Params (M)'),
        ('memory_mb', 'Peak memory (MB)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), dpi=300)
    fig.patch.set_facecolor('white')

    for ax, (key, ylabel) in zip(axes.ravel(), metrics):
        values = [item[key] for item in VARIANTS]
        bars = ax.bar(x, values, color=colors, width=0.62, edgecolor='white', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(ids, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', color='#E5E7EB', linewidth=0.8, alpha=0.85)
        for bar, value in zip(bars, values):
            ax.annotate(
                f'{value:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=7.4,
                color='#374151',
            )
        ymax = max(values)
        ax.set_ylim(0, ymax * 1.18)

    fig.tight_layout()
    return save_figure(fig, 'fig_main_ablation_mpjpe_latency_params')


def bubble_area(params_m):
    return 70 + params_m * 42


def draw_bubble_chart():
    fig, ax = plt.subplots(figsize=(6.7, 3.7), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for item in VARIANTS:
        is_final = item['id'] == 'M4'
        is_best = item['id'] == 'M3'
        ax.scatter(
            item['fps'],
            item['mpjpe'],
            s=bubble_area(item['params_m']),
            color=get_model_color(item['id']),
            marker=MODEL_MARKERS[item['id']],
            alpha=0.78 if not is_final else 0.90,
            edgecolor='#111827' if is_final else 'white',
            linewidth=1.2 if is_final else 0.75,
            zorder=4 if is_final or is_best else 3,
        )
        dx, dy = {
            'M0': (-2, -16),
            'M1': (4, -2),
            'M2': (12, 11),
            'M3': (4, -2),
            'M4': (4, 0),
        }[item['id']]
        ax.annotate(
            item['id'],
            xy=(item['fps'], item['mpjpe']),
            xytext=(dx, dy),
            textcoords='offset points',
            ha='right' if item['id'] == 'M0' else 'left',
            va='center',
            fontsize=8.5,
            fontweight='bold',
            color='#1F2937',
        )

    ax.annotate(
        '',
        xy=(159.85, 159.00),
        xytext=(45.69, 169.34),
        arrowprops={
            'arrowstyle': '->',
            'lw': 1.35,
            'color': '#6B7280',
            'alpha': 0.65,
            'shrinkA': 8,
            'shrinkB': 9,
        },
        zorder=2,
    )
    ax.text(103, 165.9, 'higher speed\nlower error', ha='center', va='center',
            color='#4B5563', fontsize=8.2)

    ax.set_xlabel('Inference speed (FPS)')
    ax.set_ylabel('MPJPE (mm, lower is better)')
    ax.set_xlim(35, 172)
    ax.set_ylim(172, 149)
    ax.set_xticks([40, 70, 100, 130, 160])
    ax.set_yticks([150, 155, 160, 165, 170])
    ax.grid(True, color='#E5E7EB', linewidth=0.8, alpha=0.78)
    ax.tick_params(colors='#4B5563')
    ax.xaxis.label.set_color('#374151')
    ax.yaxis.label.set_color('#374151')

    handles = [
        plt.Line2D([0], [0], marker=MODEL_MARKERS[item['id']], linestyle='',
                   markersize=7, markerfacecolor=get_model_color(item['id']),
                   markeredgecolor='none',
                   label=f"{item['id']}: {item['architecture']}")
        for item in VARIANTS
    ]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.01, 0.5),
              frameon=False, handlelength=0.8, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    return save_figure(fig, 'fig_accuracy_efficiency_bubble')


def normalize(values, higher_is_better):
    values = np.asarray(values, dtype=float)
    low = float(values.min())
    high = float(values.max())
    if high == low:
        return np.full_like(values, 50.0)
    if higher_is_better:
        return 100 * (values - low) / (high - low)
    return 100 * (high - values) / (high - low)


def draw_radar_chart():
    metrics = [
        ('mpjpe', 'MPJPE', 'mm', False),
        ('fps', 'FPS', 'FPS', True),
        ('params_m', 'Params', 'M', False),
        ('memory_mb', 'Peak Memory', 'MB', False),
    ]
    ids = [item['id'] for item in VARIANTS]
    scores = {item['id']: [] for item in VARIANTS}
    labels = []

    for key, label, unit, higher_is_better in metrics:
        raw = [item[key] for item in VARIANTS]
        scaled = normalize(raw, higher_is_better)
        labels.append(f'{label}\n{min(raw):.1f}-{max(raw):.1f} {unit}')
        for item, score in zip(VARIANTS, scaled):
            scores[item['id']].append(float(score))

    count = len(metrics)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=(8.2, 5.6), dpi=300, facecolor='white')
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_facecolor('white')
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=8.5, fontweight='bold', color='#243041')
    ax.tick_params(axis='x', pad=22)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], fontsize=8, color='#7A8797')
    ax.yaxis.grid(True, color='#D6DEE8', linewidth=0.9)
    ax.xaxis.grid(True, color='#D6DEE8', linewidth=0.9)
    ax.spines['polar'].set_color('#D6DEE8')

    handles = []
    for item in VARIANTS:
        values = np.asarray(scores[item['id']], dtype=float)
        closed_values = np.concatenate([values, values[:1]])
        highlight = item['id'] in {'M3', 'M4'}
        line, = ax.plot(
            closed_angles,
            closed_values,
            color=get_model_color(item['id']),
            linestyle=MODEL_LINESTYLES[item['id']],
            marker=MODEL_MARKERS[item['id']],
            linewidth=2.5 if highlight else 1.25,
            markersize=4.8 if highlight else 3.2,
            alpha=0.93 if highlight else 0.42,
            label=f"{item['id']}: {item['architecture']}",
        )
        if highlight:
            ax.fill(closed_angles, closed_values, color=get_model_color(item['id']), alpha=0.055)
        handles.append(line)

    ax.text(0.5, 0.5, 'normalized\nscore', transform=ax.transAxes,
            ha='center', va='center', fontsize=8.5, color='#6B7280')
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.005),
               ncol=2, frameon=False, fontsize=8.2, handlelength=1.4)
    fig.subplots_adjust(top=0.82, bottom=0.22, left=0.18, right=0.82)
    return save_figure(fig, 'fig_multi_metric_radar')


def main():
    configure_matplotlib()
    outputs = []
    outputs.extend(draw_main_ablation())
    outputs.extend(draw_bubble_chart())
    outputs.extend(draw_radar_chart())
    for output in outputs:
        print(output)


if __name__ == '__main__':
    main()
