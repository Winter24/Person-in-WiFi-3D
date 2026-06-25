#!/usr/bin/env python
"""Generate the full-paper M0--M9/M9_RF2 ablation figures."""

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analysis.model_palette import (
    MODEL_LINESTYLES,
    MODEL_MARKERS,
    get_model_color,
)


FIGURE_DIR = ROOT / 'paper_assets' / 'manuscript_latex' / 'resfes2026_witidar' / 'figures'

VARIANTS = [
    ('M0', 172.554, 118.7, 13.130, 155.60),
    ('M1', 169.271, 83.1, 13.199, 155.86),
    ('M2', 174.961, 69.4, 11.967, 151.16),
    ('M3', 175.658, 102.0, 10.292, 142.69),
    ('M4', 169.796, 180.8, 5.255, 31.61),
    ('M5', 171.055, 116.8, 4.023, 26.80),
    ('M6', 167.573, 247.7, 1.813, 18.41),
    ('M7', 168.342, 145.8, 7.059, 38.49),
    ('M8', 169.653, 109.4, 5.827, 33.68),
    ('M9', 168.086, 206.9, 3.618, 25.29),
    ('M9_RF2', 165.487, 209.9, 3.618, 25.29),
]


def configure_matplotlib():
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.labelsize': 9.5,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7.6,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def save(fig, stem):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in [('pdf', {}), ('png', {'dpi': 300})]:
        path = FIGURE_DIR / f'{stem}.{suffix}'
        fig.savefig(path, bbox_inches='tight', pad_inches=0.04, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def draw_main_ablation():
    ids = [row[0] for row in VARIANTS]
    metrics = [
        (1, 'MPJPE (mm)'),
        (2, 'FPS'),
        (3, 'Parameters (M)'),
        (4, 'Peak memory (MB)'),
    ]
    colors = [get_model_color(row[0]) for row in VARIANTS]
    x = np.arange(len(ids))
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.1), dpi=300)
    for ax, (index, ylabel) in zip(axes.ravel(), metrics):
        values = [row[index] for row in VARIANTS]
        bars = ax.bar(x, values, color=colors, width=0.67, edgecolor='white', linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(ids, rotation=35, ha='right', fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', color='#E5E7EB', linewidth=0.75)
        ax.set_ylim(0, max(values) * 1.18)
        for model, bar, value in zip(ids, bars, values):
            if model in {'M0', 'M6', 'M9_RF2'}:
                ax.annotate(f'{value:.1f}', (bar.get_x() + bar.get_width() / 2, value),
                            xytext=(0, 2), textcoords='offset points', ha='center', fontsize=6.8)
    fig.tight_layout()
    return save(fig, 'fig_main_ablation_mpjpe_latency_params')


def draw_bubble_chart():
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=300)
    offsets = {
        'M0': (8, -11), 'M1': (-7, -8), 'M2': (-7, 7), 'M3': (5, 0),
        'M4': (5, -8), 'M5': (5, 5), 'M6': (5, 0), 'M7': (5, 5),
        'M8': (-7, 7), 'M9': (5, 7), 'M9_RF2': (5, -8),
    }
    for model, mpjpe, fps, params, _ in VARIANTS:
        color = get_model_color(model)
        selected = model == 'M9_RF2'
        ax.scatter(fps, mpjpe, s=55 + params * 34, color=color,
                   marker=MODEL_MARKERS[model], alpha=0.82,
                   edgecolor='#111827' if selected else 'white', linewidth=1.1, zorder=3)
        dx, dy = offsets[model]
        ax.annotate(model, (fps, mpjpe), xytext=(dx, dy), textcoords='offset points',
                    ha='right' if dx < 0 else 'left', va='center', fontsize=7.5,
                    fontweight='bold' if model in {'M0', 'M6', 'M9_RF2'} else 'normal')
    ax.annotate('', xy=(209.9, 165.487), xytext=(118.7, 172.554),
                arrowprops={'arrowstyle': '->', 'color': '#6B7280', 'lw': 1.2})
    ax.set_xlabel('Inference speed (FPS)')
    ax.set_ylabel('MPJPE (mm, lower is better)')
    ax.set_xlim(55, 260)
    ax.set_ylim(178, 163.5)
    ax.grid(True, color='#E5E7EB', linewidth=0.75)
    handles = [plt.Line2D([0], [0], marker=MODEL_MARKERS[row[0]], linestyle='', markersize=6,
                          markerfacecolor=get_model_color(row[0]), markeredgecolor='none', label=row[0])
               for row in VARIANTS]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.01, 0.5),
              ncol=1, frameon=False, handlelength=0.7)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    return save(fig, 'fig_accuracy_efficiency_bubble')


def normalize(values, higher_is_better):
    values = np.asarray(values, dtype=float)
    if higher_is_better:
        return 100 * (values - values.min()) / (values.max() - values.min())
    return 100 * (values.max() - values) / (values.max() - values.min())


def draw_radar_chart():
    metric_specs = [(1, 'MPJPE', False), (2, 'FPS', True), (3, 'Params', False), (4, 'Memory', False)]
    scores = {row[0]: [] for row in VARIANTS}
    labels = []
    for index, label, higher in metric_specs:
        raw = [row[index] for row in VARIANTS]
        scaled = normalize(raw, higher)
        labels.append(f'{label}\n{min(raw):.1f}-{max(raw):.1f}')
        for row, score in zip(VARIANTS, scaled):
            scores[row[0]].append(score)
    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    fig = plt.figure(figsize=(8.0, 5.8), dpi=300)
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=8.5, fontweight='bold')
    ax.tick_params(axis='x', pad=18)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], fontsize=7, color='#7A8797')
    ax.grid(color='#D6DEE8', linewidth=0.8)
    handles = []
    for row in VARIANTS:
        model = row[0]
        color = get_model_color(model)
        values = np.r_[scores[model], scores[model][0]]
        highlight = model in {'M0', 'M6', 'M9_RF2'}
        line, = ax.plot(closed_angles, values, color=color,
                        marker=MODEL_MARKERS[model], linestyle=MODEL_LINESTYLES[model],
                        linewidth=2.4 if highlight else 1.0,
                        markersize=4.5 if highlight else 2.5,
                        alpha=0.92 if highlight else 0.30, label=model)
        if highlight:
            ax.fill(closed_angles, values, color=color, alpha=0.045)
        handles.append(line)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=4, frameon=False, fontsize=7.5)
    fig.subplots_adjust(top=0.84, bottom=0.20, left=0.15, right=0.85)
    return save(fig, 'fig_multi_metric_radar')


def main():
    configure_matplotlib()
    outputs = draw_main_ablation() + draw_bubble_chart() + draw_radar_chart()
    for output in outputs:
        print(output)


if __name__ == '__main__':
    main()
