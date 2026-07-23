#!/usr/bin/env python
"""Render the standalone Spectral Tokenizer architecture figure.

The visual grammar follows the editable Draw.io system overview: orange
component boundaries, white operator boxes, a nested gate MLP, and a dashed
residual bypass. The standalone layout adds the tensor semantics that are too
small to read in the full-system figure.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / 'paper_assets' / 'manuscript_latex' /
    'resfes2026_witidar' / 'figures' / 'fig_spectral_tokenizer')

INK = '#172033'
MUTED = '#526070'
LINE = '#444444'
ORANGE = '#E69F00'
ORANGE_DARK = '#9B5E00'
ORANGE_PALE = '#FFF8E8'
ORANGE_OUTPUT = '#FFF0C7'
GREEN = '#009E73'
GREEN_PALE = '#F1FAF7'
GRAY = '#8C9AA9'
GRAY_PALE = '#FAFBFC'
WHITE = '#FFFFFF'


def _box(
        ax, x, y, width, height, text, edge=LINE, face=WHITE,
        fontsize=7.1, linewidth=1.0, weight='normal', rounding=0.012,
        zorder=3):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f'round,pad=0.008,rounding_size={rounding}',
        linewidth=linewidth, edgecolor=edge, facecolor=face, zorder=zorder)
    ax.add_patch(patch)
    ax.text(
        x + width / 2, y + height / 2, text,
        ha='center', va='center', fontsize=fontsize, color=INK,
        fontweight=weight, linespacing=1.12, zorder=zorder + 1)
    return patch


def _arrow(
        ax, start, end, color=LINE, dashed=False, rad=0.0,
        linewidth=1.05, mutation_scale=8.0, zorder=2):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle='-|>', mutation_scale=mutation_scale,
        linewidth=linewidth, color=color,
        linestyle=(0, (4, 3)) if dashed else 'solid',
        connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0,
        zorder=zorder))


def _poly_arrow(
        ax, points, color=LINE, dashed=False, linewidth=1.0,
        mutation_scale=8.0, zorder=2):
    """Draw an orthogonal routed connector with an arrow on its final leg."""
    linestyle = (0, (4, 3)) if dashed else 'solid'
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            (start[0], end[0]), (start[1], end[1]),
            color=color, linewidth=linewidth, linestyle=linestyle,
            solid_capstyle='round', zorder=zorder)
    _arrow(
        ax, points[-2], points[-1], color=color, dashed=dashed,
        linewidth=linewidth, mutation_scale=mutation_scale, zorder=zorder)


def _junction(ax, x, y, symbol, edge=ORANGE, radius=0.018):
    circle = plt.Circle(
        (x, y), radius, facecolor=WHITE, edgecolor=edge,
        linewidth=1.15, zorder=5)
    ax.add_patch(circle)
    ax.text(
        x, y, symbol, ha='center', va='center', fontsize=8.2,
        fontweight='bold', color=edge, zorder=6)


def _stage_label(ax, x, text):
    ax.text(
        x, 0.975, text, ha='center', va='top', fontsize=7.2,
        fontweight='bold', color=ORANGE_DARK)


def _gate_mlp(ax, x, y, width, height):
    _box(
        ax, x, y, width, height, '', edge=ORANGE, face=ORANGE_PALE,
        linewidth=1.1, rounding=0.014)
    ax.text(
        x + width / 2, y + height - 0.027, 'Gate MLP',
        ha='center', va='center', fontsize=7.1, fontweight='bold',
        color=INK, zorder=5)

    inner_y = y + 0.022
    inner_h = height * 0.43
    gap = width * 0.035
    inner_w = (width - 4 * gap) / 3
    x1 = x + gap
    x2 = x1 + inner_w + gap
    x3 = x2 + inner_w + gap
    _box(
        ax, x1, inner_y, inner_w, inner_h, 'Linear\n11 -> 22',
        edge=ORANGE, fontsize=6.7, linewidth=0.85, rounding=0.007,
        zorder=4)
    _box(
        ax, x2, inner_y, inner_w, inner_h, 'GELU',
        edge=ORANGE, fontsize=6.7, linewidth=0.85, rounding=0.007,
        zorder=4)
    _box(
        ax, x3, inner_y, inner_w, inner_h, 'Linear\n22 -> 20',
        edge=ORANGE, fontsize=6.7, linewidth=0.85, rounding=0.007,
        zorder=4)
    mid_y = inner_y + inner_h / 2
    _arrow(
        ax, (x1 + inner_w, mid_y), (x2, mid_y), color=ORANGE,
        linewidth=0.75, mutation_scale=6.0, zorder=5)
    _arrow(
        ax, (x2 + inner_w, mid_y), (x3, mid_y), color=ORANGE,
        linewidth=0.75, mutation_scale=6.0, zorder=5)


def render_spectral_tokenizer(output_base=DEFAULT_OUTPUT, dpi=300):
    """Write vector PDF/SVG and 300-DPI PNG versions of the architecture."""
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_base.with_suffix('.pdf')
    png_path = output_base.with_suffix('.png')
    svg_path = output_base.with_suffix('.svg')

    mpl.rcParams.update({
        'font.family': 'Arial',
        'font.size': 7.0,
        'axes.linewidth': 0.7,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })

    fig, ax = plt.subplots(figsize=(7.1, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Subtle stage labels replace an in-figure title and make the split/fusion
    # structure readable at manuscript size.
    _stage_label(ax, 0.105, 'Projected link tokens')
    _stage_label(ax, 0.470, 'Per-link temporal and spectral branches')
    _stage_label(ax, 0.905, 'Residual fusion')

    _box(
        ax, 0.012, 0.405, 0.115, 0.205,
        'Preprocessed CSI\n'
        'X: B x 3 x 3 x\n20 x 60\n'
        'regroup: B x 9 x\n20 x 60',
        edge=ORANGE, face=ORANGE_PALE, fontsize=6.8,
        linewidth=1.15)
    _box(
        ax, 0.158, 0.405, 0.125, 0.205,
        'Linear projection\n60 -> 256\n'
        'Z: B x 9 x 20 x 256',
        edge=ORANGE, face=WHITE, fontsize=6.9, linewidth=1.1)
    _arrow(ax, (0.127, 0.507), (0.158, 0.507), color=ORANGE)

    ax.text(
        0.304, 0.507, 'Z', ha='center', va='center',
        fontsize=8.2, fontweight='bold', color=ORANGE_DARK)
    _arrow(ax, (0.283, 0.507), (0.324, 0.507), color=ORANGE)

    # Temporal branch.
    _box(
        ax, 0.346, 0.646, 0.145, 0.172,
        'Depthwise Conv1D\n'
        'along 20 packets\n'
        'k = 5, groups = 256',
        edge=LINE, face=WHITE, fontsize=6.8)
    _box(
        ax, 0.516, 0.666, 0.075, 0.132,
        'GELU\n\n$\\widetilde{Z}_s$',
        edge=LINE, face=WHITE, fontsize=6.9)
    _arrow(ax, (0.324, 0.535), (0.346, 0.732), color=LINE, rad=-0.07)
    _arrow(ax, (0.491, 0.732), (0.516, 0.732), color=LINE)
    ax.text(
        0.419, 0.842, 'independent for s = 1,...,9',
        ha='center', va='center', fontsize=6.7, color=MUTED)

    # Spectral conditioning branch, computed from the projected tokens Z.
    _box(
        ax, 0.346, 0.245, 0.112, 0.165,
        '20-point RFFT$_t$(Z)\n'
        'one-sided\nmagnitude: 11 bins',
        edge=ORANGE, face=WHITE, fontsize=6.7)
    _box(
        ax, 0.480, 0.245, 0.098, 0.165,
        'Mean over\n256 channels\n\n$d_s \\in R^{11}$',
        edge=ORANGE, face=WHITE, fontsize=6.7)
    _gate_mlp(ax, 0.600, 0.220, 0.190, 0.215)
    _box(
        ax, 0.812, 0.260, 0.078, 0.135,
        'Sigmoid\n\n$g_s \\in (0,1)^{20}$',
        edge=ORANGE, face=WHITE, fontsize=6.7)
    _arrow(ax, (0.324, 0.480), (0.346, 0.328), color=ORANGE, rad=0.06)
    _arrow(ax, (0.458, 0.328), (0.480, 0.328), color=ORANGE)
    _arrow(ax, (0.578, 0.328), (0.600, 0.328), color=ORANGE)
    _arrow(ax, (0.790, 0.328), (0.812, 0.328), color=ORANGE)

    ax.text(
        0.505, 0.122,
        'one descriptor and one 20-step gate per spatial link',
        ha='center', va='center', fontsize=6.7, color=MUTED)

    # Gate the temporal branch and project the correction.
    _junction(ax, 0.645, 0.707, 'x', edge=ORANGE)
    _arrow(ax, (0.591, 0.732), (0.625, 0.712), color=LINE, rad=0.02)
    _poly_arrow(
        ax,
        [(0.851, 0.395), (0.851, 0.480), (0.645, 0.480), (0.645, 0.689)],
        color=ORANGE, linewidth=1.0)
    ax.text(
        0.672, 0.510, '$g_s[:,:,None] \\odot \\widetilde{Z}_s$',
        ha='center', va='bottom', fontsize=6.9, color=INK)

    _box(
        ax, 0.706, 0.610, 0.105, 0.145,
        'Channel projection\n$W_c$: 256 -> 256\nzero initialized',
        edge=ORANGE, face=ORANGE_PALE, fontsize=6.8, linewidth=1.1)
    _arrow(ax, (0.663, 0.707), (0.706, 0.682), color=ORANGE)

    _junction(ax, 0.844, 0.682, '+', edge=ORANGE)
    _arrow(ax, (0.811, 0.682), (0.826, 0.682), color=ORANGE)

    # The dashed bypass follows the convention used in the source Draw.io
    # overview and makes the exact initialization behavior explicit.
    _poly_arrow(
        ax,
        [(0.283, 0.545), (0.315, 0.885), (0.844, 0.885), (0.844, 0.700)],
        color=GRAY, dashed=True, linewidth=1.0)
    ax.text(
        0.610, 0.910, 'residual bypass: Z',
        ha='center', va='center', fontsize=6.7, color=MUTED)

    _box(
        ax, 0.875, 0.610, 0.105, 0.145,
        'LayerNorm\n\n$Z_{spec}=LN(Z+\\Delta Z)$',
        edge=ORANGE, face=WHITE, fontsize=6.8)
    _arrow(ax, (0.862, 0.682), (0.875, 0.682), color=ORANGE)
    _box(
        ax, 0.844, 0.790, 0.136, 0.108,
        'Flatten link/time\n'
        'B x 180 x 256 tokens',
        edge=GREEN, face=GREEN_PALE, fontsize=6.8, linewidth=1.15)
    _arrow(ax, (0.928, 0.755), (0.918, 0.790), color=GREEN)

    _box(
        ax, 0.704, 0.038, 0.276, 0.105,
        'Initialization: W_c = 0  =>  Delta Z = 0\n'
        'The spectral mode starts at LayerNorm(Z), not Z.',
        edge=GRAY, face=GRAY_PALE, fontsize=6.7,
        linewidth=0.9, rounding=0.010)

    fig.subplots_adjust(left=0.008, right=0.992, bottom=0.015, top=0.985)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.025)
    fig.savefig(svg_path, bbox_inches='tight', pad_inches=0.025)
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', pad_inches=0.025)
    plt.close(fig)

    # Matplotlib emits trailing spaces in multiline SVG path data. Removing
    # them keeps generated artifacts clean under `git diff --check`.
    svg_text = svg_path.read_text(encoding='utf-8')
    svg_path.write_text(
        '\n'.join(line.rstrip() for line in svg_text.splitlines()) + '\n',
        encoding='utf-8')
    return pdf_path, png_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-base', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    for artifact in render_spectral_tokenizer(args.output_base, args.dpi):
        print(artifact)
