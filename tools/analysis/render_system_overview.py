"""Render a reproducible system overview for the RESFES manuscript."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / 'paper_assets' / 'manuscript_latex' / 'resfes2026_witidar' / 'figures'
SELECTED_MODEL_LABEL = 'Mamba-2 Flow (2 steps)'
STAGE_LABELS = {
    'input': 'Preprocessed CSI\n$B\\times3\\times3\\times20\\times60$',
    'spectral': 'Spectral tokenizer\n180 tokens $\\times$ 256-D',
    'encoder': 'Flattened Mamba-2\n3 time-major blocks',
    'queries': 'Lightweight query-pose head\n100 learned queries',
    'draft': 'Draft pose $\\hat p$\n14 keypoints $\\times$ 3-D',
    'flow': 'Conditional flow refinement\n$c$: 256-D; two Euler steps',
    'output': 'Refined 3D poses\naccuracy-oriented output',
}


def _add_box(ax, x, y, width, height, text, color):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle='round,pad=0.015,rounding_size=0.035',
        linewidth=1.2, edgecolor=color, facecolor='#FFFFFF')
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha='center', va='center',
            fontsize=8.2, color='#172033', linespacing=1.25)


def _arrow(ax, start, end, label=None):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=12,
                                 linewidth=1.1, color='#526070'))
    if label:
        ax.text((start[0] + end[0]) / 2, start[1] + 0.055, label,
                ha='center', va='bottom', fontsize=7.1, color='#526070')


def render_system_overview(output_base):
    """Render the vector PDF and 300-DPI PNG system overview."""
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.8, 5.0), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # A two-row serpentine layout preserves readable labels at manuscript page scale.
    # The arrows make the top row left-to-right and the lower row right-to-left.
    top = [('input', 0.05, 0.26, '#C65D00'),
           ('spectral', 0.37, 0.26, '#E6A700'),
           ('encoder', 0.69, 0.26, '#7CB342')]
    bottom = [('queries', 0.74, 0.21, '#1F77B4'),
              ('draft', 0.50, 0.21, '#C778A8'),
              ('flow', 0.26, 0.21, '#008C6A'),
              ('output', 0.02, 0.21, '#008C6A')]
    top_y, bottom_y, height = 0.57, 0.26, 0.18
    for key, x, width, color in top:
        _add_box(ax, x, top_y, width, height, STAGE_LABELS[key], color)
    for key, x, width, color in bottom:
        _add_box(ax, x, bottom_y, width, height, STAGE_LABELS[key], color)

    for (_, x0, w0, _), (_, x1, _, _) in zip(top, top[1:]):
        _arrow(ax, (x0 + w0, top_y + height / 2), (x1 - 0.01, top_y + height / 2))
    _arrow(ax, (0.82, top_y), (0.845, bottom_y + height + 0.01))
    for (_, x0, _, _), (_, x1, w1, _) in zip(bottom, bottom[1:]):
        _arrow(ax, (x0, bottom_y + height / 2), (x1 + w1 + 0.01, bottom_y + height / 2))

    ax.text(0.5, 0.89, 'Compact draft-to-refine WiFi pose estimation',
            ha='center', va='center', fontsize=14, fontweight='bold', color='#172033')
    ax.text(0.5, 0.80, SELECTED_MODEL_LABEL,
            ha='center', va='center', fontsize=10, fontweight='semibold', color='#008C6A')
    ax.text(0.5, 0.13,
            'Training: Hungarian positives supervise draft classification, keypoints, bones, and conditional velocity matching.',
            ha='center', va='center', fontsize=8.1, color='#364152')
    ax.text(0.5, 0.05,
            'Inference: score-ranked queries are refined with $\\Delta t=0.5$ at $t\\in\\{0,0.5\\}$.',
            ha='center', va='center', fontsize=8.1, color='#364152')

    outputs = {'pdf': output_base.with_suffix('.pdf'), 'png': output_base.with_suffix('.png')}
    fig.savefig(outputs['pdf'], bbox_inches='tight', pad_inches=0.04, facecolor='white')
    fig.savefig(outputs['png'], dpi=300, bbox_inches='tight', pad_inches=0.04, facecolor='white')
    plt.close(fig)
    return outputs


def main():
    outputs = render_system_overview(FIGURE_DIR / 'fig_system_overview')
    for path in outputs.values():
        print(path)


if __name__ == '__main__':
    main()
