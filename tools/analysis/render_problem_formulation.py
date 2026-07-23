"""Render the publication-size sensing-to-task schematic used in Section III."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = (
    ROOT / 'paper_assets' / 'manuscript_latex' /
    'resfes2026_witidar' / 'figures')

INK = '#172033'
MUTED = '#526070'
GRID = '#D7DEE7'
INPUT = '#D55E00'
QUERY = '#0072B2'
MATCHED = '#009E73'
NEUTRAL = '#8C9AA9'
WHITE = '#FFFFFF'


def _panel(ax, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.add_patch(Rectangle(
        (0.01, 0.02), 0.98, 0.96, linewidth=0.7,
        edgecolor=GRID, facecolor=WHITE, zorder=0))
    ax.text(0.04, 0.94, title, ha='left', va='top', fontsize=7.8,
            fontweight='bold', color=INK)


def _box(ax, xy, width, height, text, edge, fontsize=7.0,
         face=WHITE, linewidth=0.9, radius=0.014):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle=f'round,pad=0.008,rounding_size={radius}',
        linewidth=linewidth, edgecolor=edge, facecolor=face)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha='center', va='center', fontsize=fontsize, color=INK,
            linespacing=1.12)
    return patch


def _arrow(ax, start, end, color=MUTED, width=0.9, dashed=False,
           connectionstyle='arc3,rad=0'):
    linestyle = (0, (3, 2)) if dashed else 'solid'
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle='-|>', mutation_scale=7.5,
        linewidth=width, linestyle=linestyle, color=color,
        connectionstyle=connectionstyle, shrinkA=0, shrinkB=0))


def _person(ax, center, scale=1.0, color=INK):
    x, y = center
    linewidth = 1.05 * scale
    ax.add_patch(Circle(
        (x, y + 0.056 * scale), 0.015 * scale,
        edgecolor=color, facecolor=WHITE, linewidth=linewidth))
    segments = [
        ((x, y + 0.041 * scale), (x, y - 0.020 * scale)),
        ((x, y + 0.020 * scale), (x - 0.035 * scale, y - 0.004 * scale)),
        ((x, y + 0.020 * scale), (x + 0.035 * scale, y - 0.004 * scale)),
        ((x, y - 0.020 * scale), (x - 0.028 * scale, y - 0.080 * scale)),
        ((x, y - 0.020 * scale), (x + 0.028 * scale, y - 0.080 * scale)),
    ]
    for start, end in segments:
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color=color, linewidth=linewidth, solid_capstyle='round')


def _receiver(ax, xy, label):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), 0.15, 0.115,
        boxstyle='round,pad=0.006,rounding_size=0.012',
        linewidth=0.9, edgecolor=QUERY, facecolor=WHITE)
    ax.add_patch(patch)
    ax.text(x + 0.075, y + 0.083, label, ha='center', va='center',
            fontsize=7.1, fontweight='semibold', color=QUERY)
    for antenna_x in (x + 0.045, x + 0.075, x + 0.105):
        ax.plot([antenna_x, antenna_x], [y + 0.019, y + 0.050],
                color=QUERY, linewidth=0.8)
        ax.plot([antenna_x - 0.010, antenna_x, antenna_x + 0.010],
                [y + 0.058, y + 0.050, y + 0.058],
                color=QUERY, linewidth=0.8)


def _draw_sensing_panel(ax):
    _panel(ax, '(a) Multi-person WiFi sensing')
    ax.add_patch(Rectangle((0.055, 0.16), 0.89, 0.68,
                           linewidth=0.75, edgecolor=GRID,
                           facecolor='#FBFCFE'))
    _box(ax, (0.065, 0.445), 0.13, 0.125, '1 TX', INPUT,
         fontsize=7.4, linewidth=1.0)
    receiver_positions = [(0.78, 0.68), (0.80, 0.435), (0.74, 0.205)]
    for index, position in enumerate(receiver_positions, 1):
        _receiver(ax, position, f'RX{index}')

    _person(ax, (0.45, 0.59), scale=0.92)
    _person(ax, (0.58, 0.43), scale=0.82)
    _person(ax, (0.40, 0.30), scale=0.74)

    transmitter = (0.195, 0.507)
    receiver_inputs = [(0.78, 0.737), (0.80, 0.492), (0.74, 0.262)]
    for endpoint in receiver_inputs:
        _arrow(ax, transmitter, endpoint, color=QUERY, width=1.0)

    reflected_paths = [
        ([transmitter[0], 0.34, 0.78], [transmitter[1], 0.79, 0.737]),
        ([transmitter[0], 0.56, 0.80], [transmitter[1], 0.69, 0.492]),
        ([transmitter[0], 0.48, 0.74], [transmitter[1], 0.22, 0.262]),
    ]
    for xs, ys in reflected_paths:
        ax.plot(xs, ys, color=NEUTRAL, linewidth=0.8,
                linestyle=(0, (3, 2)))

    ax.plot([0.09, 0.18], [0.115, 0.115], color=QUERY, linewidth=1.0)
    ax.text(0.20, 0.115, 'direct', ha='left', va='center', fontsize=7.0,
            color=MUTED)
    ax.plot([0.40, 0.49], [0.115, 0.115], color=NEUTRAL,
            linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(0.51, 0.115, 'reflected', ha='left', va='center', fontsize=7.0,
            color=MUTED)
    ax.text(0.50, 0.065, '3 RX x 3 antennas', ha='center', va='center',
            fontsize=7.1, color=INK, fontweight='semibold')


def _csi_stack(ax):
    for offset in (0.020, 0.010, 0.0):
        ax.add_patch(Rectangle(
            (0.055 + offset, 0.455 + offset), 0.145, 0.165,
            linewidth=0.75, edgecolor=MUTED, facecolor=WHITE))
    for row in range(4):
        for col in range(4):
            ax.add_patch(Rectangle(
                (0.078 + col * 0.024, 0.478 + row * 0.024), 0.019, 0.019,
                linewidth=0.35, edgecolor=NEUTRAL, facecolor='#F4F7FA'))
    ax.text(0.132, 0.690, 'Complex\nCSI', ha='center', va='center',
            fontsize=7.1, fontweight='semibold', color=INK)
    ax.text(0.132, 0.397, '30 subcarriers\n20 packets',
            ha='center', va='center', fontsize=7.0, color=MUTED,
            linespacing=1.18)


def _tensor_tile(ax):
    x, y, width, height = 0.72, 0.535, 0.23, 0.18
    ax.add_patch(Rectangle((x, y), width, height, linewidth=0.9,
                           edgecolor=MATCHED, facecolor='#F2FBF7'))
    for col in range(1, 5):
        ax.plot([x + col * width / 5] * 2, [y, y + height],
                color=MATCHED, linewidth=0.45)
    for row in range(1, 3):
        ax.plot([x, x + width], [y + row * height / 3] * 2,
                color=MATCHED, linewidth=0.45)
    ax.text(x + width / 2, y + height + 0.042, '3 x 3 x 20 x 60',
            ha='center', va='center', fontsize=7.3,
            fontweight='semibold', color=MATCHED)


def _token_strip(ax):
    start_x, y = 0.565, 0.255
    for index in range(10):
        x = start_x + index * 0.037
        ax.add_patch(FancyBboxPatch(
            (x, y), 0.027, 0.055,
            boxstyle='round,pad=0.002,rounding_size=0.005',
            linewidth=0.65, edgecolor=QUERY, facecolor='#F1F7FB'))
    ax.text(0.745, 0.180, '9 streams x 20\npackets =\n180 WiFi tokens',
            ha='center', va='center', fontsize=7.1,
            color=QUERY, fontweight='semibold', linespacing=1.12)


def _draw_tensor_panel(ax):
    _panel(ax, '(b) CSI tensorization')
    _csi_stack(ax)
    _box(ax, (0.255, 0.615), 0.31, 0.145,
         'Amplitude |H|\nDWT db11', INPUT, fontsize=7.1,
         face='#FFF8F4', linewidth=0.9)
    _box(ax, (0.255, 0.350), 0.31, 0.170,
         'Phase\nunwrap ->\nrelative -> affine',
         QUERY, fontsize=7.0, face='#F4F9FC', linewidth=0.9)
    _arrow(ax, (0.20, 0.555), (0.255, 0.687), color=INPUT, width=0.8)
    _arrow(ax, (0.20, 0.520), (0.255, 0.435), color=QUERY, width=0.8)

    concat_center = (0.645, 0.520)
    ax.add_patch(Circle(concat_center, 0.038, edgecolor=INK,
                        facecolor=WHITE, linewidth=0.85))
    ax.text(*concat_center, '+', ha='center', va='center', fontsize=9.0,
            color=INK)
    _arrow(ax, (0.565, 0.687), (0.612, 0.548), color=INPUT, width=0.8)
    _arrow(ax, (0.565, 0.435), (0.612, 0.493), color=QUERY, width=0.8)
    _arrow(ax, (0.683, 0.520), (0.72, 0.610), color=MATCHED, width=0.85)

    _tensor_tile(ax)
    _arrow(ax, (0.835, 0.535), (0.835, 0.315), color=MATCHED, width=0.85)
    _token_strip(ax)
    _box(ax, (0.075, 0.075), 0.36, 0.100,
         'fixed, non-learned\npreprocessing', NEUTRAL,
         fontsize=7.0, face='#FAFBFC', linewidth=0.7)


def _query_card(ax, xy, label, edge):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), 0.225, 0.075,
        boxstyle='round,pad=0.005,rounding_size=0.012',
        linewidth=0.85, edgecolor=edge, facecolor=WHITE)
    ax.add_patch(patch)
    ax.text(x + 0.052, y + 0.0375, label, ha='center', va='center',
            fontsize=7.1, color=edge, fontweight='semibold')
    for index, alpha in enumerate((0.95, 0.55, 0.20)):
        ax.add_patch(Rectangle(
            (x + 0.115 + index * 0.030, y + 0.023), 0.023, 0.029,
            linewidth=0.4, edgecolor=edge, facecolor=edge, alpha=alpha))


def _draw_task_panel(ax):
    _panel(ax, '(c) Permutation-invariant pose set')
    ax.text(0.17, 0.825, 'Q = 100', ha='center', va='center',
            fontsize=7.2, color=QUERY, fontweight='semibold')
    ax.text(0.17, 0.780, 'scored queries', ha='center', va='center',
            fontsize=7.0, color=QUERY)
    ax.text(0.82, 0.825, 'M = 1-3', ha='center', va='center',
            fontsize=7.2, color=MATCHED, fontweight='semibold')
    ax.text(0.82, 0.782, 'target poses', ha='center', va='center',
            fontsize=7.0, color=MATCHED)
    ax.text(0.82, 0.738, 'pose: 14 x 3', ha='center', va='center',
            fontsize=7.0, color=MUTED)
    ax.text(0.51, 0.690, 'Hungarian\none-to-one', ha='center', va='center',
            fontsize=7.2, color=INK, fontweight='semibold')

    query_rows = [(0.05, 0.585, 'q1', QUERY),
                  (0.05, 0.455, 'q2', QUERY),
                  (0.05, 0.325, 'q3', QUERY),
                  (0.05, 0.175, 'q100', NEUTRAL)]
    for x, y, label, color in query_rows:
        _query_card(ax, (x, y), label, color)
    ax.text(0.155, 0.285, '...', ha='center', va='center',
            fontsize=8.0, color=MUTED)

    target_y = [0.615, 0.470, 0.325]
    for y in target_y:
        _person(ax, (0.84, y), scale=0.55, color=MATCHED)
    for query_y, pose_y in zip((0.622, 0.492, 0.362), target_y):
        _arrow(ax, (0.275, query_y), (0.79, pose_y),
               color=MATCHED, width=0.9)

    _box(ax, (0.655, 0.165), 0.22, 0.085, 'no-person', NEUTRAL,
         fontsize=7.1, face='#F7F8FA', linewidth=0.8)
    _arrow(ax, (0.275, 0.212), (0.655, 0.207), color=NEUTRAL,
           width=0.8, dashed=True)
    ax.text(0.50, 0.075,
            'Kinect targets only | inference input: CSI',
            ha='center', va='center', fontsize=7.0, color=MUTED)


def _inter_panel_arrow(fig, x_start, x_end):
    fig.add_artist(FancyArrowPatch(
        (x_start, 0.50), (x_end, 0.50), transform=fig.transFigure,
        arrowstyle='-|>', mutation_scale=8, linewidth=0.8,
        color=MUTED, clip_on=False))


def render_problem_formulation(output_base):
    """Render vector PDF and 300-DPI PNG versions of the three-panel figure."""
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    old_settings = {
        key: mpl.rcParams[key]
        for key in ('pdf.fonttype', 'ps.fonttype', 'font.family')
    }
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    try:
        fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8), dpi=300)
        _draw_sensing_panel(axes[0])
        _draw_tensor_panel(axes[1])
        _draw_task_panel(axes[2])
        fig.subplots_adjust(left=0.006, right=0.994, bottom=0.015, top=0.99,
                            wspace=0.028)
        _inter_panel_arrow(fig, 0.332, 0.342)
        _inter_panel_arrow(fig, 0.659, 0.669)

        outputs = {
            'pdf': output_base.with_suffix('.pdf'),
            'png': output_base.with_suffix('.png'),
        }
        fig.savefig(outputs['pdf'], bbox_inches='tight', pad_inches=0.018,
                    facecolor=WHITE)
        fig.savefig(outputs['png'], dpi=300, bbox_inches='tight',
                    pad_inches=0.018, facecolor=WHITE)
        plt.close(fig)
    finally:
        for key, value in old_settings.items():
            mpl.rcParams[key] = value
    return outputs


def main():
    outputs = render_problem_formulation(FIGURE_DIR / 'fig_problem_formulation')
    for path in outputs.values():
        print(path)


if __name__ == '__main__':
    main()
