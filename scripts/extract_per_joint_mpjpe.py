#!/usr/bin/env python3
"""Convert per-joint eval JSON(s) into a LaTeX `tabular` for rev2 patch P5.

Expects JSON produced by `tools/test.py --metrics-out` (after the per-axis
patch on 2026-06-19) with keys:
  - per_joint_mpjpe:  {joint_name: float_mm}
  - per_joint_mpjdle: {joint_name: {'h': float, 'v': float, 'd': float}}

Joint order in the output table follows BASELINE Table 2 (CVPR 2024
Person-in-WiFi 3D §5.3) so reviewers can compare row-by-row.

Usage:
    python scripts/extract_per_joint_mpjpe.py \
        --eval-json paper_assets/logs/rev2/M9_RF2_eval.json \
        --compare-json paper_assets/logs/rev2/M0_eval.json \
        --out paper_assets/manuscript_latex/resfes2026_witidar/tables/per_joint_table.tex

Exit codes:
    0  success
    2  required JSON key missing (per_joint_mpjpe / per_joint_mpjdle)
    3  input file missing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Display order roughly matches baseline Table 2.
JOINT_DISPLAY_ORDER = [
    'Neck', 'Head',
    'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow',
    'L_Wrist', 'R_Wrist',
    'L_Hip', 'R_Hip',
    'L_Knee', 'R_Knee',
    'L_Ankle', 'R_Ankle',
]

# Render names for the LaTeX table.
JOINT_LATEX = {
    'Neck': 'neck',
    'Head': 'head',
    'L_Shoulder': 'left shoulder',
    'R_Shoulder': 'right shoulder',
    'L_Elbow': 'left elbow',
    'R_Elbow': 'right elbow',
    'L_Wrist': 'left wrist',
    'R_Wrist': 'right wrist',
    'L_Hip': 'left hip',
    'R_Hip': 'right hip',
    'L_Knee': 'left knee',
    'R_Knee': 'right knee',
    'L_Ankle': 'left ankle',
    'R_Ankle': 'right ankle',
}


def load_eval(path: Path) -> dict:
    if not path.exists():
        sys.stderr.write(f'ERROR: file not found: {path}\n')
        sys.exit(3)
    with path.open() as f:
        d = json.load(f)
    for key in ('per_joint_mpjpe', 'per_joint_mpjdle'):
        if key not in d:
            sys.stderr.write(
                f'ERROR: {path} missing key "{key}". '
                f'Re-run eval after applying the 2026-06-19 wifi_pose.py patch.\n')
            sys.exit(2)
    return d


def fmt_row_solo(joint: str, e: dict) -> str:
    pj = e['per_joint_mpjpe'][joint]
    pd = e['per_joint_mpjdle'][joint]
    return (f"{JOINT_LATEX[joint]} & {pj:.1f} & "
            f"{pd['h']:.1f} & {pd['v']:.1f} & {pd['d']:.1f}")


def fmt_row_compare(joint: str, e1: dict, e2: dict) -> str:
    pj1 = e1['per_joint_mpjpe'][joint]
    pj2 = e2['per_joint_mpjpe'][joint]
    return f"{JOINT_LATEX[joint]} & {pj1:.1f} & {pj2:.1f}"


def render(eval_d: dict, compare_d: dict | None) -> str:
    if compare_d is None:
        # Single-model: PJPE + PJDLE(h/v/d), mirrors baseline Table 2
        rows = []
        for j in JOINT_DISPLAY_ORDER:
            rows.append('    ' + fmt_row_solo(j, eval_d) + r' \\')
        # Mean row
        names = list(eval_d['per_joint_mpjpe'].keys())
        mean_pj = sum(eval_d['per_joint_mpjpe'].values()) / len(names)
        mean_h = sum(v['h'] for v in eval_d['per_joint_mpjdle'].values()) / len(names)
        mean_v = sum(v['v'] for v in eval_d['per_joint_mpjdle'].values()) / len(names)
        mean_d = sum(v['d'] for v in eval_d['per_joint_mpjdle'].values()) / len(names)
        rows.append(r'    \midrule')
        rows.append(f'    \\textbf{{Mean}} & \\textbf{{{mean_pj:.1f}}} & '
                    f'\\textbf{{{mean_h:.1f}}} & \\textbf{{{mean_v:.1f}}} & '
                    f'\\textbf{{{mean_d:.1f}}} \\\\')

        body = '\n'.join(rows)
        return rf"""\begin{{table}}[htbp]
\caption{{Per-joint mean per-joint position error (MPJPE) and per-axis
projected dimension localization error (PJDLE) for the selected M9\_RF2
model, evaluated on the same test split used in Table~\ref{{tab:main}}.
Mirrors baseline Table~2 of \cite{{yan2024personwifi3d}} for direct
comparison. Units: millimeters.}}
\label{{tab:per-joint}}
\centering
\begin{{tabular}}{{l c c c c}}
\toprule
joint & MPJPE & PJDLE(h) & PJDLE(v) & PJDLE(d) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    else:
        rows = []
        for j in JOINT_DISPLAY_ORDER:
            rows.append('    ' + fmt_row_compare(j, eval_d, compare_d) + r' \\')
        mean_e = sum(eval_d['per_joint_mpjpe'].values()) / 14
        mean_c = sum(compare_d['per_joint_mpjpe'].values()) / 14
        rows.append(r'    \midrule')
        rows.append(f'    \\textbf{{Mean}} & \\textbf{{{mean_e:.1f}}} & '
                    f'\\textbf{{{mean_c:.1f}}} \\\\')
        body = '\n'.join(rows)
        return rf"""\begin{{table}}[htbp]
\caption{{Per-joint MPJPE for the selected M9\_RF2 model versus the original
PETR-style M0 baseline, evaluated on the same test split as
Table~\ref{{tab:main}}. Lower is better. Units: millimeters.}}
\label{{tab:per-joint}}
\centering
\begin{{tabular}}{{l c c}}
\toprule
joint & M9\_RF2 (ours) & M0 (PETR) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-json', type=Path, required=True,
                    help='Eval JSON for the selected model (e.g. M9_RF2_eval.json)')
    ap.add_argument('--compare-json', type=Path, default=None,
                    help='Optional second JSON to render side-by-side (e.g. M0_eval.json)')
    ap.add_argument('--out', type=Path, required=True,
                    help='Output .tex path')
    args = ap.parse_args()

    eval_d = load_eval(args.eval_json)
    compare_d = load_eval(args.compare_json) if args.compare_json else None

    out = render(eval_d, compare_d)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out)
    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
