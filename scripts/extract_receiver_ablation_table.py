#!/usr/bin/env python3
"""Convert receiver-subset eval JSONs into a LaTeX `tabular` for rev2 §VI.E.

Mirrors baseline Table 5 (CVPR 2024 Person-in-WiFi 3D §5.6 "Number of
receivers"). Reports overall MPJPE plus the 1P/2P/3P breakdown for each
receiver subset.

Usage:
    python scripts/extract_receiver_ablation_table.py \
        --full   paper_assets/logs/rev2/M9_RF2_R1R2R3_sanity.json \
        --r1r2   paper_assets/logs/rev2/M9_RF2_R1R2_eval.json \
        --r1r3   paper_assets/logs/rev2/M9_RF2_R1R3_eval.json \
        --r2r3   paper_assets/logs/rev2/M9_RF2_R2R3_eval.json \
        --r2     paper_assets/logs/rev2/M9_RF2_R2_eval.json \
        --out    paper_assets/manuscript_latex/resfes2026_witidar/tables/receiver_ablation_table.tex
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    if not path.exists():
        sys.stderr.write(f'ERROR: file not found: {path}\n')
        sys.exit(3)
    return json.loads(path.read_text())


def row(label: str, d: dict) -> str:
    matched = d.get('matched_persons', 0)
    missed = d.get('missed_persons', 0)
    return (f'{label} & {d["mpjpe"]:.1f} & '
            f'{d.get("mpjpe_1p", 0):.1f} & '
            f'{d.get("mpjpe_2p", 0):.1f} & '
            f'{d.get("mpjpe_3p", 0):.1f} & '
            f'{matched} & {missed}')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', type=Path, required=True,
                    help='Sanity eval with all 3 receivers (= M9_RF2 baseline number)')
    ap.add_argument('--r1r2', type=Path, required=True)
    ap.add_argument('--r1r3', type=Path, required=True)
    ap.add_argument('--r2r3', type=Path, required=True)
    ap.add_argument('--r2',   type=Path, required=True)
    ap.add_argument('--out',  type=Path, required=True)
    args = ap.parse_args()

    full  = load(args.full)
    r1r2  = load(args.r1r2)
    r1r3  = load(args.r1r3)
    r2r3  = load(args.r2r3)
    r2    = load(args.r2)

    rows = [
        row(r'3 receivers (R1{+}R2{+}R3)', full),
        row(r'R1{+}R2',                    r1r2),
        row(r'R1{+}R3',                    r1r3),
        row(r'R2{+}R3',                    r2r3),
        row(r'R2 only',                    r2),
    ]
    body = '\n'.join('    ' + r + r' \\' for r in rows)

    tex = rf"""\begin{{table}}[htbp]
\caption{{Per-receiver evaluation of M9\_RF2 on the same test split as
Table~\ref{{tab:main}}. Receivers not in the subset are zero-masked
at the input of the WiTiDAR pipeline (the model is not retrained).
Mirrors baseline Table~5 of \cite{{yan2024personwifi3d}}. Units: mm; lower
is better.}}
\label{{tab:receiver-ablation}}
\centering
\begin{{tabular}}{{l c c c c c c}}
\toprule
receiver subset & MPJPE & 1P & 2P & 3P & matched & missed \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex)
    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
