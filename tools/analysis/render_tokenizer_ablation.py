#!/usr/bin/env python
"""Render the T0--T4 component table from an audited JSON manifest."""

import argparse
import json
from pathlib import Path


MODE_LABELS = {
    'linear': 'Linear projection',
    'linear_ln': 'Projection + LayerNorm',
    'temporal_residual': 'Temporal residual',
    'spectral_gate_residual': 'Spectral-gated residual',
    'spectral': 'Full temporal--spectral residual',
}


def _format(value, digits):
    if value is None:
        return '--'
    return f'{float(value):.{digits}f}'


def render_component_table(manifest_path, output_path):
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not manifest.get('endpoint_audit', {}).get('valid', False):
        raise ValueError(
            'Tokenizer endpoint audit is not valid; do not generate the paper table.')

    rows = manifest.get('rows', [])
    if [row.get('run_id') for row in rows] != [f'T{i}' for i in range(5)]:
        raise ValueError('Manifest must contain ordered T0--T4 rows.')

    lines = [
        r'Table~\ref{tab:tokenizer-components} separates projection, normalization, temporal processing, and spectral gating under one shared control protocol.',
        '',
        r'\begin{table}[!t]',
        r'\centering',
        r'\caption{Single-seed diagnostic ablation of Spectral Tokenizer components under the Transformer-PETR control protocol. All variants use 20 epochs and seed 42. Lower MPJPE, latency, parameters, and peak allocated memory are better; higher FPS is better.}',
        r'\label{tab:tokenizer-components}',
        r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{l l c c c c c c c c}',
        r'\toprule',
        r'ID & Computation & MPJPE & 1P & 2P & 3P & FPS & Params (M) & Memory (MB) & Latency (ms) \\',
        r'\midrule',
    ]
    for row in rows:
        values = [
            row['run_id'],
            MODE_LABELS[row['mode']],
            _format(row.get('mpjpe'), 3),
            _format(row.get('mpjpe_1p'), 3),
            _format(row.get('mpjpe_2p'), 3),
            _format(row.get('mpjpe_3p'), 3),
            _format(row.get('fps'), 1),
            _format(row.get('params_m'), 3),
            _format(row.get('peak_memory_allocated_mb'), 2),
            _format(row.get('latency_ms'), 2),
        ]
        lines.append(' & '.join(values) + r' \\')
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}}',
        r'\end{table}',
        '',
    ])
    gates = manifest.get('interpretation_gates', {})
    if gates:
        if gates.get('spectral_conditioning_contributes'):
            lines.append(
                'T4 improves on both T1 and T2 under this diagnostic, which '
                'supports a contribution from spectral conditioning beyond '
                'LayerNorm and the ungated temporal branch.')
        else:
            lines.append(
                'T4 does not improve on both T1 and T2 under this diagnostic; '
                'the result therefore does not isolate a positive contribution '
                'from spectral conditioning.')
        if gates.get('temporal_spectral_combination_is_best'):
            lines.append(
                'It also improves on T2 and T3, supporting the combined '
                'temporal--spectral residual as the best tokenizer setting '
                'among these controls.')
        else:
            lines.append(
                'The full temporal--spectral composition is not the best of T2--T4, '
                'so no superiority claim is made for the combined branch.')
        if gates.get('normalization_explains_most_gain'):
            lines.append(
                'Because T1 accounts for at least half of the T0--T4 MPJPE change, '
                'the tokenizer is not treated as a top-level accuracy contribution.')
        lines.append('')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines), encoding='ascii')
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(render_component_table(args.manifest, args.output))
