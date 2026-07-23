#!/usr/bin/env python
"""Validate and summarize the T0--T4 tokenizer component ablation."""

import argparse
import csv
import json
from pathlib import Path


RUN_IDS = ('T0', 'T1', 'T2', 'T3', 'T4')
MODES = {
    'T0': 'linear',
    'T1': 'linear_ln',
    'T2': 'temporal_residual',
    'T3': 'spectral_gate_residual',
    'T4': 'spectral',
}
METRICS = (
    'mpjpe', 'mpjpe_1p', 'mpjpe_2p', 'mpjpe_3p',
    'fps', 'params_m', 'peak_memory_allocated_mb', 'latency_ms',
)


def _load_json(path):
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def _number(mapping, key):
    value = mapping.get(key)
    if value is None:
        return None
    return float(value)


def _row(run_id, log_root):
    evaluation_path = log_root / f'{run_id}_eval.json'
    benchmark_path = log_root / f'{run_id}_benchmark.json'
    if not evaluation_path.is_file() or not benchmark_path.is_file():
        raise FileNotFoundError(
            f'Missing evaluation or benchmark JSON for {run_id}: '
            f'{evaluation_path}, {benchmark_path}')
    evaluation = _load_json(evaluation_path)
    benchmark = _load_json(benchmark_path)
    row = {'run_id': run_id, 'mode': MODES[run_id]}
    for key in METRICS[:4]:
        row[key] = _number(evaluation, key)
    for key in METRICS[4:]:
        row[key] = _number(benchmark, key)
    row['evaluation_json'] = str(evaluation_path)
    row['benchmark_json'] = str(benchmark_path)
    return row


def summarize(rows, t0_reference, t4_reference, tolerance):
    by_id = {row['run_id']: row for row in rows}
    for run_id in RUN_IDS:
        if by_id[run_id]['mpjpe'] is None:
            raise ValueError(f'{run_id} evaluation JSON has no mpjpe value.')

    endpoint_deltas = {
        'T0': by_id['T0']['mpjpe'] - t0_reference,
        'T4': by_id['T4']['mpjpe'] - t4_reference,
    }
    endpoint_valid = all(
        abs(delta) <= tolerance for delta in endpoint_deltas.values())
    interpretation = {
        'spectral_conditioning_contributes': (
            by_id['T4']['mpjpe'] < by_id['T1']['mpjpe']
            and by_id['T4']['mpjpe'] < by_id['T2']['mpjpe']),
        'temporal_spectral_combination_is_best': (
            by_id['T4']['mpjpe'] < by_id['T2']['mpjpe']
            and by_id['T4']['mpjpe'] < by_id['T3']['mpjpe']),
        'normalization_explains_most_gain': (
            (by_id['T0']['mpjpe'] - by_id['T1']['mpjpe'])
            >= (by_id['T1']['mpjpe'] - by_id['T4']['mpjpe'])),
    }
    return {
        'protocol': {
            'epochs': 20,
            'seed': 42,
            'deterministic': True,
            'encoder_head': 'Transformer-PETR control',
            'scope': 'single-seed diagnostic ablation',
        },
        'endpoint_audit': {
            'reference_mpjpe_mm': {'T0': t0_reference, 'T4': t4_reference},
            'tolerance_mm': tolerance,
            'delta_mm': endpoint_deltas,
            'valid': endpoint_valid,
        },
        'interpretation_gates': interpretation,
        'rows': rows,
    }


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ('run_id', 'mode') + METRICS + (
        'evaluation_json', 'benchmark_json')
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-root', type=Path, required=True)
    parser.add_argument('--csv-out', type=Path, required=True)
    parser.add_argument('--json-out', type=Path, required=True)
    parser.add_argument('--t0-reference-mpjpe', type=float, default=172.540)
    parser.add_argument('--t4-reference-mpjpe', type=float, default=169.271)
    parser.add_argument('--endpoint-tolerance-mm', type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [_row(run_id, args.log_root) for run_id in RUN_IDS]
    summary = summarize(
        rows, args.t0_reference_mpjpe, args.t4_reference_mpjpe,
        args.endpoint_tolerance_mm)
    write_csv(rows, args.csv_out)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if not summary['endpoint_audit']['valid']:
        raise SystemExit(
            'Endpoint audit failed: T0 or T4 differs from the current ladder '
            'by more than the allowed tolerance. Audit provenance before use.')
    print(args.csv_out)
    print(args.json_out)


if __name__ == '__main__':
    main()
