#!/usr/bin/env python
"""Append or upsert an experiment log row into a CSV file.

Reads evaluation and benchmark JSON artifacts and merges them into a
single row keyed by ``experiment_id``.  If a row with the same
``experiment_id`` already exists in the CSV it is **updated in-place**
(upsert); otherwise a new row is appended.

Usage example
-------------
python tools/analysis/append_experiment_log.py \
    --experiment-id M0 \
    --config configs/wifi/petr_wifi.py \
    --checkpoint work_dirs/paper/M0/latest.pth \
    --eval-json paper_assets/logs/M0_eval.json \
    --benchmark-json paper_assets/logs/M0_benchmark.json \
    --csv paper_assets/logs/experiment_log.csv \
    --notes "M0 linear transformer baseline"
"""

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from collections import OrderedDict


# Canonical column order
COLUMNS = [
    'experiment_id',
    'config',
    'checkpoint',
    'mpjpe',
    'mpjpe_1p',
    'mpjpe_2p',
    'mpjpe_3p',
    'mpjpeh',
    'mpjpev',
    'mpjped',
    'count_1p',
    'count_2p',
    'count_3p',
    'latency_ms',
    'fps',
    'peak_memory_allocated_mb',
    'params_m',
    'flops_g',
    'notes',
    'created_at',
]


def _safe_get(d, *keys, default=''):
    """Retrieve a value from a nested dict by trying multiple key paths."""
    for k in keys:
        if k in d:
            v = d[k]
            return '' if v is None else v
    return default


def parse_args():
    p = argparse.ArgumentParser(
        description='Append / upsert an experiment log row to CSV')
    p.add_argument('--experiment-id', required=True,
                   help='unique experiment identifier (e.g. M0)')
    p.add_argument('--config', required=True, help='config file path')
    p.add_argument('--checkpoint', default='', help='checkpoint file path')
    p.add_argument('--eval-json', required=True,
                   help='path to evaluation metrics JSON '
                        '(output of tools/test.py --metrics-out)')
    p.add_argument('--benchmark-json', required=True,
                   help='path to benchmark results JSON '
                        '(output of tools/analysis/benchmark.py --out)')
    p.add_argument('--csv', default='paper_assets/logs/experiment_log.csv',
                   help='destination CSV file '
                        '(default: paper_assets/logs/experiment_log.csv)')
    p.add_argument('--notes', default='', help='free-text notes')
    return p.parse_args()


def build_row(args):
    """Build an OrderedDict row from CLI args + JSON files."""
    with open(args.eval_json, 'r') as f:
        ev = json.load(f)
    with open(args.benchmark_json, 'r') as f:
        bm = json.load(f)

    row = OrderedDict()
    row['experiment_id'] = args.experiment_id
    row['config'] = args.config
    row['checkpoint'] = args.checkpoint

    # Evaluation metrics
    row['mpjpe'] = _safe_get(ev, 'mpjpe')
    row['mpjpe_1p'] = _safe_get(ev, 'mpjpe_1p')
    row['mpjpe_2p'] = _safe_get(ev, 'mpjpe_2p')
    row['mpjpe_3p'] = _safe_get(ev, 'mpjpe_3p')
    row['mpjpeh'] = _safe_get(ev, 'mpjpeh')
    row['mpjpev'] = _safe_get(ev, 'mpjpev')
    row['mpjped'] = _safe_get(ev, 'mpjped')
    row['count_1p'] = _safe_get(ev, 'count_1p')
    row['count_2p'] = _safe_get(ev, 'count_2p')
    row['count_3p'] = _safe_get(ev, 'count_3p')

    # Benchmark metrics
    row['latency_ms'] = _safe_get(bm, 'latency_ms')
    row['fps'] = _safe_get(bm, 'fps')
    row['peak_memory_allocated_mb'] = _safe_get(bm, 'peak_memory_allocated_mb')
    row['params_m'] = _safe_get(bm, 'params_m')
    row['flops_g'] = _safe_get(bm, 'flops_g')

    # Meta
    row['notes'] = args.notes
    row['created_at'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    return row


def read_csv(csv_path):
    """Read existing CSV into a list of OrderedDicts.  Returns (rows, fieldnames)."""
    if not os.path.exists(csv_path):
        return [], COLUMNS
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or COLUMNS
        rows = [OrderedDict((c, r.get(c, '')) for c in fieldnames)
                for r in reader]
    return rows, fieldnames


def write_csv(csv_path, rows, fieldnames):
    """Write rows back to CSV, creating parent directories if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def upsert(rows, new_row):
    """Insert or update *new_row* in *rows* list, keyed by experiment_id.

    Returns the (possibly modified) list and a boolean indicating whether
    the row was an update (True) or a fresh insert (False).

    On update, the original ``created_at`` value is preserved so that the
    column always reflects the *first* time the experiment was logged.
    """
    eid = new_row['experiment_id']
    for idx, existing in enumerate(rows):
        if existing.get('experiment_id') == eid:
            # Preserve the original creation timestamp.
            if existing.get('created_at'):
                new_row['created_at'] = existing['created_at']
            rows[idx] = new_row
            return rows, True
    rows.append(new_row)
    return rows, False


def main():
    args = parse_args()
    row = build_row(args)
    rows, fieldnames = read_csv(args.csv)

    # Ensure all COLUMNS appear in fieldnames (merge order: existing + new)
    for c in COLUMNS:
        if c not in fieldnames:
            fieldnames.append(c)

    rows, was_update = upsert(rows, row)
    write_csv(args.csv, rows, fieldnames)

    action = 'Updated' if was_update else 'Appended'
    print(f"{action} experiment '{args.experiment_id}' in {args.csv}")


if __name__ == '__main__':
    main()
