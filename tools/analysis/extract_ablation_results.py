#!/usr/bin/env python
"""Extract the last validation row from each paper run log.

The script scans every direct subdirectory under ``work_dirs/paper``,
finds the latest ``*.log.json`` file in each run directory, extracts the
last JSON line whose ``mode`` is ``"val"``, and writes all rows to a
single text file such as::

    M1:{"mode": "val", ...}

    M2:{"mode": "val", ...}
"""

import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Extract the last validation JSON row per paper run.')
    parser.add_argument(
        '--paper-dir',
        default='work_dirs/paper',
        help='parent directory containing run subdirectories '
             '(default: work_dirs/paper)')
    parser.add_argument(
        '--output',
        default=None,
        help='output text file path '
             '(default: <paper-dir>/ablation.txt)')
    return parser.parse_args(argv)


def find_latest_log_json(run_dir):
    log_files = sorted(
        run_dir.glob('*.log.json'),
        key=lambda path: (path.stat().st_mtime, path.name))
    if not log_files:
        raise FileNotFoundError(f'No .log.json file found in {run_dir}')
    return log_files[-1]


def extract_last_val_record(log_path):
    last_val_record = None
    with log_path.open('r', encoding='utf-8') as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue

            record = json.loads(line)
            if record.get('mode') == 'val':
                last_val_record = record

    if last_val_record is None:
        raise ValueError(f'No validation row found in {log_path}')
    return last_val_record


def collect_ablation_records(paper_dir):
    records = []
    for run_dir in sorted(path for path in paper_dir.iterdir() if path.is_dir()):
        log_path = find_latest_log_json(run_dir)
        val_record = extract_last_val_record(log_path)
        records.append((run_dir.name, val_record))
    return records


def write_ablation_file(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'{run_name}:{json.dumps(record, ensure_ascii=False)}'
        for run_name, record in records
    ]
    output_path.write_text('\n\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    args = parse_args(argv)
    paper_dir = Path(args.paper_dir).resolve()
    output_path = Path(args.output).resolve() if args.output else (
        paper_dir / 'ablation.txt'
    )

    records = collect_ablation_records(paper_dir)
    write_ablation_file(records, output_path)
    print(f'Wrote {len(records)} runs to {output_path}')


if __name__ == '__main__':
    main()

