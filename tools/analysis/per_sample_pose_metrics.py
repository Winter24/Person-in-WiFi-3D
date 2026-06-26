#!/usr/bin/env python
"""Per-sample pose metric export and paired inference-only analysis.

This module is intentionally independent from MMCV/MMDetection so the metric
decomposition and bootstrap logic can be unit-tested on a CPU-only machine.
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    linear_sum_assignment = None


def _as_array(value):
    return np.asarray(value, dtype=np.float32)


def _match_persons(gt_keypoints, pred_keypoints, match_threshold_mm):
    gt = _as_array(gt_keypoints)
    pred = _as_array(pred_keypoints)
    n_gt = int(gt.shape[0])
    n_pred = int(pred.shape[0])
    if n_gt == 0 or n_pred == 0:
        return [], np.zeros((n_gt, n_pred), dtype=np.float32)

    diff = gt[:, None, :, :] - pred[None, :, :, :]
    cost_matrix = np.linalg.norm(diff, axis=-1).mean(axis=-1) * 1000.0
    if linear_sum_assignment is None:
        remaining_gt = set(range(n_gt))
        remaining_pred = set(range(n_pred))
        pairs = []
        while remaining_gt and remaining_pred:
            best = min(
                ((cost_matrix[g, p], g, p)
                 for g in remaining_gt for p in remaining_pred),
                key=lambda item: item[0])
            _, gt_idx, pred_idx = best
            pairs.append((gt_idx, pred_idx))
            remaining_gt.remove(gt_idx)
            remaining_pred.remove(pred_idx)
    else:
        gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
        pairs = list(zip(gt_indices.tolist(), pred_indices.tolist()))

    valid_pairs = [
        (int(gt_idx), int(pred_idx), float(cost_matrix[gt_idx, pred_idx]))
        for gt_idx, pred_idx in pairs
        if float(cost_matrix[gt_idx, pred_idx]) <= match_threshold_mm
    ]
    return valid_pairs, cost_matrix


def build_sample_record(sample_index,
                        sample_id,
                        gt_keypoints,
                        pred_keypoints,
                        confidences=None,
                        match_threshold_mm=500.0,
                        miss_penalty_mm=500.0):
    """Build a JSON-safe per-sample metric record."""
    gt = _as_array(gt_keypoints)
    pred = _as_array(pred_keypoints)
    confidences = [] if confidences is None else [float(x) for x in confidences]
    n_gt = int(gt.shape[0])
    n_pred = int(pred.shape[0])

    matched_pairs, _ = _match_persons(gt, pred, match_threshold_mm)
    matched_count = len(matched_pairs)
    missed_count = max(0, n_gt - matched_count)
    false_positive_count = max(0, n_pred - matched_count)

    matched_errors = []
    per_joint_sum = None
    per_joint_dim_sum = None
    for gt_idx, pred_idx, _ in matched_pairs:
        joint_diff = gt[gt_idx] - pred[pred_idx]
        joint_error = np.linalg.norm(joint_diff, axis=-1) * 1000.0
        dim_error = np.abs(joint_diff) * 1000.0
        matched_errors.append(float(np.mean(joint_error)))
        if per_joint_sum is None:
            per_joint_sum = np.zeros_like(joint_error, dtype=np.float64)
            per_joint_dim_sum = np.zeros_like(dim_error, dtype=np.float64)
        per_joint_sum += joint_error
        per_joint_dim_sum += dim_error

    matched_error_sum = float(np.sum(matched_errors))
    miss_penalty_sum = float(missed_count * miss_penalty_mm)
    denominator = float(n_gt) if n_gt else 1.0
    matched_mpjpe = (
        matched_error_sum / matched_count if matched_count else float('nan'))

    record = {
        'sample_index': int(sample_index),
        'sample_id': str(sample_id),
        'scene_cardinality': n_gt,
        'predicted_persons': n_pred,
        'matched_persons': matched_count,
        'missed_persons': missed_count,
        'false_positive_persons': false_positive_count,
        'matched_mpjpe': matched_mpjpe,
        'overall_mpjpe': (matched_error_sum + miss_penalty_sum) / denominator,
        'matched_error_sum': matched_error_sum,
        'miss_penalty_sum': miss_penalty_sum,
        'matched_localization_contribution': matched_error_sum / denominator,
        'miss_penalty_contribution': miss_penalty_sum / denominator,
        'match_threshold_mm': float(match_threshold_mm),
        'miss_penalty_mm': float(miss_penalty_mm),
        'matched_pairs': [
            {
                'gt_index': gt_idx,
                'pred_index': pred_idx,
                'pair_mpjpe': pair_error,
            }
            for gt_idx, pred_idx, pair_error in matched_pairs
        ],
        'kept_confidences': confidences,
    }
    if matched_count and per_joint_sum is not None:
        record['per_joint_mpjpe'] = (
            per_joint_sum / matched_count).astype(float).tolist()
        record['per_joint_mpjdle'] = (
            per_joint_dim_sum / matched_count).astype(float).tolist()
    else:
        record['per_joint_mpjpe'] = []
        record['per_joint_mpjdle'] = []
    return record


def aggregate_records(records):
    total_gt = sum(int(row.get('scene_cardinality', 0)) for row in records)
    matched = sum(int(row.get('matched_persons', 0)) for row in records)
    missed = sum(int(row.get('missed_persons', 0)) for row in records)
    false_pos = sum(int(row.get('false_positive_persons', 0)) for row in records)
    matched_error = sum(float(row.get('matched_error_sum', 0.0)) for row in records)
    miss_penalty = sum(float(row.get('miss_penalty_sum', 0.0)) for row in records)
    denom = float(total_gt) if total_gt else 1.0
    return {
        'samples': len(records),
        'total_gt_persons': total_gt,
        'matched_persons': matched,
        'missed_persons': missed,
        'false_positive_persons': false_pos,
        'overall_mpjpe': (matched_error + miss_penalty) / denom,
        'matched_contribution': matched_error / denom,
        'miss_penalty_contribution': miss_penalty / denom,
    }


def _align_by_sample_id(baseline_records, final_records):
    baseline_by_id = {row['sample_id']: row for row in baseline_records}
    final_by_id = {row['sample_id']: row for row in final_records}
    shared_ids = sorted(set(baseline_by_id) & set(final_by_id))
    return [baseline_by_id[x] for x in shared_ids], [final_by_id[x] for x in shared_ids]


def _delta_summary(baseline_records, final_records):
    baseline = aggregate_records(baseline_records)
    final = aggregate_records(final_records)
    return {
        'delta_overall_mpjpe': final['overall_mpjpe'] - baseline['overall_mpjpe'],
        'delta_matched_contribution': (
            final['matched_contribution'] - baseline['matched_contribution']),
        'delta_miss_penalty_contribution': (
            final['miss_penalty_contribution'] - baseline['miss_penalty_contribution']),
        'baseline_overall_mpjpe': baseline['overall_mpjpe'],
        'final_overall_mpjpe': final['overall_mpjpe'],
        'baseline_missed_persons': baseline['missed_persons'],
        'final_missed_persons': final['missed_persons'],
        'samples': baseline['samples'],
        'total_gt_persons': baseline['total_gt_persons'],
    }


def _ci(values):
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def _compare_subset(baseline_records, final_records, n_boot, seed):
    baseline_records, final_records = _align_by_sample_id(
        baseline_records, final_records)
    point = _delta_summary(baseline_records, final_records)
    if not baseline_records:
        point['ci95_delta_overall_mpjpe'] = [None, None]
        point['ci95_delta_matched_contribution'] = [None, None]
        point['ci95_delta_miss_penalty_contribution'] = [None, None]
        return point

    rng = np.random.default_rng(seed)
    n = len(baseline_records)
    boot_overall = []
    boot_matched = []
    boot_miss = []
    for _ in range(n_boot):
        indices = rng.integers(0, n, size=n)
        sample_baseline = [baseline_records[i] for i in indices]
        sample_final = [final_records[i] for i in indices]
        delta = _delta_summary(sample_baseline, sample_final)
        boot_overall.append(delta['delta_overall_mpjpe'])
        boot_matched.append(delta['delta_matched_contribution'])
        boot_miss.append(delta['delta_miss_penalty_contribution'])

    point['ci95_delta_overall_mpjpe'] = _ci(boot_overall)
    point['ci95_delta_matched_contribution'] = _ci(boot_matched)
    point['ci95_delta_miss_penalty_contribution'] = _ci(boot_miss)
    return point


def compare_record_sets(baseline_records, final_records, n_boot=10000, seed=20260626):
    """Compare two per-sample exports with paired bootstrap CIs."""
    report = {
        'all': _compare_subset(baseline_records, final_records, n_boot, seed)
    }
    for cardinality in (1, 2, 3):
        baseline_subset = [
            row for row in baseline_records
            if int(row.get('scene_cardinality', 0)) == cardinality
        ]
        final_subset = [
            row for row in final_records
            if int(row.get('scene_cardinality', 0)) == cardinality
        ]
        report[f'{cardinality}p'] = _compare_subset(
            baseline_subset, final_subset, n_boot, seed + cardinality)
    return report


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, sort_keys=True) + '\n')


def read_jsonl(path):
    rows = []
    with Path(path).open('r', encoding='utf-8') as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare two per-sample pose metric JSONL exports.')
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--final', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--n-boot', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260626)
    return parser.parse_args()


def main():
    args = parse_args()
    report = compare_record_sets(
        read_jsonl(args.baseline),
        read_jsonl(args.final),
        n_boot=args.n_boot,
        seed=args.seed,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Wrote paired analysis to {out_path}')


if __name__ == '__main__':
    main()
