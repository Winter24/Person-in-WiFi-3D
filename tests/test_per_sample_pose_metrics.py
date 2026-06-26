import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / 'tools' / 'analysis' / 'per_sample_pose_metrics.py'


def load_module():
    spec = importlib.util.spec_from_file_location(
        'per_sample_pose_metrics', SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestPerSamplePoseMetrics(unittest.TestCase):
    def test_sample_record_decomposes_overall_into_matched_and_miss_terms(self):
        module = load_module()
        record = module.build_sample_record(
            sample_index=7,
            sample_id='S11_06_319',
            gt_keypoints=np.zeros((2, 14, 3), dtype=np.float32),
            pred_keypoints=np.stack([
                np.full((14, 3), 0.01, dtype=np.float32),
                np.full((14, 3), 2.0, dtype=np.float32),
            ]),
            confidences=[0.9, 0.2],
            match_threshold_mm=500.0,
            miss_penalty_mm=500.0,
        )

        self.assertEqual(record['sample_index'], 7)
        self.assertEqual(record['sample_id'], 'S11_06_319')
        self.assertEqual(record['scene_cardinality'], 2)
        self.assertEqual(record['matched_persons'], 1)
        self.assertEqual(record['missed_persons'], 1)
        self.assertEqual(record['false_positive_persons'], 1)
        self.assertEqual(record['kept_confidences'], [0.9, 0.2])
        self.assertAlmostEqual(
            record['overall_mpjpe'],
            (record['matched_error_sum'] + record['miss_penalty_sum']) / 2)
        self.assertAlmostEqual(record['miss_penalty_contribution'], 250.0)

    def test_aggregate_records_reconstructs_metric(self):
        module = load_module()
        records = [
            {
                'sample_id': 'a',
                'scene_cardinality': 1,
                'matched_persons': 1,
                'missed_persons': 0,
                'matched_error_sum': 100.0,
                'miss_penalty_sum': 0.0,
            },
            {
                'sample_id': 'b',
                'scene_cardinality': 3,
                'matched_persons': 2,
                'missed_persons': 1,
                'matched_error_sum': 300.0,
                'miss_penalty_sum': 500.0,
            },
        ]

        aggregate = module.aggregate_records(records)

        self.assertEqual(aggregate['total_gt_persons'], 4)
        self.assertEqual(aggregate['matched_persons'], 3)
        self.assertEqual(aggregate['missed_persons'], 1)
        self.assertAlmostEqual(aggregate['overall_mpjpe'], 225.0)
        self.assertAlmostEqual(aggregate['matched_contribution'], 100.0)
        self.assertAlmostEqual(aggregate['miss_penalty_contribution'], 125.0)

    def test_paired_bootstrap_is_deterministic_and_preserves_decomposition(self):
        module = load_module()
        baseline = [
            {
                'sample_id': 'a',
                'scene_cardinality': 1,
                'matched_persons': 1,
                'missed_persons': 0,
                'matched_error_sum': 200.0,
                'miss_penalty_sum': 0.0,
            },
            {
                'sample_id': 'b',
                'scene_cardinality': 2,
                'matched_persons': 1,
                'missed_persons': 1,
                'matched_error_sum': 300.0,
                'miss_penalty_sum': 500.0,
            },
        ]
        final = [
            {
                'sample_id': 'a',
                'scene_cardinality': 1,
                'matched_persons': 1,
                'missed_persons': 0,
                'matched_error_sum': 150.0,
                'miss_penalty_sum': 0.0,
            },
            {
                'sample_id': 'b',
                'scene_cardinality': 2,
                'matched_persons': 2,
                'missed_persons': 0,
                'matched_error_sum': 300.0,
                'miss_penalty_sum': 0.0,
            },
        ]

        report_1 = module.compare_record_sets(
            baseline, final, n_boot=200, seed=123)
        report_2 = module.compare_record_sets(
            baseline, final, n_boot=200, seed=123)

        self.assertEqual(report_1, report_2)
        self.assertAlmostEqual(
            report_1['all']['delta_overall_mpjpe'],
            report_1['all']['delta_matched_contribution']
            + report_1['all']['delta_miss_penalty_contribution'])
        self.assertLess(report_1['all']['delta_overall_mpjpe'], 0)

    def test_jsonl_roundtrip(self):
        module = load_module()
        rows = [{'sample_id': 'a', 'scene_cardinality': 1}]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'rows.jsonl'
            module.write_jsonl(rows, path)
            loaded = module.read_jsonl(path)

        self.assertEqual(loaded, rows)

    def test_wifi_pose_evaluator_exposes_per_sample_export_option(self):
        source = (ROOT / 'opera' / 'datasets' / 'wifi_pose.py').read_text(
            encoding='utf-8')
        self.assertIn('per_sample_metrics_out=None', source)
        self.assertIn('build_sample_record', source)
        self.assertIn('write_jsonl(per_sample_records', source)

    def test_server_inference_runbook_exports_required_models(self):
        script = ROOT / 'tools' / 'analysis' / 'run_evidence_inference.sh'
        self.assertTrue(script.exists())
        source = script.read_text(encoding='utf-8')
        for model_id in ['M0', 'M6', 'M9_no_flow', 'M9_RF1', 'M9_RF2']:
            self.assertIn(model_id, source)
        self.assertIn('per_sample_metrics_out=', source)
        self.assertIn('analyze_per_sample_metrics.py', source)


if __name__ == '__main__':
    unittest.main()
