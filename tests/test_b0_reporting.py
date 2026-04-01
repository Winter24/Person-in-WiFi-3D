"""Unit tests for B0 evaluation & reporting gap-closure.

Tests cover:
  1. WifiPoseDataset.evaluate() per-person breakdown
  2. benchmark.py JSON schema & CPU/CUDA guards
  3. append_experiment_log.py CSV create / upsert
"""

import csv
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Ensure the repo root is on sys.path so we can import helpers directly
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'tools' / 'analysis'))


# ---------------------------------------------------------------------------
# Helper: load a Python module by path (for scripts under tools/)
# ---------------------------------------------------------------------------
def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Helper: try to import WifiPoseDataset; skip test if mmcv/mmdet absent
# ---------------------------------------------------------------------------
def _import_wifi_pose():
    """Return (WifiPoseDataset, _json_serializer) or raise unittest.SkipTest."""
    try:
        from opera.datasets.wifi_pose import WifiPoseDataset, _json_serializer
        return WifiPoseDataset, _json_serializer
    except ImportError as exc:
        raise unittest.SkipTest(
            f"opera.datasets.wifi_pose not importable (mmcv/mmdet missing): {exc}"
        )


# ===========================================================================
# 1. Evaluation breakdown tests
# ===========================================================================
class TestEvaluateBreakdown(unittest.TestCase):
    """Tests for WifiPoseDataset.evaluate() per-person-count bucketing."""

    _JOINT_NAMES = [
        'Head', 'Neck', 'R_Shoulder', 'L_Shoulder', 'R_Elbow',
        'L_Elbow', 'R_Hip', 'L_Wrist', 'R_Wrist', 'R_Knee',
        'R_Ankle', 'L_Knee', 'L_Hip', 'L_Ankle'
    ]
    _TARGET_BONES = [
        (3, 2), (12, 6), (3, 5), (5, 7), (2, 4),
        (4, 8), (12, 11), (11, 13), (6, 9), (9, 10),
    ]

    @staticmethod
    def _make_fake_results(n_persons, n_joints=14):
        """Create a fake (det_bboxes, det_keypoints) result tuple."""
        kpt = np.random.randn(n_persons, n_joints, 3).astype(np.float32) * 0.01
        det_bboxes = [np.zeros((0, 5), dtype=np.float32)]
        det_keypoints = [kpt]
        return (det_bboxes, det_keypoints)

    @staticmethod
    def _make_fake_gt(n_persons, n_joints=14):
        """Return a tensor-like ndarray simulating gt_keypoints."""
        import torch
        return torch.randn(n_persons, n_joints, 3).float() * 0.01

    def _make_ds(self, WifiPoseDataset):
        """Return a minimal WifiPoseDataset instance with __init__ skipped."""
        with patch.object(WifiPoseDataset, '__init__', lambda self, **kw: None):
            ds = WifiPoseDataset.__new__(WifiPoseDataset)
            ds.JOINT_NAMES = self._JOINT_NAMES
            ds.TARGET_BONES = self._TARGET_BONES
        return ds

    @staticmethod
    def _deterministic_match(gt_kpts, pred_kpts):
        """Replacement for calc_mpjpe_and_match that always returns a valid result.

        Returns fixed-value metrics so counts/finiteness assertions are stable
        regardless of random input magnitudes or scipy availability.
        """
        import torch
        n = min(gt_kpts.shape[0], pred_kpts.shape[0])
        matched_gt = gt_kpts[:n]
        matched_pred = pred_kpts[:n]
        diff = matched_gt - matched_pred
        per_joint_error_3d = torch.norm(diff, p=2, dim=-1)
        mpjpe = per_joint_error_3d.mean() * 1000
        per_dim = torch.abs(diff)
        metrics = [
            mpjpe.cpu().numpy(),
            (per_dim[..., 0].mean() * 1000).cpu().numpy(),
            (per_dim[..., 1].mean() * 1000).cpu().numpy(),
            (per_dim[..., 2].mean() * 1000).cpu().numpy(),
        ]
        per_joint = per_joint_error_3d.mean(dim=0).cpu().numpy() * 1000
        return metrics, per_joint, matched_pred, matched_gt

    def test_result_dict_has_all_breakdown_keys(self):
        """evaluate() must return mpjpe, mpjpe_Xp, count_Xp, matched_Xp for X in 1,2,3."""
        import torch
        WifiPoseDataset, _ = _import_wifi_pose()

        person_counts = [1, 2, 3]
        results = [self._make_fake_results(pc) for pc in person_counts]
        gt_frames = [
            {'gt_keypoints': self._make_fake_gt(pc), 'img_name': f'fake_{i}'}
            for i, pc in enumerate(person_counts)
        ]

        ds = self._make_ds(WifiPoseDataset)
        ds.get_item_single_frame = lambda i: gt_frames[i]
        ds.calc_mpjpe_and_match = self._deterministic_match

        result = ds.evaluate(results)

        # All expected keys must be present
        expected_keys = [
            'mpjpe', 'mpjpeh', 'mpjpev', 'mpjped',
            'mpjpe_1p', 'mpjpe_2p', 'mpjpe_3p',
            'count_1p', 'count_2p', 'count_3p',
            'matched_1p', 'matched_2p', 'matched_3p',
        ]
        for key in expected_keys:
            self.assertIn(key, result, f"Missing key: {key}")

        # count_Xp = GT-split denominator (one frame per split in this test)
        self.assertEqual(result['count_1p'], 1,
                         "count_1p should equal the number of 1-person GT frames")
        self.assertEqual(result['count_2p'], 1,
                         "count_2p should equal the number of 2-person GT frames")
        self.assertEqual(result['count_3p'], 1,
                         "count_3p should equal the number of 3-person GT frames")

        # matched_Xp <= count_Xp
        self.assertLessEqual(result['matched_1p'], result['count_1p'])
        self.assertLessEqual(result['matched_2p'], result['count_2p'])
        self.assertLessEqual(result['matched_3p'], result['count_3p'])

        # All per-split MPJPE values should be finite (deterministic match ensures this)
        for key in ['mpjpe', 'mpjpe_1p', 'mpjpe_2p', 'mpjpe_3p']:
            self.assertIsInstance(result[key], float)
            self.assertTrue(np.isfinite(result[key]), f"{key} not finite")

    def test_missing_bucket_returns_nan(self):
        """If a bucket has zero GT frames, its mpjpe_Xp should be NaN and count_Xp == 0."""
        import torch
        WifiPoseDataset, _ = _import_wifi_pose()

        # Only 2-person frames
        results = [self._make_fake_results(2) for _ in range(3)]
        gt_frames = [
            {'gt_keypoints': self._make_fake_gt(2), 'img_name': f'fake_{i}'}
            for i in range(3)
        ]

        ds = self._make_ds(WifiPoseDataset)
        ds.get_item_single_frame = lambda i: gt_frames[i]
        ds.calc_mpjpe_and_match = self._deterministic_match

        result = ds.evaluate(results)

        self.assertTrue(np.isnan(result['mpjpe_1p']))
        self.assertTrue(np.isnan(result['mpjpe_3p']))
        # count_Xp is the GT denominator — zero because no such GT frames exist
        self.assertEqual(result['count_1p'], 0)
        self.assertEqual(result['count_3p'], 0)
        # All 3 frames have 2-person GT
        self.assertEqual(result['count_2p'], 3)

    def test_metrics_out_json_export(self):
        """--metrics-out should create a valid JSON with full schema."""
        import torch
        WifiPoseDataset, _ = _import_wifi_pose()

        results = [self._make_fake_results(2)]
        gt_frames = [
            {'gt_keypoints': self._make_fake_gt(2), 'img_name': 'fake_0'}
        ]

        ds = self._make_ds(WifiPoseDataset)
        ds.get_item_single_frame = lambda i: gt_frames[i]
        ds.calc_mpjpe_and_match = self._deterministic_match

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, 'test_eval.json')
            ds.evaluate(results, metrics_out=out_path)

            self.assertTrue(os.path.isfile(out_path))
            with open(out_path, 'r') as f:
                data = json.load(f)

            for key in ['mpjpe', 'mpjpe_1p', 'mpjpe_2p', 'mpjpe_3p',
                         'count_1p', 'count_2p', 'count_3p',
                         'matched_1p', 'matched_2p', 'matched_3p',
                         'per_joint_mpjpe', 'bone_length_error']:
                self.assertIn(key, data, f"JSON missing key: {key}")


# ===========================================================================
# 2. Benchmark reporting tests
# ===========================================================================
class TestBenchmarkScript(unittest.TestCase):
    """Tests for tools/analysis/benchmark.py (static / schema checks)."""

    def test_out_flag_creates_valid_json_schema(self):
        """If --out is given, the JSON must contain required fields."""
        required_keys = [
            'config', 'checkpoint', 'device', 'input_shape', 'times',
            'warmup', 'params_m', 'trainable_params_m', 'flops_g',
            'latency_ms', 'fps',
            'peak_memory_allocated_mb', 'peak_memory_reserved_mb',
        ]

        benchmark_src = (REPO_ROOT / 'tools' / 'analysis' / 'benchmark.py').read_text()
        for key in required_keys:
            # Accept both dict-literal style  ('key': ...)
            # and dict() keyword-argument style  (key=...)
            present = (f"'{key}'" in benchmark_src) or (f"{key}=" in benchmark_src)
            self.assertTrue(present,
                            f"benchmark.py report dict missing key '{key}'")

    def test_cpu_path_no_cuda_sync(self):
        """When device=cpu, torch.cuda.synchronize should NOT be called."""
        benchmark_src = (REPO_ROOT / 'tools' / 'analysis' / 'benchmark.py').read_text()

        # The script should guard cuda.synchronize behind use_cuda / _is_cuda
        self.assertIn('_is_cuda', benchmark_src,
                       "benchmark.py should have _is_cuda guard")
        self.assertIn('if use_cuda', benchmark_src,
                       "cuda sync must be guarded by `if use_cuda`")

    def test_cuda_path_has_memory_fields(self):
        """CUDA path should measure peak_memory_allocated_mb."""
        benchmark_src = (REPO_ROOT / 'tools' / 'analysis' / 'benchmark.py').read_text()
        self.assertIn('max_memory_allocated', benchmark_src)
        self.assertIn('max_memory_reserved', benchmark_src)
        self.assertIn('peak_memory_allocated_mb', benchmark_src)

    def test_default_shape_is_wifi(self):
        """Default --shape should be [1, 3, 3, 20, 60] for WiFi CSI."""
        benchmark_src = (REPO_ROOT / 'tools' / 'analysis' / 'benchmark.py').read_text()
        self.assertIn('[1, 3, 3, 20, 60]', benchmark_src)

    def test_invalid_cuda_ordinal_is_normalized(self):
        """Invalid cuda:N requests should fall back safely."""
        benchmark_src = (REPO_ROOT / 'tools' / 'analysis' / 'benchmark.py').read_text()
        self.assertIn('torch.cuda.device_count()', benchmark_src)
        self.assertIn('Invalid CUDA device ordinal', benchmark_src)
        self.assertIn("Falling back to 'cuda:0'", benchmark_src)

    def test_missing_cuda_fails_clearly(self):
        """Explicit CUDA benchmark requests should not silently fall back to CPU."""
        benchmark_src = (REPO_ROOT / 'tools' / 'analysis' / 'benchmark.py').read_text()
        self.assertIn('torch.cuda.is_available() is False', benchmark_src)
        self.assertIn('Do not fall back to CPU for Mamba-based models', benchmark_src)

    def test_shape_arg_actually_used(self):
        """args.shape should feed into torch.randn (not ignored)."""
        benchmark_src = (REPO_ROOT / 'tools' / 'analysis' / 'benchmark.py').read_text()
        self.assertIn('args.shape', benchmark_src)
        self.assertIn('torch.randn(*input_shape)', benchmark_src)


# ===========================================================================
# 3. Experiment log CSV tests
# ===========================================================================
class TestAppendExperimentLog(unittest.TestCase):
    """Tests for tools/analysis/append_experiment_log.py."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmpdir, 'experiment_log.csv')

        # Create sample eval JSON
        self.eval_json_path = os.path.join(self.tmpdir, 'eval.json')
        eval_data = {
            'mpjpe': 85.3,
            'mpjpeh': 40.1,
            'mpjpev': 35.2,
            'mpjped': 30.0,
            'mpjpe_1p': 70.0,
            'mpjpe_2p': 85.0,
            'mpjpe_3p': 100.5,
            'count_1p': 2586,
            'count_2p': 3184,
            'count_3p': 2054,
        }
        with open(self.eval_json_path, 'w') as f:
            json.dump(eval_data, f)

        # Create sample benchmark JSON
        self.bench_json_path = os.path.join(self.tmpdir, 'bench.json')
        bench_data = {
            'latency_ms': 12.5,
            'fps': 80.0,
            'peak_memory_allocated_mb': 512.0,
            'params_m': 28.5,
            'flops_g': 4.2,
        }
        with open(self.bench_json_path, 'w') as f:
            json.dump(bench_data, f)

        # Load the module
        self.mod = _load_module(
            'append_experiment_log',
            str(REPO_ROOT / 'tools' / 'analysis' / 'append_experiment_log.py'))

    def _make_args(self, experiment_id='B0', notes=''):
        args = MagicMock()
        args.experiment_id = experiment_id
        args.config = 'configs/wifi/petr_wifi.py'
        args.checkpoint = 'checkpoints/B0.pth'
        args.eval_json = self.eval_json_path
        args.benchmark_json = self.bench_json_path
        args.csv = self.csv_path
        args.notes = notes
        return args

    def test_creates_csv_from_scratch(self):
        """First run should create the CSV with header + 1 data row."""
        args = self._make_args()
        row = self.mod.build_row(args)
        rows, fns = self.mod.read_csv(self.csv_path)
        rows, was_update = self.mod.upsert(rows, row)
        self.mod.write_csv(self.csv_path, rows, fns)

        self.assertFalse(was_update)
        self.assertTrue(os.path.isfile(self.csv_path))

        with open(self.csv_path, 'r', newline='') as f:
            reader = list(csv.DictReader(f))

        self.assertEqual(len(reader), 1)
        self.assertEqual(reader[0]['experiment_id'], 'B0')
        # Use float comparison to be robust against trailing zeros etc.
        self.assertAlmostEqual(float(reader[0]['mpjpe']), 85.3, places=5)
        self.assertEqual(reader[0]['count_1p'], '2586')
        self.assertAlmostEqual(float(reader[0]['latency_ms']), 12.5, places=5)

    def test_upsert_does_not_duplicate(self):
        """Running twice with the same experiment_id should keep exactly 1 row."""
        for _ in range(2):
            args = self._make_args()
            row = self.mod.build_row(args)
            rows, fns = self.mod.read_csv(self.csv_path)
            rows, _ = self.mod.upsert(rows, row)
            self.mod.write_csv(self.csv_path, rows, fns)

        with open(self.csv_path, 'r', newline='') as f:
            reader = list(csv.DictReader(f))
        self.assertEqual(len(reader), 1)

    def test_upsert_updates_existing_row(self):
        """An upsert should update values of an existing experiment_id."""
        # First insert
        args = self._make_args(notes='v1')
        row = self.mod.build_row(args)
        rows, fns = self.mod.read_csv(self.csv_path)
        rows, was_update = self.mod.upsert(rows, row)
        self.mod.write_csv(self.csv_path, rows, fns)
        self.assertFalse(was_update)

        # Second upsert with different notes
        args2 = self._make_args(notes='v2')
        row2 = self.mod.build_row(args2)
        rows2, fns2 = self.mod.read_csv(self.csv_path)
        rows2, was_update2 = self.mod.upsert(rows2, row2)
        self.mod.write_csv(self.csv_path, rows2, fns2)
        self.assertTrue(was_update2)

        with open(self.csv_path, 'r', newline='') as f:
            reader = list(csv.DictReader(f))
        self.assertEqual(len(reader), 1)
        self.assertEqual(reader[0]['notes'], 'v2')

    def test_upsert_preserves_created_at(self):
        """Upsert must NOT overwrite created_at from the original insert."""
        # First insert
        args1 = self._make_args(notes='v1')
        row1 = self.mod.build_row(args1)
        original_created_at = row1['created_at']
        rows, fns = self.mod.read_csv(self.csv_path)
        rows, _ = self.mod.upsert(rows, row1)
        self.mod.write_csv(self.csv_path, rows, fns)

        # Second upsert — build_row will generate a new timestamp
        args2 = self._make_args(notes='v2')
        row2 = self.mod.build_row(args2)
        # row2['created_at'] may differ from original_created_at by at least 0s;
        # force a different value to make the assertion meaningful
        row2['created_at'] = '2099-01-01T00:00:00Z'
        rows2, fns2 = self.mod.read_csv(self.csv_path)
        rows2, _ = self.mod.upsert(rows2, row2)
        self.mod.write_csv(self.csv_path, rows2, fns2)

        with open(self.csv_path, 'r', newline='') as f:
            reader = list(csv.DictReader(f))
        self.assertEqual(len(reader), 1)
        self.assertEqual(reader[0]['created_at'], original_created_at,
                         "created_at should be preserved from the original insert, not overwritten")

    def test_multiple_experiments(self):
        """Different experiment_ids should each get their own row."""
        for eid in ['B0', 'B1', 'B2']:
            args = self._make_args(experiment_id=eid)
            row = self.mod.build_row(args)
            rows, fns = self.mod.read_csv(self.csv_path)
            rows, _ = self.mod.upsert(rows, row)
            self.mod.write_csv(self.csv_path, rows, fns)

        with open(self.csv_path, 'r', newline='') as f:
            reader = list(csv.DictReader(f))
        self.assertEqual(len(reader), 3)
        ids = [r['experiment_id'] for r in reader]
        self.assertEqual(ids, ['B0', 'B1', 'B2'])

    def test_row_has_all_columns(self):
        """Built row should contain every column from COLUMNS."""
        args = self._make_args()
        row = self.mod.build_row(args)
        for col in self.mod.COLUMNS:
            self.assertIn(col, row, f"Row missing column: {col}")

    def test_created_at_is_populated(self):
        """created_at should be a non-empty ISO-ish timestamp."""
        args = self._make_args()
        row = self.mod.build_row(args)
        self.assertIn('created_at', row)
        self.assertGreater(len(row['created_at']), 0)
        self.assertIn('T', row['created_at'])  # ISO format check


if __name__ == '__main__':
    unittest.main()
