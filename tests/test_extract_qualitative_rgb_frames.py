import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / 'tools' / 'analysis' / 'extract_qualitative_rgb_frames.py'


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestExtractQualitativeRgbFrames(unittest.TestCase):
    def test_script_exists(self):
        self.assertTrue(SCRIPT_PATH.exists(), f'Missing script: {SCRIPT_PATH}')

    def test_script_help_runs_from_repo_root(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), '--help'],
            cwd=ROOT,
            capture_output=True,
            text=True)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('--source-video-root', completed.stdout)

    def test_default_specs_match_fixed_manuscript_samples(self):
        module = _load_module('extract_qualitative_rgb_frames', SCRIPT_PATH)

        specs = module.build_default_frame_specs()

        self.assertEqual(
            [(spec.sample_name, spec.video_id, spec.frame_id) for spec in specs],
            [
                ('S11_06_319', 'S11_06', 319),
                ('S52_40_322', 'S52_40', 322),
                ('S23_12_337', 'S23_12', 337),
            ])

    def test_resolve_exact_frame_index_rejects_missing_id(self):
        module = _load_module('extract_qualitative_rgb_frames', SCRIPT_PATH)

        self.assertEqual(
            module.resolve_exact_frame_index(322, {321: 18, 322: 19}),
            19)
        with self.assertRaisesRegex(KeyError, '337'):
            module.resolve_exact_frame_index(337, {336: 20, 338: 22})


if __name__ == '__main__':
    unittest.main()
