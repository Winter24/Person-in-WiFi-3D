import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / 'scripts' / 'run_paper_postprocess.sh'


class TestPaperPostprocessScript(unittest.TestCase):
    def test_script_exists(self):
        self.assertTrue(
            SCRIPT_PATH.exists(),
            f'Missing script: {SCRIPT_PATH}')

    def test_script_has_checkpoint_overrides(self):
        text = SCRIPT_PATH.read_text(encoding='utf-8')
        for run_id in ('B0', 'B1', 'B2', 'B4', 'B5'):
            self.assertIn(f'{run_id}_CKPT=""', text)

    def test_script_has_checkpoint_fallback_logic(self):
        text = SCRIPT_PATH.read_text(encoding='utf-8')
        self.assertIn('latest.pth', text)
        self.assertIn('epoch_*.pth', text)

    def test_script_calls_expected_tools(self):
        text = SCRIPT_PATH.read_text(encoding='utf-8')
        self.assertIn('python tools/test.py', text)
        self.assertIn('python tools/analysis/benchmark.py', text)
        self.assertIn('python tools/analysis/append_experiment_log.py', text)

    def test_script_targets_expected_runs(self):
        text = SCRIPT_PATH.read_text(encoding='utf-8')
        self.assertIn('RUN_IDS=(B0 B1 B2 B4 B5)', text)


if __name__ == '__main__':
    unittest.main()
