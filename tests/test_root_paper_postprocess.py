import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / 'tools' / 'analysis' / 'run_root_paper_postprocess.py'


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRootPaperPostprocess(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix='root_paper_postprocess_', dir=ROOT / 'tests'))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_script_exists(self):
        self.assertTrue(SCRIPT_PATH.exists(), f'Missing script: {SCRIPT_PATH}')

    def test_discover_run_dirs_returns_directories_sorted(self):
        module = _load_module('run_root_paper_postprocess', SCRIPT_PATH)

        paper_dir = self.tmp_dir / 'paper'
        paper_dir.mkdir()
        (paper_dir / 'M5_v2').mkdir()
        (paper_dir / 'M1_v1').mkdir()
        (paper_dir / 'readme.txt').write_text('ignore me', encoding='utf-8')

        run_dirs = module.discover_run_dirs(paper_dir)

        self.assertEqual([path.name for path in run_dirs], ['M1_v1', 'M5_v2'])

    def test_resolve_checkpoint_prefers_latest_checkpoint(self):
        module = _load_module('run_root_paper_postprocess', SCRIPT_PATH)

        run_dir = self.tmp_dir / 'M1_v1'
        run_dir.mkdir()
        (run_dir / 'epoch_4.pth').write_text('', encoding='utf-8')
        latest = run_dir / 'latest.pth'
        latest.write_text('', encoding='utf-8')

        self.assertEqual(module.resolve_checkpoint(run_dir), latest)

    def test_run_one_uses_folder_name_as_experiment_id(self):
        module = _load_module('run_root_paper_postprocess', SCRIPT_PATH)

        run_dir = self.tmp_dir / 'M5_v2'
        run_dir.mkdir()
        config_path = run_dir / 'wi_tidir_wifi.py'
        config_path.write_text('model = dict(type="Demo")', encoding='utf-8')
        checkpoint_path = run_dir / 'latest.pth'
        checkpoint_path.write_text('', encoding='utf-8')

        output_dir = self.tmp_dir / 'outputs'
        eval_work_root = self.tmp_dir / 'paper_eval'
        output_dir.mkdir()
        eval_work_root.mkdir()

        args = module.parse_args([
            '--paper-dir', str(self.tmp_dir),
            '--output-dir', str(output_dir),
            '--eval-work-root', str(eval_work_root),
        ])

        with mock.patch.object(module.subprocess, 'run') as mock_run:
            module.run_one(args, run_dir)

        commands = [call.args[0] for call in mock_run.call_args_list]

        self.assertEqual(len(commands), 3)
        self.assertEqual(commands[0][0:2], ['python', 'tools/test.py'])
        self.assertIn('--metrics-out', commands[0])
        self.assertIn(str(output_dir / 'M5_v2_eval.json'), commands[0])
        self.assertEqual(commands[1][0:2], ['python', 'tools/analysis/benchmark.py'])
        self.assertIn(str(output_dir / 'M5_v2_benchmark.json'), commands[1])
        self.assertEqual(commands[2][0:2], ['python', 'tools/analysis/append_experiment_log.py'])
        self.assertIn('M5_v2', commands[2])
        self.assertIn(str(output_dir / 'experiment_log.csv'), commands[2])


if __name__ == '__main__':
    unittest.main()
