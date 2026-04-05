import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / 'tools' / 'analysis' / 'extract_ablation_results.py'


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExtractAblationResults(unittest.TestCase):
    def test_script_exists(self):
        self.assertTrue(
            SCRIPT_PATH.exists(),
            f'Missing script: {SCRIPT_PATH}')

    def test_main_writes_last_val_row_per_run(self):
        module = _load_module('extract_ablation_results', SCRIPT_PATH)

        with tempfile.TemporaryDirectory() as tmpdir:
            paper_dir = Path(tmpdir) / 'paper'
            paper_dir.mkdir()

            samples = {
                'B1': [
                    {'mode': 'train', 'epoch': 5, 'iter': 2750, 'loss': 5.9},
                    {'mode': 'val', 'epoch': 4, 'iter': 7824, 'mpjpe': 250.0},
                    {'mode': 'val', 'epoch': 5, 'iter': 7824, 'mpjpe': 242.66737},
                ],
                'B2': [
                    {'mode': 'train', 'epoch': 5, 'iter': 2800, 'loss': 5.8},
                    {'mode': 'val', 'epoch': 5, 'iter': 7824, 'mpjpe': 265.33777},
                ],
                'B0_bone': [
                    {'mode': 'val', 'epoch': 4, 'iter': 7824, 'mpjpe': 245.0},
                    {'mode': 'train', 'epoch': 5, 'iter': 2800, 'loss': 5.7},
                    {'mode': 'val', 'epoch': 5, 'iter': 7824, 'mpjpe': 243.16991},
                ],
            }

            for run_name, rows in samples.items():
                run_dir = paper_dir / run_name
                run_dir.mkdir()
                log_path = run_dir / f'{run_name.lower()}_test.log.json'
                with log_path.open('w', encoding='utf-8') as file_obj:
                    for row in rows:
                        file_obj.write(json.dumps(row) + '\n')

            output_path = paper_dir / 'ablation.txt'
            module.main([
                '--paper-dir', str(paper_dir),
                '--output', str(output_path),
            ])

            self.assertTrue(output_path.exists())

            content = output_path.read_text(encoding='utf-8')
            expected_lines = [
                'B0_bone:{"mode": "val", "epoch": 5, "iter": 7824, "mpjpe": 243.16991}',
                'B1:{"mode": "val", "epoch": 5, "iter": 7824, "mpjpe": 242.66737}',
                'B2:{"mode": "val", "epoch": 5, "iter": 7824, "mpjpe": 265.33777}',
            ]
            self.assertEqual(content.strip().split('\n\n'), expected_lines)


if __name__ == '__main__':
    unittest.main()
