import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / 'tools' / 'analysis' / 'experiment_manifest.py'


def load_module():
    spec = importlib.util.spec_from_file_location(
        'experiment_manifest', SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestExperimentManifest(unittest.TestCase):
    def test_manifest_keeps_metrics_and_train_provenance_separate(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            experiment_log = temp_path / 'experiment_log.csv'
            with experiment_log.open('w', newline='', encoding='utf-8') as file_obj:
                writer = csv.DictWriter(
                    file_obj,
                    fieldnames=[
                        'experiment_id', 'config', 'checkpoint', 'mpjpe',
                        'fps', 'params_m', 'peak_memory_allocated_mb',
                    ])
                writer.writeheader()
                writer.writerow({
                    'experiment_id': 'M9',
                    'config': 'configs/wifi/final.py',
                    'checkpoint': 'work_dirs/M9/latest.pth',
                    'mpjpe': '168.0858819410895',
                    'fps': '206.9299',
                    'params_m': '3.6175',
                    'peak_memory_allocated_mb': '25.29',
                })

            log_dir = temp_path / 'launch_logs'
            log_dir.mkdir()
            (log_dir / 'M9.log').write_text(
                '\n'.join([
                    'Config: configs/wifi/final.py',
                    'samples_per_gpu=32',
                    'optimizer = dict(type="AdamW", lr=2e-05, weight_decay=0.0001)',
                    'seed=42, deterministic=True',
                ]),
                encoding='utf-8')

            manifest = module.build_manifest(
                experiment_log=experiment_log,
                launch_log_dir=log_dir,
                commit='abc123',
                dataset_signature='sha-demo',
                hardware='RTX A6000',
            )

        record = manifest['experiments'][0]
        self.assertEqual(record['experiment_id'], 'M9')
        self.assertEqual(record['config'], 'configs/wifi/final.py')
        self.assertEqual(record['checkpoint'], 'work_dirs/M9/latest.pth')
        self.assertEqual(record['commit'], 'abc123')
        self.assertEqual(record['dataset']['ordered_sample_sha256'], 'sha-demo')
        self.assertEqual(record['training']['samples_per_gpu'], 32)
        self.assertAlmostEqual(record['training']['weight_decay'], 0.0001)
        self.assertEqual(record['training']['seed'], 42)
        self.assertEqual(record['hardware'], 'RTX A6000')
        self.assertEqual(record['provenance_status'], 'complete')
        self.assertAlmostEqual(record['metrics']['mpjpe'], 168.0858819410895)

    def test_missing_launch_log_marks_metrics_only_instead_of_guessing(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            experiment_log = temp_path / 'experiment_log.csv'
            experiment_log.write_text(
                'experiment_id,config,checkpoint,mpjpe\n'
                'M0,configs/wifi/petr.py,work_dirs/M0/latest.pth,172.54\n',
                encoding='utf-8')

            manifest = module.build_manifest(
                experiment_log=experiment_log,
                launch_log_dir=temp_path / 'missing_logs',
                commit='abc123',
                dataset_signature='sha-demo',
                hardware=None,
            )

        record = manifest['experiments'][0]
        self.assertEqual(record['provenance_status'], 'metrics_only')
        self.assertIsNone(record['training']['samples_per_gpu'])
        self.assertIsNone(record['training']['weight_decay'])
        self.assertIn('launch log not found', record['warnings'][0])

    def test_write_manifest_exports_json_and_latex_table(self):
        module = load_module()
        manifest = {
            'experiments': [
                {
                    'experiment_id': 'M0',
                    'config': 'configs/wifi/petr.py',
                    'checkpoint': 'work_dirs/M0/latest.pth',
                    'commit': 'abc123',
                    'dataset': {'ordered_sample_sha256': 'sha-demo'},
                    'training': {
                        'samples_per_gpu': None,
                        'weight_decay': None,
                        'seed': None,
                    },
                    'metrics': {'mpjpe': 172.54},
                    'provenance_status': 'metrics_only',
                    'warnings': ['launch log not found'],
                    'hardware': None,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            out_json = Path(temp_dir) / 'manifest.json'
            out_tex = Path(temp_dir) / 'manifest_table.tex'

            module.write_manifest(manifest, out_json)
            module.write_manifest_table(manifest, out_tex)

            exported = json.loads(out_json.read_text(encoding='utf-8'))
            tex = out_tex.read_text(encoding='utf-8')

        self.assertEqual(exported['experiments'][0]['experiment_id'], 'M0')
        self.assertIn('M0', tex)
        self.assertIn('petr.py', tex)
        self.assertIn('M0/latest.pth', tex)
        self.assertIn('metrics\\_only', tex)


if __name__ == '__main__':
    unittest.main()
