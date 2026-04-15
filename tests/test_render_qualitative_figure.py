import csv
import importlib.util
import shutil
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / 'tools' / 'analysis' / 'render_qualitative_figure.py'


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestRenderQualitativeFigure(unittest.TestCase):
    def setUp(self):
        self.work_dir = ROOT / 'tests' / f'tmp_render_qualitative_{uuid.uuid4().hex}'
        self.work_dir.mkdir(parents=True, exist_ok=False)

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def _write_experiment_log(self, rows):
        log_path = self.work_dir / 'experiment_log.csv'
        with log_path.open('w', newline='', encoding='utf-8') as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=['experiment_id', 'config', 'checkpoint', 'notes'])
            writer.writeheader()
            writer.writerows(rows)
        return log_path

    def test_script_exists(self):
        self.assertTrue(
            SCRIPT_PATH.exists(),
            f'Missing script: {SCRIPT_PATH}')

    def test_load_experiment_specs_uses_config_basenames(self):
        module = _load_module('render_qualitative_figure', SCRIPT_PATH)
        log_path = self._write_experiment_log([
            {
                'experiment_id': 'M0',
                'config': '/content/drive/.../M0/petr_wifi.py',
                'checkpoint': '/content/drive/.../M0/latest.pth',
                'notes': 'baseline',
            },
            {
                'experiment_id': 'M4',
                'config': '/content/drive/.../M4/wi_tidir_wifi.py',
                'checkpoint': '/content/drive/.../M4/latest.pth',
                'notes': 'ours',
            },
        ])

        specs = module.load_experiment_specs(log_path)

        self.assertEqual(specs['M0']['config_name'], 'petr_wifi.py')
        self.assertEqual(specs['M4']['config_name'], 'wi_tidir_wifi.py')

    def test_resolve_model_assets_prefers_local_checkpoint_and_csv_config_name(self):
        module = _load_module('render_qualitative_figure', SCRIPT_PATH)
        paper_dir = self.work_dir / 'paper_M1-5'
        model_dir = paper_dir / 'M3'
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = model_dir / 'epoch_20.pth'
        checkpoint_path.write_bytes(b'checkpoint')
        log_path = self._write_experiment_log([
            {
                'experiment_id': 'M3',
                'config': '/content/drive/.../M3/wi_tidir_wifi_transformer.py',
                'checkpoint': '/content/drive/.../M3/latest.pth',
                'notes': 'flow transformer',
            },
        ])

        resolved = module.resolve_model_assets(
            project_root=ROOT,
            paper_dir=paper_dir,
            model_id='M3',
            specs=module.load_experiment_specs(log_path))

        self.assertEqual(resolved['checkpoint'], checkpoint_path)
        self.assertEqual(resolved['config'], ROOT / 'configs' / 'wifi' / 'wi_tidir_wifi_transformer.py')

    def test_panel_titles_follow_ground_truth_then_model_names(self):
        module = _load_module('render_qualitative_figure', SCRIPT_PATH)

        titles = module.build_panel_titles(['M0', 'M3', 'M4'])

        self.assertEqual(
            titles,
            ['Ground Truth', 'M0: Person-in-WiFi 3D', 'M3: FlowPose-WiFi (Transformer)', 'M4: FlowPose-WiFi (Ours)'])


if __name__ == '__main__':
    unittest.main()
