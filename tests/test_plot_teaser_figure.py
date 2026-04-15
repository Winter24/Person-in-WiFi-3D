import csv
import importlib.util
import importlib
import shutil
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / 'tools' / 'analysis' / 'plot_teaser_figure.py'


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestPlotTeaserFigure(unittest.TestCase):
    def setUp(self):
        self.work_dir = ROOT / 'tests' / f'tmp_plot_teaser_{uuid.uuid4().hex}'
        self.work_dir.mkdir(parents=True, exist_ok=False)

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def _write_csv(self, rows):
        csv_path = self.work_dir / 'experiment_log.csv'
        with csv_path.open('w', newline='', encoding='utf-8') as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=['experiment_id', 'mpjpe', 'fps', 'params_m'])
            writer.writeheader()
            writer.writerows(rows)
        return csv_path

    def test_script_exists(self):
        self.assertTrue(
            SCRIPT_PATH.exists(),
            f'Missing script: {SCRIPT_PATH}')

    def test_load_experiment_rows_filters_missing_metrics(self):
        module = _load_module('plot_teaser_figure', SCRIPT_PATH)
        csv_path = self._write_csv([
            {'experiment_id': 'M0', 'mpjpe': '169.34', 'fps': '12.3', 'params_m': '13.13'},
            {'experiment_id': 'M5', 'mpjpe': '164.89', 'fps': '148.7', 'params_m': '5.11'},
            {'experiment_id': 'BROKEN', 'mpjpe': '', 'fps': '111.0', 'params_m': '5.1'},
        ])

        rows = module.load_experiment_rows(csv_path)

        self.assertEqual([row['experiment_id'] for row in rows], ['M0', 'M5'])
        self.assertEqual(rows[0]['display_name'], 'M0 (Baseline)')
        self.assertEqual(rows[1]['display_name'], 'M5 (Ours)')
        self.assertIsInstance(rows[0]['mpjpe'], float)
        self.assertIsInstance(rows[0]['fps'], float)
        self.assertIsInstance(rows[0]['params_m'], float)

    def test_load_experiment_rows_maps_legacy_ids_to_m_series(self):
        module = _load_module('plot_teaser_figure', SCRIPT_PATH)
        csv_path = self._write_csv([
            {'experiment_id': 'B0', 'mpjpe': '169.34', 'fps': '51.8', 'params_m': '13.13'},
            {'experiment_id': 'B1', 'mpjpe': '180.32', 'fps': '41.7', 'params_m': '13.36'},
            {'experiment_id': 'B2', 'mpjpe': '170.86', 'fps': '51.1', 'params_m': '11.25'},
            {'experiment_id': 'A3', 'mpjpe': '159.16', 'fps': '137.9', 'params_m': '7.22'},
            {'experiment_id': 'B4', 'mpjpe': '166.51', 'fps': '109.3', 'params_m': '5.11'},
            {'experiment_id': 'B5', 'mpjpe': '164.89', 'fps': '154.4', 'params_m': '5.11'},
        ])

        rows = module.load_experiment_rows(
            csv_path,
            runs=['M0', 'M1', 'M2', 'M3', 'M4', 'M5'],
            alias_path=module.DEFAULT_ALIAS_PATH)

        self.assertEqual(
            [row['experiment_id'] for row in rows],
            ['M0', 'M1', 'M2', 'M3', 'M4', 'M5'])
        self.assertEqual(rows[0]['source_experiment_id'], 'B0')
        self.assertEqual(rows[3]['source_experiment_id'], 'A3')
        self.assertEqual(rows[5]['display_name'], 'M5 (Ours)')

    def test_bubble_size_scale_preserves_parameter_order(self):
        module = _load_module('plot_teaser_figure', SCRIPT_PATH)
        rows = [
            {'experiment_id': 'M0', 'params_m': 13.13},
            {'experiment_id': 'M2', 'params_m': 11.25},
            {'experiment_id': 'M5', 'params_m': 5.11},
        ]

        sizes = module.bubble_size_scale(rows)

        self.assertGreater(sizes['M0'], sizes['M2'])
        self.assertGreater(sizes['M2'], sizes['M5'])

    def test_annotation_spec_moves_m0_label_left_to_avoid_overlap(self):
        module = _load_module('plot_teaser_figure', SCRIPT_PATH)

        spec = module.annotation_spec('M0', highlight='M4')

        self.assertLess(spec['dx'], 0.0)
        self.assertEqual(spec['ha'], 'right')
        self.assertEqual(spec['fontweight'], 'normal')

    def test_plot_teaser_figure_writes_publication_outputs_and_inverts_y_axis(self):
        if importlib.util.find_spec('matplotlib') is None:
            self.skipTest('matplotlib is not available in this environment')
        module = _load_module('plot_teaser_figure', SCRIPT_PATH)
        rows = [
            {'experiment_id': 'M0', 'display_name': 'M0 (Baseline)', 'mpjpe': 169.34, 'fps': 12.3, 'params_m': 13.13},
            {'experiment_id': 'M1', 'display_name': 'M1', 'mpjpe': 180.32, 'fps': 28.0, 'params_m': 13.36},
            {'experiment_id': 'M2', 'display_name': 'M2', 'mpjpe': 170.86, 'fps': 65.0, 'params_m': 11.25},
            {'experiment_id': 'M4', 'display_name': 'M4', 'mpjpe': 166.51, 'fps': 121.0, 'params_m': 5.11},
            {'experiment_id': 'M5', 'display_name': 'M5 (Ours)', 'mpjpe': 164.89, 'fps': 156.4, 'params_m': 5.11},
        ]

        output_prefix = self.work_dir / 'figure1_teaser'
        fig, ax = module.plot_teaser_figure(rows, output_prefix, xmax=200, highlight='M5')

        self.assertTrue(ax.yaxis_inverted())
        self.assertEqual(tuple(round(v, 1) for v in ax.get_xlim()), (0.0, 200.0))
        for suffix in ('.png', '.pdf', '.svg'):
            self.assertTrue((self.work_dir / f'figure1_teaser{suffix}').exists())
        self.assertNotIn('Better', ax.texts[-1].get_text() if ax.texts else '')
        all_text = ' '.join(text.get_text() for text in ax.texts)
        self.assertNotIn('Better', all_text)
        self.assertIn('M5 (Ours)', all_text)
        fig.clf()


if __name__ == '__main__':
    unittest.main()
