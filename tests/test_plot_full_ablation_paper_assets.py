import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / 'tools' / 'analysis' / 'plot_full_ablation_paper_assets.py'


def load_module():
    spec = importlib.util.spec_from_file_location(
        'plot_full_ablation_paper_assets', SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestFlowSolverFigure(unittest.TestCase):
    VALUES = {
        'M6': (167.573, 54),
        'M9_no_flow': (167.732, 82),
        'M9': (168.086, 76),
        'M9_RF2': (165.487, 72),
        'T_FW2_20e_RF2': (167.981, 86),
        'T_FW2_20e_RF4': (169.454, 99),
    }

    def write_eval_files(self, eval_dir):
        for experiment_id, (mpjpe, missed_persons) in self.VALUES.items():
            payload = {
                'mpjpe': mpjpe,
                'missed_persons': missed_persons,
            }
            (eval_dir / f'{experiment_id}_eval.json').write_text(
                json.dumps(payload), encoding='utf-8')

    def test_build_flow_plot_rows_preserves_order_labels_and_values(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            eval_dir = Path(temp_dir)
            self.write_eval_files(eval_dir)

            rows = module.build_flow_plot_rows({}, eval_dir)

        self.assertEqual(
            [(row['display_label'], row['experiment_id']) for row in rows],
            [
                ('Mamba-2 Draft', 'M6'),
                ('Mamba-2 Flow (refinement bypassed)', 'M9_no_flow'),
                ('Mamba-2 Flow (1 step)', 'M9'),
                ('Mamba-2 Flow (2 steps)', 'M9_RF2'),
            ])
        self.assertEqual(
            [row['mpjpe'] for row in rows],
            [167.573, 167.732, 168.086, 165.487])
        self.assertEqual(
            [row['missed_persons'] for row in rows],
            [54, 82, 76, 72])
        self.assertEqual(
            [row['experiment_id'] for row in rows if row['selected']],
            ['M9_RF2'])
        palette = __import__(
            'tools.analysis.model_palette',
            fromlist=['FLOW_SETTING_STYLES'])
        self.assertEqual(
            [row['style'] for row in rows],
            [palette.FLOW_SETTING_STYLES[row['experiment_id']] for row in rows])

    def test_cli_supports_regenerating_only_flow_figure(self):
        module = load_module()
        original_argv = sys.argv
        try:
            sys.argv = ['plot_full_ablation_paper_assets.py', '--figures', 'flow']
            args = module.parse_args()
        finally:
            sys.argv = original_argv
        self.assertEqual(args.figures, ['flow'])

    def test_plot_flow_ablation_exports_publication_pdf_and_png(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            eval_dir = temp_path / 'eval'
            eval_dir.mkdir()
            self.write_eval_files(eval_dir)
            output_base = temp_path / 'fig_flow_solver_ablation'

            outputs = module.plot_flow_ablation({}, eval_dir, output_base)

            self.assertEqual(set(outputs), {'pdf', 'png'})
            self.assertEqual(outputs['pdf'], output_base.with_suffix('.pdf'))
            self.assertEqual(outputs['png'], output_base.with_suffix('.png'))
            self.assertGreater(outputs['pdf'].stat().st_size, 1000)
            self.assertGreater(outputs['png'].stat().st_size, 1000)
            with Image.open(outputs['png']) as image:
                self.assertGreaterEqual(image.width, 2160)
                self.assertGreaterEqual(image.height, 900)

    def test_legacy_pdf_export_works_without_reportlab(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            png_path = temp_path / 'source.png'
            pdf_path = temp_path / 'output.pdf'
            Image.new('RGB', (120, 80), 'white').save(png_path)

            module._write_pdf_from_png(png_path, pdf_path)

            self.assertTrue(pdf_path.exists())
            self.assertGreater(pdf_path.stat().st_size, 500)


if __name__ == '__main__':
    unittest.main()
