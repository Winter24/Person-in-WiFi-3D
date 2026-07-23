import importlib.util
import tempfile
import unittest
from pathlib import Path

import fitz
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / 'tools' / 'analysis' / 'render_problem_formulation.py'


def load_module():
    spec = importlib.util.spec_from_file_location(
        'render_problem_formulation', SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestRenderProblemFormulation(unittest.TestCase):
    def test_renderer_creates_readable_vector_and_raster_three_panel_figure(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs = module.render_problem_formulation(
                Path(tmp_dir) / 'problem_formulation')

            self.assertTrue(outputs['pdf'].exists())
            self.assertTrue(outputs['png'].exists())
            self.assertGreater(outputs['pdf'].stat().st_size, 1_000)

            document = fitz.open(outputs['pdf'])
            try:
                text = '\n'.join(page.get_text() for page in document)
            finally:
                document.close()
            normalized_text = ' '.join(text.split())

            for label in [
                'Multi-person WiFi sensing',
                'CSI tensorization',
                'Permutation-invariant pose set',
                '1 TX',
                '3 RX',
                '30 subcarriers',
                '20 packets',
                '180 WiFi tokens',
                'Q = 100',
                'M = 1-3',
                '14 x 3',
                'Hungarian one-to-one',
                'no-person',
                'Kinect targets only',
            ]:
                with self.subTest(label=label):
                    self.assertIn(label, normalized_text)

            for redundant_label in [
                'INFERENCE',
                '1st',
                '2nd',
                '3rd',
                'Mamba-2',
                'Rectified Flow',
            ]:
                with self.subTest(redundant_label=redundant_label):
                    self.assertNotIn(redundant_label, normalized_text)

            with Image.open(outputs['png']) as image:
                self.assertGreaterEqual(image.width, 2_000)
                self.assertGreaterEqual(image.height, 760)


if __name__ == '__main__':
    unittest.main()
