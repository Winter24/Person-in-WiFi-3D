import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / 'tools' / 'analysis' / 'render_system_overview.py'


def load_module():
    spec = importlib.util.spec_from_file_location('render_system_overview', SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestRenderSystemOverview(unittest.TestCase):
    def test_renderer_creates_vector_and_raster_schematic_with_public_labels(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs = module.render_system_overview(Path(tmp_dir) / 'system_overview')

            self.assertTrue(outputs['pdf'].exists())
            self.assertTrue(outputs['png'].exists())
            self.assertGreater(outputs['pdf'].stat().st_size, 500)
            self.assertIn('Mamba-2 Flow (2 steps)', module.SELECTED_MODEL_LABEL)
            self.assertIn('180 tokens', module.STAGE_LABELS['spectral'])


if __name__ == '__main__':
    unittest.main()
