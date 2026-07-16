import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / 'tools' / 'analysis' / 'plot_full_paper_figures.py'


def load_module():
    spec = importlib.util.spec_from_file_location(
        'plot_full_paper_figures', SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestFullPaperFigureLabels(unittest.TestCase):
    def test_dense_figure_uses_functional_short_labels(self):
        module = load_module()

        self.assertEqual(module.public_short_labels()[0], 'PETR\nRef.')
        self.assertEqual(module.public_short_labels()[-1], 'Mamba-2\nFlow 2')


if __name__ == '__main__':
    unittest.main()
