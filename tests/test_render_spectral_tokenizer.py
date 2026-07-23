import importlib.util
import tempfile
import unittest
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "analysis" / "render_spectral_tokenizer.py"


class SpectralTokenizerRendererTests(unittest.TestCase):
    def test_renderer_exports_vector_pdf_and_png_with_required_labels(self):
        spec = importlib.util.spec_from_file_location("render_spectral_tokenizer", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = Path(tmpdir) / "fig_spectral_tokenizer"
            pdf_path, png_path = module.render_spectral_tokenizer(output_base)

            self.assertTrue(Path(pdf_path).is_file())
            self.assertTrue(Path(png_path).is_file())
            document = fitz.open(pdf_path)
            try:
                text = "\n".join(page.get_text() for page in document)
            finally:
                document.close()

        for label in (
            "regroup: B x 9 x",
            "20 x 60",
            "60 -> 256",
            "20-point RFFT",
            "11 bins",
            "11 -> 22",
            "22 -> 20",
            "gs",
            "W_c",
            "B x 180 x 256",
            "LayerNorm(Z), not Z",
        ):
            with self.subTest(label=label):
                self.assertIn(label, text)


if __name__ == "__main__":
    unittest.main()
