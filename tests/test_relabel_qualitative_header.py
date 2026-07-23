import tempfile
import unittest
from pathlib import Path

from PIL import Image


class TestRelabelQualitativeHeader(unittest.TestCase):
    def test_relabel_preserves_all_pixels_below_header(self):
        from tools.analysis.relabel_qualitative_header import relabel_header

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            source = tmp_dir / 'source.png'
            output = tmp_dir / 'output.png'
            pdf = tmp_dir / 'output.pdf'
            image = Image.new('RGB', (1000, 500), '#F4F4F4')
            for y in range(80, 500):
                for x in range(1000):
                    image.putpixel((x, y), ((x + y) % 255, y % 255, x % 255))
            image.save(source)

            result = relabel_header(
                source,
                output,
                pdf_path=pdf,
                model_id='M0',
                header_height=80,
            )

            before = Image.open(source).convert('RGB')
            after = Image.open(output).convert('RGB')
            self.assertEqual(before.size, after.size)
            self.assertEqual(
                before.crop((0, 80, 1000, 500)).tobytes(),
                after.crop((0, 80, 1000, 500)).tobytes(),
            )
            self.assertNotEqual(
                before.crop((400, 0, 600, 80)).tobytes(),
                after.crop((400, 0, 600, 80)).tobytes(),
            )
            self.assertEqual(result['label'], 'Transformer-PETR Control')
            self.assertTrue(pdf.exists())


if __name__ == '__main__':
    unittest.main()
