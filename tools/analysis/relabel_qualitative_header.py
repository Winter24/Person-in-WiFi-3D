#!/usr/bin/env python
"""Relabel one qualitative-figure header without rerunning model inference."""

import argparse
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analysis.model_labels import display_label
from tools.analysis.model_palette import get_model_color


DEFAULT_FIGURE = (
    ROOT / 'paper_assets' / 'manuscript_latex' / 'resfes2026_witidar'
    / 'figures' / 'fig_qualitative_full_paper.png'
)


def _font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _centered_text(draw, center_x, y, text, font, fill='#111111'):
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    draw.text((center_x - width / 2, y), text, font=font, fill=fill)


def relabel_header(source_path, output_path, pdf_path=None, model_id='M0',
                   column_index=2, column_count=5, header_height=None):
    """Replace a model column header while preserving all content below it."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    image = Image.open(source_path).convert('RGB')
    width, height = image.size
    header_height = header_height or round(height * 0.049)
    if not 0 < header_height < height:
        raise ValueError(f'Invalid header height: {header_height}')
    if not 0 <= column_index < column_count:
        raise ValueError(f'Invalid column index: {column_index}')

    column_width = width / column_count
    left = round(column_index * column_width)
    right = round((column_index + 1) * column_width)
    center_x = (left + right) / 2
    draw = ImageDraw.Draw(image)
    draw.rectangle((left, 0, right, header_height - 1), fill='white')

    id_font = _font(max(10, round(header_height * 0.27)))
    label_font = _font(max(11, round(header_height * 0.31)))
    label = display_label(model_id)
    _centered_text(draw, center_x, round(header_height * 0.05),
                   f'{model_id}:', id_font)
    _centered_text(draw, center_x, round(header_height * 0.39),
                   label, label_font)
    line_y = round(header_height * 0.87)
    half_line = round(column_width * 0.12)
    draw.line((center_x - half_line, line_y, center_x + half_line, line_y),
              fill=get_model_color(model_id), width=max(2, round(height / 1000)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, dpi=(300, 300))
    if pdf_path is not None:
        pdf_path = Path(pdf_path)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(pdf_path, 'PDF', resolution=300.0)
    return {
        'png': output_path,
        'pdf': Path(pdf_path) if pdf_path is not None else None,
        'label': label,
        'header_height': header_height,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Relabel the accepted qualitative figure without inference.')
    parser.add_argument('--source', default=str(DEFAULT_FIGURE))
    parser.add_argument('--output', default=str(DEFAULT_FIGURE))
    parser.add_argument(
        '--pdf',
        default=str(DEFAULT_FIGURE.with_suffix('.pdf')),
    )
    parser.add_argument('--model-id', default='M0')
    return parser.parse_args()


def main():
    args = parse_args()
    result = relabel_header(
        args.source,
        args.output,
        pdf_path=args.pdf,
        model_id=args.model_id,
    )
    print(result['png'])
    print(result['pdf'])


if __name__ == '__main__':
    main()
