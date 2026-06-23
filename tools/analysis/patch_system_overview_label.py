#!/usr/bin/env python
"""Normalize the selected-model label in the paper system overview."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
FIGURE = (
    ROOT
    / 'paper_assets'
    / 'manuscript_latex'
    / 'resfes2026_witidar'
    / 'figures'
    / 'fig_system_overview.png'
)


def main():
    image = Image.open(FIGURE).convert('RGB')
    if image.size != (1075, 601):
        raise ValueError(f'Unexpected system-overview size: {image.size}')

    draw = ImageDraw.Draw(image)
    background = (225, 213, 231)
    draw.rectangle((746, 500, 812, 509), fill=(245, 245, 245))
    draw.line((779, 493, 779, 505), fill=(0, 0, 0), width=2)
    draw.polygon(((775, 504), (783, 504), (779, 510)), fill=(0, 0, 0))
    draw.rectangle((690, 510, 860, 548), fill=background)

    title_font = ImageFont.truetype(r'C:\Windows\Fonts\arialbd.ttf', 12)
    draw.text(
        (775, 516),
        'Euler Integration (2-step, M9_RF2)',
        fill=(0, 0, 0),
        font=title_font,
        anchor='ma',
    )
    image.save(FIGURE, dpi=(300, 300))
    print(FIGURE)


if __name__ == '__main__':
    main()
