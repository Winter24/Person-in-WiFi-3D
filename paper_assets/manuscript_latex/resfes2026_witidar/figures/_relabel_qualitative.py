"""Relabel slide page 27 of flow_slides.pdf:
M3 (Accuracy) -> M7 (Transformer + Flow)
M4 (Efficiency) -> M9_RF2 (Selected)

Output: fig_qualitative_relabeled.png
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
SRC_PDF = HERE / "flow_slides.pdf"
OUT_PNG = HERE / "fig_qualitative_relabeled.png"
RENDER_DPI = 220

# --- 1. Render page 27 of the slide deck to a high-resolution image ---
try:
    import fitz  # PyMuPDF
except ImportError:
    print("Need PyMuPDF: pip install pymupdf", file=sys.stderr)
    sys.exit(1)

doc = fitz.open(str(SRC_PDF))
page = doc[26]  # page 27 (0-indexed)
zoom = RENDER_DPI / 72.0
mat = fitz.Matrix(zoom, zoom)
pix = page.get_pixmap(matrix=mat, alpha=False)
src_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
W, H = src_img.size
print(f"Rendered slide 27 at {W}x{H}")

# --- 2. Cover the original "M3 (Accuracy)" and "M4 (Efficiency)" headers ---
draw = ImageDraw.Draw(src_img)

# Coordinates determined from slide layout (5 columns, headers in top row of grid):
#   col1 (labels):      x = 0.060..0.135
#   col2 (Ground Truth) x = 0.135..0.355
#   col3 (M0 Baseline)  x = 0.355..0.575
#   col4 (M3 Accuracy)  x = 0.575..0.785
#   col5 (M4 Efficiency)x = 0.785..0.985
# Headers Y range:      y = 0.165..0.255 (table top row)

def rect_fill(x0_pct, y0_pct, x1_pct, y1_pct, color="white"):
    draw.rectangle(
        [int(x0_pct * W), int(y0_pct * H), int(x1_pct * W), int(y1_pct * H)],
        fill=color,
    )

rect_fill(0.575, 0.148, 0.785, 0.262)  # cover M3 (Accuracy)
rect_fill(0.785, 0.148, 0.985, 0.262)  # cover M4 (Efficiency)

# --- 3. Draw replacement labels (two-line, fits column width) ---
font_size = int(0.034 * H)
try:
    font_bold = ImageFont.truetype("arialbd.ttf", font_size)
except OSError:
    font_bold = ImageFont.truetype("arial.ttf", font_size)

def centered_block(lines, cx_pct, cy_pct, font, color="black", line_gap=0.25):
    """Draw multi-line text centered at (cx_pct, cy_pct)."""
    sizes = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        sizes.append((bbox[2] - bbox[0], bbox[3] - bbox[1], bbox[1]))
    line_h = max(h for _, h, _ in sizes)
    total_h = line_h * len(lines) + line_gap * line_h * (len(lines) - 1)
    cy = cy_pct * H
    cx = cx_pct * W
    y0 = cy - total_h / 2
    for i, (line, (w, h, b)) in enumerate(zip(lines, sizes)):
        x = int(cx - w / 2)
        y = int(y0 + i * (line_h * (1 + line_gap)) - b)
        draw.text((x, y), line, fill=color, font=font)

centered_block(["M7", "(Transformer + Flow)"], 0.680, 0.205, font_bold)
centered_block(["M9_RF2", "(Mamba2 + Flow)"],  0.885, 0.205, font_bold)

# --- 4. Save ---
src_img.save(str(OUT_PNG), "PNG", optimize=True)
print(f"Saved {OUT_PNG} ({OUT_PNG.stat().st_size/1024:.0f} KB)")
