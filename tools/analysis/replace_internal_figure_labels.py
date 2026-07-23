#!/usr/bin/env python3
"""Replace implementation-specific labels in a vector PDF figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz


LABELS = ("VelocityMLP", "Velocity MLP")
REPLACEMENT = "Velocity field"
INK = (47 / 255, 66 / 255, 136 / 255)


def replace_labels(input_path: Path, output_path: Path) -> int:
    document = fitz.open(input_path)
    replacements = 0

    for page in document:
        matches = []
        for label in LABELS:
            matches.extend(page.search_for(label))

        for rect in matches:
            page.add_redact_annot(rect + (-2, -1, 2, 1), fill=(1, 1, 1))
        if matches:
            page.apply_redactions()

        for rect in matches:
            fontsize = min(14.35, rect.height * 0.81)
            text_width = fitz.get_text_length(
                REPLACEMENT, fontname="hebo", fontsize=fontsize
            )
            x = rect.x0 + (rect.width - text_width) / 2
            baseline = rect.y1 - rect.height * 0.19
            page.insert_text(
                (x, baseline),
                REPLACEMENT,
                fontsize=fontsize,
                fontname="hebo",
                color=INK,
                overlay=True,
            )
            replacements += 1

    if replacements == 0:
        document.close()
        raise ValueError(f"No internal labels found in {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path, garbage=4, deflate=True)
    document.close()
    return replacements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    count = replace_labels(args.input, args.output)
    print(f"Replaced {count} labels in {args.output}")


if __name__ == "__main__":
    main()
