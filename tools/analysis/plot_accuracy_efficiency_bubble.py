#!/usr/bin/env python
"""Regenerate the full-paper accuracy-efficiency bubble chart."""

from plot_full_paper_figures import configure_matplotlib, draw_bubble_chart


if __name__ == '__main__':
    configure_matplotlib()
    for path in draw_bubble_chart():
        print(path)
