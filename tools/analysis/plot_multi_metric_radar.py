#!/usr/bin/env python
"""Regenerate the full-paper multi-metric radar chart."""

from plot_full_paper_figures import configure_matplotlib, draw_radar_chart


if __name__ == '__main__':
    configure_matplotlib()
    for path in draw_radar_chart():
        print(path)
