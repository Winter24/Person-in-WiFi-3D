# RESFES 2026 WiTiDAR Manuscript Package

This folder contains the LaTeX manuscript source, references, tables, and figures for the internal RESFES 2026 WiFi 3D pose paper draft.

Main files:

- `main.tex`: current manuscript source.
- `main.rev2.tex`: revision-2 copy of the manuscript source.
- `references.bib`: bibliography used by `main.tex`.
- `references.rev2.bib`: revision-2 copy of the bibliography.
- `figures/`: copied paper figures and slide-derived visual assets.
- `tables/`: generated table fragments and Markdown source tables.

Compile from this folder:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Auxiliary LaTeX build files are intentionally not tracked.
