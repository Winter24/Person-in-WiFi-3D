# RESFES 2026 WiTiDAR Manuscript Package

This folder contains the LaTeX manuscript source, references, tables, and figures for the internal RESFES 2026 WiFi 3D pose paper draft.

Official title:

**Flow is All You Need for WiFi: Rectifying 3D Human Poses without Complex Architectures**

Main files:

- `main.tex`: source-of-truth manuscript content.
- `main.rev2.tex`: thin revision-2 wrapper that inputs `main.tex`.
- `references.bib`: bibliography used by `main.tex`.
- `references.rev2.bib`: revision-2 copy of the bibliography.
- `figures/`: copied paper figures and slide-derived visual assets.
- `tables/`: generated table fragments and Markdown source tables.

Compile the source-of-truth PDF from this folder:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

If a venue requires the revision wrapper filename, compile `main.rev2.tex` with `bibtex main.rev2`; the body content is still read from `main.tex`.

Submission source should include only the manuscript sources and assets required to build the paper: `main.tex`, `main.rev2.tex`, `references.bib`, table fragments used by the manuscript, figure files referenced by the manuscript, and the final PDF. Do not package LaTeX build intermediates (`.aux`, `.bbl`, `.blg`, `.fdb_latexmk`, `.fls`, `.log`, `.out`, `.synctex.gz`), local tool settings, internal runbooks, or response placeholders.
