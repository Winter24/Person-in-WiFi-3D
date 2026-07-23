# RESFES 2026 WiTiDAR Manuscript Package

This folder contains the LaTeX manuscript source, references, tables, and figures for the internal RESFES 2026 WiFi 3D pose paper draft.

Official title:

**Flow is All You Need for WiFi: Rectifying 3D Human Poses without Complex Architectures**

Compile package:

- `main.tex`: source-of-truth manuscript content.
- `references.bib`: bibliography used by `main.tex`.
- `figures/`: figure assets referenced by `main.tex`.
- `tables/`: LaTeX table fragments referenced by `main.tex`.
- `main.pdf`: compiled manuscript.

Compile the source-of-truth PDF from this folder:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Run the tokenizer component diagnostic from the repository root on the GPU server:

```bash
bash tools/analysis/run_spectral_tokenizer_ablation.sh all
```

The runner trains and evaluates T0--T4, benchmarks each checkpoint, checks that
the T0 and T4 endpoints reproduce the current M0 and M1 results within 1 mm,
and writes `tables/tokenizer_component_ablation.tex` only after that audit
passes. Rebuild `main.pdf` after copying the validated artifacts back to this
workspace. The current PDF intentionally omits the conditional component table
until those server results exist.

Submission source should include only `main.tex`, `references.bib`, the table fragments and figure files referenced by the manuscript, and the final PDF. Do not package LaTeX build intermediates (`.aux`, `.bbl`, `.blg`, `.fdb_latexmk`, `.fls`, `.log`, `.out`, `.synctex.gz`), local tool settings, internal runbooks, or response placeholders.
