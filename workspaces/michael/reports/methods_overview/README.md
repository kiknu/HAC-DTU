# Asteroid lightcurve inversion: methods and results

- `methods_overview.pdf`: the 10-page presentation draft.
- `methods_overview.tex`: editable LaTeX source with embedded bibliography.
- `methods_overview_source.zip`: Overleaf source bundle; select `methods_overview.tex` as the main document. All figures and table fragments are included. No Python run or experiment data is needed to compile the document.
- `evidence.json`: hashes of scientific inputs and generated outputs, checkpoint/curve checks, selected iterations, display scales/views, renderer settings, and software versions.
- `validation.json` and `visual_review.json`: automated checks and a page-by-page visual review tied to the PDF hash.

The ten-asteroid L4 table includes verified saved 200-update fits listed in
`inputs.json`. An em dash means **no result reported**. Asteroids 4–10 use
calibration transferred from asteroid 3. The qualitative examples use
asteroids 2 and 6 at L4 with 200 updates, and asteroid 3 at L6 with 120
updates. Photometric agreement is not verification of the geometry.

## Compile the document

From this directory:

```sh
tectonic --keep-logs methods_overview.tex
```

Alternatively run `pdflatex -halt-on-error methods_overview.tex` twice.
Keep `figures/` and `tables/` beside the source. No BibTeX run is needed.

## Regenerate figures and tables

Regeneration requires the original project checkout, its Python environment,
and the saved artifacts and reference meshes identified in `evidence.json`.
The source bundle is self-contained for LaTeX compilation, not for repeating
the original fits. From the project root:

```sh
MPLCONFIGDIR=/tmp/hac-report-mpl .venv/bin/python reports/methods_overview/make_figure.py
```

Use the actual project environment path if `.venv` is elsewhere. Dependencies
include PyTorch/PyTorch3D with CPU rasterization, Matplotlib, NumPy, Pillow,
trimesh, and Embree. No GPU is used and no fit, calibration, experiment
configuration, or saved experiment artifact is modified.

The default reuses the asteroid list in `inputs.json`. To deliberately update
the list from available report manifests, pass `--refresh-inputs`. A reported
fit must pass checks of saved manifest hashes, selected checkpoint parameters,
rebuilt/exported meshes, observations, and RMSE recomputed from full-turn
curves. Other representation tables are generated from their saved metrics.

`data/image_formation_native.npz` preserves the floating-point 96×96 linear,
encoded, quantized, and thresholded arrays for two cameras at 45°. The six
96×96 PNGs are eight-bit previews. The panel figure uses a common 0–1 scale
and nearest-neighbor enlargement. The thresholded image retains intensity;
it is not a binary mask.

Geometry figures use CPU orthographic first-hit views with flat triangle
normals and illustrative lighting. They show the actual mesh triangles,
without display smoothing or geometric simplification. Public reconstruction
panels show the raw references. Initial and selected shapes receive only the
saved global scale alignment; view directions and framing match within each
comparison. The separate asteroid-3 proxy figure verifies the outermost-ray
construction against the saved 1,280-face mesh.

## Verify and package

Run verification in the original checkout so the scientific input hashes
can be checked. After recompiling, use a Python environment with PyMuPDF,
NumPy, and Pillow:

```sh
python verify_report.py
```

This verifies input/output hashes, table values, native pixel dimensions,
quantization/thresholding, page count, figure links, citations, and LaTeX
warnings, and renders every page to `/tmp/methods-overview-review/`.
Inspect every rendered page for equations, captions, labels, clipping, and
layout. Record the PDF SHA-256 and inspected page numbers in
`visual_review.json`; then run:

```sh
python verify_report.py --package
```

Packaging requires a visual-review record matching the exact PDF. It includes
only source, figures, table/data assets, regeneration scripts, and provenance;
it does not include fit checkpoints or raw reference meshes.
