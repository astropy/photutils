# Photutils Benchmarks

This directory contains standalone benchmark scripts for photutils.
They are not part of the installed package or the test suite. Helper
functions shared by the scripts live in `bench_utils.py`.

## Aperture statistics (`bench_aperture_stats.py`)

Benchmarks for `ApertureStats`:

- computing the median and all properties for all of the pixel-based
  aperture types, with and without sigma clipping
- the cold cost of each individual property (including its lazy
  dependencies) for a circular aperture, sorted by decreasing
  sigma-clipped time

```bash
python benchmarks/bench_aperture_stats.py
python benchmarks/bench_aperture_stats.py --which types --n-sources 10000
```

## Aperture photometry (`bench_aperture_photometry.py`)

Benchmarks for `AperturePhotometry`: computing the flux (with an
error array) for all of the pixel-based aperture types and each
overlap method (exact, center, and subpixel).

```bash
python benchmarks/bench_aperture_photometry.py
python benchmarks/bench_aperture_photometry.py --n-sources 100000
```

## Centroids (`bench_centroids.py`)

Benchmarks for the `photutils.centroids` subpackage:

- the per-call cost of the single-source centroid functions
  (`centroid_com`, `centroid_quadratic`, `centroid_1dg`, and
  `centroid_2dg`), with and without an error array
- `centroid_sources` at many positions for each centroid function.

```bash
python benchmarks/bench_centroids.py
python benchmarks/bench_centroids.py --which sources \
    --n-sources 5000 --n-threads 1,4,8
```

## Geometry (`bench_geometry.py`)

Benchmarks for the `photutils.geometry` subpackage:

- the per-call cost of the overlap grid functions
  (`circular_overlap_grid`, `elliptical_overlap_grid`,
  `rectangular_overlap_grid`, and `polygon_overlap_grid`) for each
  overlap method (exact, center, and subpixel)
- the scaling of the exact mode with grid size
- the scaling of `polygon_overlap_grid` with the number of vertices,
  for convex and non-convex (star-shaped) polygons

```bash
python benchmarks/bench_geometry.py
python benchmarks/bench_geometry.py --which polygon --n-vertices 8,64,512
```

## Background (`bench_background.py`)

Benchmarks for the `photutils.background` subpackage:

- `Background2D` construction across image sizes, box sizes, and
  `n_threads` values (with speedups relative to the first
  `n_threads` value)
- full-size background and background RMS map generation
- the scalar background and background RMS estimator classes, with
  and without sigma clipping
- `LocalBackground` at many positions

Examples:

```bash
# Run everything with the default settings
python benchmarks/bench_background.py

# Only the Background2D n_threads scaling benchmark
python benchmarks/bench_background.py --which background2d \
    --sizes 2048,4096 --box-sizes 64 --n-threads 1,2,4,8
```
