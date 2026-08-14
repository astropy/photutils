# Photutils Benchmarks

This directory contains standalone benchmark scripts for photutils.
They are not part of the installed package or the test suite.

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
