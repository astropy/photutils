# Photutils Benchmarks

This directory contains standalone benchmark scripts for photutils.
They are not part of the installed package or the test suite. Helper
functions shared by the scripts live in `bench_helpers.py`.

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

## Morphology (`bench_morphology.py`)

Benchmarks for the `photutils.morphology` subpackage:

- the per-call cost of `data_properties`: catalog construction alone
  and with the morphological properties computed, with and without a
  mask, background, and WCS
- the scaling of `gini` with array size, with and without a mask

```bash
python benchmarks/bench_morphology.py
python benchmarks/bench_morphology.py --which gini --sizes 256,1024,4096
```

## Profiles (`bench_profiles.py`)

Benchmarks for the `photutils.profiles` subpackage:

- the per-call cost of constructing each profile class
  (`RadialProfile`, `CurveOfGrowth`, `EnsquaredCurveOfGrowth`, and
  `EllipticalCurveOfGrowth`) and computing its profile, errors, and
  areas, for each overlap method (exact, center, and subpixel)
- the scaling of the profile computation with the number of radial
  bins
- the cost of the extra lazily-computed `RadialProfile` attributes
  (`data_profile`, `gaussian_fit`, and `moffat_fit`)

```bash
python benchmarks/bench_profiles.py
python benchmarks/bench_profiles.py --which radii-scaling --n-radii-list 100,400,1600
```

## PSF matching (`bench_psf_matching.py`)

Benchmarks for the `photutils.psf_matching` subpackage:

- the per-call cost of the kernel-making functions (`make_kernel`
  with and without a window, and `make_wiener_kernel` for the
  scalar, Laplacian, and biharmonic penalties)
- the scaling of the kernel computation with the PSF size
- the per-call cost of the window classes
- `resize_psf` for down- and upsampling with each spline order

```bash
python benchmarks/bench_psf_matching.py
python benchmarks/bench_psf_matching.py --which size-scaling --sizes 101,201,401
```

## Datasets (`bench_datasets.py`)

Benchmarks for the `photutils.datasets` subpackage:

- the scaling of `make_model_image` with the number of sources
- `make_model_image` for each discretization method (`center`,
  `interp`, and `oversample`)
- the scaling of `make_model_params` with the number of sources,
  including the minimum-separation (KDTree) filtering
- the noise functions (`make_noise_image` for each distribution and
  `apply_poisson_noise`)
- the WCS factories (`make_wcs` and `make_gwcs`)
- the example-image functions (`make_4gaussians_image` and
  `make_100gaussians_image`)

```bash
python benchmarks/bench_datasets.py
python benchmarks/bench_datasets.py --which model-image --n-sources-list 500,2000
```

## Detection (`bench_detection.py`)

Benchmarks for the `photutils.detection` subpackage:

- the star finder classes (`DAOStarFinder`, `IRAFStarFinder`, and
  `StarFinder`) for full detection runs and, for the DAOFIND-style
  finders, catalog-only runs with the source positions given via
  `xycoords`
- the `find_peaks` detection modes (`box_size`, `footprint`,
  `min_separation`, and centroiding)
- the fast circular (`min_separation`) peak detection versus the
  equivalent circular-footprint maximum filter (with speedups)
- concurrent `find_stars` calls on a shared `DAOStarFinder` instance
  across thread counts (with speedups relative to the first thread
  count)

Examples:

```bash
# Run everything with the default settings
python benchmarks/bench_detection.py

# Only the min-separation speedup benchmark
python benchmarks/bench_detection.py --which min-separation --radii 10,50
```

## Segmentation (`bench_segmentation.py`)

Benchmarks for the `photutils.segmentation` subpackage, using an
image of blended Gaussian-source pairs so that deblending has work to
do:

- source detection (`detect_threshold` and `detect_sources` with
  4- and 8-connectivity)
- source deblending (`deblend_sources`) across the threshold modes
  (`linear`, `exponential`, and `sinh`) and `n_processes` values
- the combined `SourceFinder` class with and without deblending
- `SegmentationImage` operations (cached properties, relabeling,
  border-label removal, square and circular source masks, and
  polygons)
- all `SourceCatalog` properties (from the catalog `properties`
  attribute, computed with convolved data, error, background, and
  WCS inputs) plus the method-based measurements (`flux_radius`,
  `circular_photometry`, `kron_photometry`, and `to_table`), each on
  a fresh catalog (times include dependent properties, e.g.,
  `centroid_win` includes the Kron flux and `flux_radius` it depends
  on)
- concurrent `SourceCatalog` measurement jobs across thread counts
  (with speedups relative to the first thread count)

Examples:

```bash
# Run everything with the default settings
python benchmarks/bench_segmentation.py

# Only the deblending benchmark with a larger image
python benchmarks/bench_segmentation.py --which deblend \
    --n-sources 4000 --n-processes 1,4,8
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

## Utils (`bench_utils.py`)

Benchmarks for the `photutils.utils` subpackage:

- `ImageDepth` versus the number of apertures, in the overlapping
  and non-overlapping modes
- concurrent `ImageDepth` calls on a shared instance across thread
  counts (with speedups relative to the first thread count)
- `ShepardIDWInterpolator` construction and evaluation versus the
  numbers of data points, query positions, and neighbors
- cutout generation with `_make_cutouts` and `CutoutImage`
- `calc_total_error` versus image size, with scalar and 2D gains
- the NaN-ignoring statistics functions on float64 (bottleneck) and
  float32 (NumPy) arrays
- `make_random_xycoords` with and without a minimum separation
- the per-call cost of the local WCS helper functions

Examples:

```bash
# Run everything with the default settings
python benchmarks/bench_utils.py

# Only the ImageDepth thread-scaling benchmark
python benchmarks/bench_utils.py --which depth-threads --threads 1,4,8
```
