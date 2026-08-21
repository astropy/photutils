#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Validate SourceCatalog results across photutils versions.

The ``dump`` command computes every array-valued ``SourceCatalog``
property (plus the ``flux_radius``, ``circular_photometry``,
``kron_photometry``, and default ``to_table`` outputs) for two
deterministic scenes and several catalog configurations, using
whichever photutils is importable, and writes them to a file. The
``compare`` command loads two such files (e.g., from a released
version and a development branch) and reports every entry that
differs, grouped by property, along with the properties that exist in
only one of the versions.

The scenes are the blended Gaussian-pair image used by the
``SourceCatalog`` benchmarks in ``bench_segmentation.py`` (with error,
background, mask, and WCS inputs) and a small edge-case scene with
close source pairs, sources touching every image edge, masked pixels
spanning a close pair, and non-finite data values. The configurations
cover the three ``aperture_mask_method`` values, a bare catalog, a
local background, the minimum-circular-radius Kron aperture, sliced
and scalar catalogs, and a ``detection_catalog``.

The ``dump`` command uses only APIs available in photutils 3.0.0, so
it can be run against that release and later. Run it once per
photutils installation (e.g., with ``PYTHONPATH`` pointing at an
unpacked wheel, from a directory that does not contain a photutils
checkout), then compare::

    python benchmarks/validate_catalog_versions.py dump old.pkl
    python benchmarks/validate_catalog_versions.py dump new.pkl
    python benchmarks/validate_catalog_versions.py compare old.pkl new.pkl

The dump files are pickles; only load files that you created.

Run ``python benchmarks/validate_catalog_versions.py --help`` to see
the available options.
"""

import argparse
import pickle
import sys
import warnings
from collections import defaultdict

import astropy.units as u
import numpy as np
from astropy.convolution import convolve
from astropy.coordinates import SkyCoord
from astropy.modeling.models import Gaussian2D
from astropy.stats import gaussian_fwhm_to_sigma

import photutils
from photutils.datasets import make_wcs
from photutils.segmentation import (SourceCatalog, detect_sources,
                                    make_2dgaussian_kernel)


def make_bench_scene(*, n_sources=300, spacing=25, fwhm=4.0, offset=6.0,
                     seed=0):
    """
    Build the blended Gaussian-pair scene of the SourceCatalog
    benchmarks, with error, background, mask, and WCS inputs.

    This replicates ``bench_segmentation.make_source_image`` and
    ``make_inputs`` without importing them, so that the dump can run
    against any photutils installation.

    Parameters
    ----------
    n_sources : int, optional
        The total number of Gaussian sources.

    spacing : int, optional
        The grid cell size in pixels.

    fwhm : float, optional
        The FWHM of the Gaussian sources in pixels.

    offset : float, optional
        The separation of the two sources within a cell in pixels.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    scene : dict
        The ``data``, ``convolved_data``, ``segm``, ``error``,
        ``background``, ``mask``, and ``wcs`` inputs.
    """
    n_cells = (n_sources + 1) // 2
    n_grid = int(np.ceil(np.sqrt(n_cells)))
    size = n_grid * spacing
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, (size, size))
    sigma = fwhm * gaussian_fwhm_to_sigma
    yy, xx = np.mgrid[0:spacing, 0:spacing]
    cen = (spacing - 1) / 2.0
    half = offset / 2.0
    for i in range(n_sources):
        cell, pair = divmod(i, 2)
        gy, gx = divmod(cell, n_grid)
        sign = 1.0 if pair == 0 else -1.0
        amplitude = 100.0 if pair == 0 else 60.0
        xc = cen + sign * half + rng.uniform(-0.5, 0.5)
        yc = cen + sign * half + rng.uniform(-0.5, 0.5)
        model = Gaussian2D(amplitude, xc, yc, sigma, sigma)
        x0 = gx * spacing
        y0 = gy * spacing
        data[y0:y0 + spacing, x0:x0 + spacing] += model(xx, yy)

    kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data = convolve(data, kernel)
    segm = detect_sources(convolved_data, 5.0, 10)
    error = np.full(data.shape, 1.0)
    error[::7, ::11] = 1.5
    background = rng.normal(0.1, 0.02, data.shape)
    mask = np.zeros(data.shape, dtype=bool)
    mask[::23, ::17] = True
    return {'data': data, 'convolved_data': convolved_data, 'segm': segm,
            'error': error, 'background': background, 'mask': mask,
            'wcs': make_wcs(data.shape)}


def make_edge_scene(*, seed=0):
    """
    Build a small scene exercising the edge cases of the catalog
    computations.

    The scene contains isolated sources, close/overlapping pairs,
    sources touching every image edge, masked pixels inside a source
    and spanning a close pair, and non-finite data values inside a
    source.

    Parameters
    ----------
    seed : int, optional
        The random number generator seed.

    Returns
    -------
    scene : dict
        The ``data``, ``convolved_data``, ``segm``, ``error``,
        ``background``, ``mask``, and ``wcs`` inputs.
    """
    rng = np.random.default_rng(seed)
    ny = nx = 151
    yy, xx = np.mgrid[0:ny, 0:nx]
    data = rng.normal(0.0, 0.1, (ny, nx))
    positions = [(20, 20), (24, 26), (75, 75), (75, 81), (140, 20),
                 (5, 100), (100, 4), (147, 147), (50, 120), (120, 50),
                 (100, 100)]
    for i, (xc, yc) in enumerate(positions):
        amp = 5.0 + i
        sig = 1.5 + 0.2 * (i % 4)
        data += amp * np.exp(-((xx - xc) ** 2 + (yy - yc) ** 2)
                             / (2 * sig ** 2))
    error = np.full((ny, nx), 0.1)
    error[::17, ::13] = 0.3
    mask = np.zeros((ny, nx), dtype=bool)
    mask[18:20, 22:24] = True  # inside a source of a close pair
    mask[73:75, 72:84] = True  # spans a close pair
    data[77, 78] = np.nan  # non-finite inside a source
    data[75, 83] = np.inf
    segm = detect_sources(data, 1.0, 5)
    kernel = make_2dgaussian_kernel(2.0, size=5)
    convolved_data = convolve(data, kernel)
    return {'data': data, 'convolved_data': convolved_data, 'segm': segm,
            'error': error, 'background': np.full(data.shape, 0.05),
            'mask': mask, 'wcs': make_wcs(data.shape)}


def build_configurations(scene):
    """
    Build the catalog keyword configurations for a scene.

    Parameters
    ----------
    scene : dict
        The scene inputs.

    Returns
    -------
    result : dict
        The ``SourceCatalog`` keyword arguments keyed by configuration
        name.
    """
    full = {'convolved_data': scene['convolved_data'],
            'error': scene['error'], 'background': scene['background'],
            'mask': scene['mask'], 'wcs': scene['wcs']}
    return {
        'full_correct': dict(full),
        'full_mask': {**full, 'aperture_mask_method': 'mask'},
        'full_none': {**full, 'aperture_mask_method': 'none'},
        'bare': {},
        'localbkg': {'error': scene['error'], 'local_bkg_width': 6},
        'kron_circ': {'error': scene['error'],
                      'kron_params': (2.5, 1.4, 6.0)},
    }


def to_plain(value):
    """
    Convert a property value to plain NumPy data.

    Parameters
    ----------
    value : object
        The property value.

    Returns
    -------
    result : `~numpy.ndarray`, list of `~numpy.ndarray`, or `None`
        The plain data, or `None` if the value has no array form (e.g.,
        a list of aperture objects).
    """
    if isinstance(value, SkyCoord):
        return np.column_stack([value.ra.deg, value.dec.deg])
    if isinstance(value, u.Quantity):
        value = np.asarray(value.value)
    if isinstance(value, np.ndarray):
        return None if value.dtype == object else value
    if (isinstance(value, (list, tuple)) and len(value)
            and isinstance(value[0], np.ndarray)):
        # A list of per-source arrays (cutouts)
        return [np.ma.getdata(item) if isinstance(item, np.ma.MaskedArray)
                else np.asarray(item) for item in value]
    return None


def dump_catalog(catalog, prefix, out):
    """
    Dump the results of a catalog into a dictionary.

    Parameters
    ----------
    catalog : `~photutils.segmentation.SourceCatalog`
        The catalog.

    prefix : str
        The key prefix (the scene and configuration names).

    out : dict
        The dictionary to update, keyed by ``'<prefix>:<property>'``.
    """
    for name in catalog.properties:
        try:
            value = getattr(catalog, name)
        except Exception as exc:  # noqa: BLE001
            out[f'{prefix}:{name}'] = f'ERROR: {exc!r}'
            continue
        plain = to_plain(value)
        if plain is not None:
            out[f'{prefix}:{name}'] = plain

    for fraction in (0.5, 0.3):
        out[f'{prefix}:flux_radius({fraction})'] = np.asarray(
            catalog.flux_radius(fraction).value)
    flux, flux_err = catalog.circular_photometry(5.0)
    out[f'{prefix}:circular_photometry(5).flux'] = np.asarray(flux)
    out[f'{prefix}:circular_photometry(5).flux_err'] = np.asarray(flux_err)
    for kron_params in ((2.5, 1.4), (1.8, 1.0, 2.0)):
        flux, flux_err = catalog.kron_photometry(kron_params)
        name = f'kron_photometry({kron_params})'
        out[f'{prefix}:{name}.flux'] = np.asarray(flux)
        out[f'{prefix}:{name}.flux_err'] = np.asarray(flux_err)

    table = catalog.to_table()
    for column in table.colnames:
        plain = to_plain(table[column])
        if plain is None:
            plain = np.asarray(table[column])
        out[f'{prefix}:to_table.{column}'] = plain


def dump(filename):
    """
    Compute the catalog results of the importable photutils and write
    them to a file.

    Parameters
    ----------
    filename : str
        The output file.
    """
    warnings.simplefilter('ignore')
    out = {'__version__': photutils.__version__}
    scenes = (('bench', make_bench_scene()), ('edge', make_edge_scene()))
    for scene_name, scene in scenes:
        configurations = build_configurations(scene)
        for config_name, kwargs in configurations.items():
            catalog = SourceCatalog(scene['data'], scene['segm'], **kwargs)
            dump_catalog(catalog, f'{scene_name}/{config_name}', out)

        catalog = SourceCatalog(scene['data'], scene['segm'],
                                **configurations['full_correct'])
        dump_catalog(catalog[[0, 2, 3]], f'{scene_name}/slice', out)
        dump_catalog(catalog[1], f'{scene_name}/scalar', out)

        detection_catalog = SourceCatalog(scene['convolved_data'],
                                          scene['segm'],
                                          error=scene['error'])
        catalog = SourceCatalog(scene['data'], scene['segm'],
                                error=scene['error'],
                                detection_catalog=detection_catalog)
        dump_catalog(catalog, f'{scene_name}/detcat', out)

    with open(filename, 'wb') as fh:
        pickle.dump(out, fh)
    print(f'photutils {photutils.__version__} ({photutils.__file__}): '
          f'{len(out) - 1} entries written to {filename}')


def _compare_arrays(value, reference, *, rtol, atol):
    """
    Compare two dumped arrays (see ``compare_values``).

    Parameters
    ----------
    value, reference : array_like
        The dumped arrays.

    rtol, atol : float
        The tolerances of the comparison.

    Returns
    -------
    result : tuple
        The ``(equal, max_abs, max_rel, note)`` tuple of
        ``compare_values``.
    """
    value = np.asarray(value)
    reference = np.asarray(reference)
    if value.shape != reference.shape:
        return (False, np.nan, np.nan,
                f'shape {value.shape} versus {reference.shape}')
    if value.dtype.kind in 'OUSb' or reference.dtype.kind in 'OUSb':
        equal = np.array_equal(value, reference)
        return equal, np.nan, np.nan, '' if equal else 'value mismatch'

    value = value.astype(float)
    reference = reference.astype(float)
    if not (np.array_equal(np.isnan(value), np.isnan(reference))
            and np.array_equal(np.isinf(value), np.isinf(reference))):
        return False, np.nan, np.nan, 'NaN/inf pattern mismatch'
    infinite = np.isinf(reference)
    if not np.array_equal(value[infinite], reference[infinite]):
        return False, np.nan, np.nan, 'inf sign mismatch'
    finite = np.isfinite(reference)
    diff = np.abs(value[finite] - reference[finite])
    if diff.size == 0:
        return True, 0.0, 0.0, ''
    scale = np.maximum(np.abs(reference[finite]), 1e-300)
    equal = bool(np.allclose(value[finite], reference[finite], rtol=rtol,
                             atol=atol))
    return equal, float(diff.max()), float((diff / scale).max()), ''


def compare_values(value, reference, *, rtol, atol):
    """
    Compare two dumped values.

    Parameters
    ----------
    value, reference : object
        The dumped values.

    rtol, atol : float
        The tolerances of the comparison.

    Returns
    -------
    equal : bool
        Whether the values agree within the tolerances.

    max_abs, max_rel : float
        The maximum absolute and relative differences of the finite
        values (NaN if not applicable).

    note : str
        A description of a structural mismatch, or an empty string.
    """
    if isinstance(value, str) or isinstance(reference, str):
        equal = value == reference
        return equal, np.nan, np.nan, '' if equal else 'error mismatch'
    if not (isinstance(value, list) or isinstance(reference, list)):
        return _compare_arrays(value, reference, rtol=rtol, atol=atol)

    # Lists of per-source arrays (cutouts)
    if not (isinstance(value, list) and isinstance(reference, list)):
        return False, np.nan, np.nan, 'list versus array'
    if len(value) != len(reference):
        return (False, np.nan, np.nan,
                f'length {len(value)} versus {len(reference)}')
    worst = (True, 0.0, 0.0, '')
    for item, ref_item in zip(value, reference, strict=True):
        result = _compare_arrays(item, ref_item, rtol=rtol, atol=atol)
        if not result[0]:
            worst = (False, np.nanmax([worst[1], result[1]]),
                     np.nanmax([worst[2], result[2]]), result[3])
    return worst


def compare(reference_file, new_file, *, rtol, atol):
    """
    Compare two dump files and print the differences.

    Parameters
    ----------
    reference_file, new_file : str
        The reference and new dump files.

    rtol, atol : float
        The tolerances of the comparison.

    Returns
    -------
    n_differ : int
        The number of differing entries.
    """
    with open(reference_file, 'rb') as fh:
        reference = pickle.load(fh)  # noqa: S301
    with open(new_file, 'rb') as fh:
        new = pickle.load(fh)  # noqa: S301
    ref_version = reference.pop('__version__', 'unknown')
    new_version = new.pop('__version__', 'unknown')
    print(f'reference: photutils {ref_version} ({len(reference)} entries)')
    print(f'new:       photutils {new_version} ({len(new)} entries)')

    ref_keys = set(reference)
    new_keys = set(new)
    only_ref = sorted({key.split(':', 1)[1] for key in ref_keys - new_keys})
    only_new = sorted({key.split(':', 1)[1] for key in new_keys - ref_keys})
    if only_ref:
        print(f'\nproperties only in the reference: {only_ref}')
    if only_new:
        print(f'\nproperties only in the new version: {only_new}')

    common = sorted(ref_keys & new_keys)
    same = defaultdict(int)
    differ = defaultdict(list)
    for key in common:
        name = key.split(':', 1)[1]
        result = compare_values(new[key], reference[key], rtol=rtol,
                                atol=atol)
        if result[0]:
            same[name] += 1
        else:
            differ[name].append((key, *result[1:]))

    print(f'\n== identical within rtol={rtol:g}, atol={atol:g}: '
          f'{sum(same.values())} entries, {len(same)} properties ==')
    n_differ = sum(len(items) for items in differ.values())
    print(f'\n== differing: {n_differ} entries, {len(differ)} '
          'properties ==')
    for name in sorted(differ):
        items = differ[name]
        rel = [item[2] for item in items if np.isfinite(item[2])]
        max_rel = f'{max(rel):.2e}' if rel else 'n/a'
        notes = sorted({item[3] for item in items} - {''})
        print(f'  {name:44s} n={len(items):3d}  max rel diff '
              f'{max_rel:>8s}  {notes}')
        for key, max_abs, max_rel, note in items:
            print(f'      {key:56s} max abs {max_abs:.2e} '
                  f'max rel {max_rel:.2e} {note}')
    return n_differ


def main():
    """
    Run the SourceCatalog cross-version validation.
    """
    parser = argparse.ArgumentParser(
        description='Validate SourceCatalog results across photutils '
                    'versions.')
    subparsers = parser.add_subparsers(dest='command', required=True)
    dump_parser = subparsers.add_parser(
        'dump', help='compute the results of the importable photutils '
                     'and write them to a file')
    dump_parser.add_argument('filename', help='the output file')
    compare_parser = subparsers.add_parser(
        'compare', help='compare two dump files')
    compare_parser.add_argument('reference', help='the reference dump '
                                                  'file')
    compare_parser.add_argument('new', help='the new dump file')
    compare_parser.add_argument('--rtol', type=float, default=1e-10,
                                help='relative tolerance '
                                     '(default: %(default)s)')
    compare_parser.add_argument('--atol', type=float, default=1e-6,
                                help='absolute tolerance '
                                     '(default: %(default)s)')
    args = parser.parse_args()

    if args.command == 'dump':
        dump(args.filename)
    else:
        n_differ = compare(args.reference, args.new, rtol=args.rtol,
                           atol=args.atol)
        if n_differ:
            sys.exit(1)


if __name__ == '__main__':
    main()
