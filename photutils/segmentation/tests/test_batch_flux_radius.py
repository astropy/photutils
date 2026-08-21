# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the batch flux_radius solve driver.
"""

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import root_scalar

from photutils.geometry import circular_overlap_grid
from photutils.segmentation._batch_catalog import batch_flux_radius_solve
from photutils.segmentation.tests._batch_scene import (make_batch_scene,
                                                       make_catalog)


def _reference_solve(args_list, fraction):
    # Verbatim port of the pre-change ``flux_radius`` solve loop and
    # ``_flux_radius_fcn``, operating on the optimizer-args entries
    def fcn(radius, clean_data, grid_params, normflux):
        xmin_e, xmax_e, ymin_e, ymax_e, nx, ny, exact, subpx = \
            grid_params
        weights = circular_overlap_grid(xmin_e, xmax_e, ymin_e,
                                        ymax_e, nx, ny, radius,
                                        exact, subpx)
        return 1.0 - (np.sum(clean_data * weights) / normflux)

    radius = []
    for entry in args_list:
        if entry is None:
            radius.append(np.nan)
            continue
        clean_data, grid_params, kronflux, max_radius = entry
        normflux = kronflux * fraction
        found = False
        min_radius = 0.1
        max_radius_delta = 0.1 * max_radius
        while max_radius > min_radius and found is False:
            try:
                result = root_scalar(
                    fcn, args=(clean_data, grid_params, normflux),
                    bracket=[min_radius, max_radius],
                    method='brentq')
                result = result.root
                found = True
            except ValueError:
                max_radius -= max_radius_delta
        if found is False:
            result = np.nan
        radius.append(result)
    return np.array(radius)


@pytest.fixture(scope='module')
def scene():
    return make_batch_scene()


@pytest.mark.parametrize('method', ['correct', 'mask', 'none'])
@pytest.mark.parametrize('fraction', [0.2, 0.5, 0.9])
def test_matches_reference(scene, method, fraction):
    cat = make_catalog(scene, aperture_mask_method=method)
    args = cat._flux_radius_optimizer_args
    expected = _reference_solve(args, fraction)
    result = batch_flux_radius_solve(args, fraction=fraction)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


def test_none_entries(scene):
    cat = make_catalog(scene)
    args = list(cat._flux_radius_optimizer_args)
    args[0] = None
    result = batch_flux_radius_solve(args, fraction=0.5)
    assert np.isnan(result[0])
    assert_allclose(result, _reference_solve(args, 0.5), rtol=1e-12,
                    equal_nan=True)


def test_bracket_shrink_and_no_solution(scene):
    # Force the shrink path: negative data beyond a ring makes the
    # enclosed flux non-monotonic, so the initial bracket has equal
    # signs at both ends
    cat = make_catalog(scene)
    entry = [e for e in cat._flux_radius_optimizer_args
             if e is not None][0]
    clean_data = entry[0].copy()
    yc = entry[0].shape[0] / 2
    xc = entry[0].shape[1] / 2
    yy, xx = np.mgrid[0:entry[0].shape[0], 0:entry[0].shape[1]]
    rr = np.hypot(xx - xc, yy - yc)
    clean_data[rr > 3] = -np.abs(clean_data[rr > 3]) - 1.0
    forced = [[np.ascontiguousarray(clean_data), entry[1], entry[2],
               entry[3]]]
    assert_allclose(batch_flux_radius_solve(forced, fraction=0.5),
                    _reference_solve(forced, 0.5), rtol=1e-12,
                    equal_nan=True)

    # A milder outer ring, where shrinking the bracket does reveal a
    # sign change and a root is found on a later retry
    shrink_data = entry[0].copy()
    shrink_data[rr > 3] = -1.0
    shrink = [[np.ascontiguousarray(shrink_data), entry[1], entry[2],
               entry[3]]]
    expected = _reference_solve(shrink, 0.5)
    assert np.isfinite(expected[0])
    assert_allclose(batch_flux_radius_solve(shrink, fraction=0.5),
                    expected, rtol=1e-12)

    # And a hopeless case that shrinks to no solution -> NaN
    hopeless = [[np.ascontiguousarray(np.full_like(clean_data,
                                                   -1.0)),
                 entry[1], entry[2], entry[3]]]
    assert np.isnan(batch_flux_radius_solve(hopeless,
                                            fraction=0.5)[0])


def test_catalog_flux_radius(scene):
    cat = make_catalog(scene)
    expected = _reference_solve(cat._flux_radius_optimizer_args, 0.5)
    result = cat.flux_radius(0.5)
    assert result.unit == u.pix
    assert_allclose(result.value, expected, rtol=1e-12,
                    equal_nan=True)

    # Named property
    cat.flux_radius(0.3, name='r30')
    assert_allclose(cat.r30.value,
                    _reference_solve(
                        cat._flux_radius_optimizer_args, 0.3),
                    rtol=1e-12, equal_nan=True)
