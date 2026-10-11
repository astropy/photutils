# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the gaussian module.
"""

from contextlib import nullcontext
from unittest.mock import patch

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import Gaussian1D
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_array_equal

from photutils.centroids._utils import _gaussian1d_moments, _gaussian2d_moments
from photutils.centroids.gaussian import centroid_1dg, centroid_2dg
from photutils.centroids.tests.helpers import make_gaussian_source


@pytest.mark.parametrize('x_std', [3.2, 4.0])
@pytest.mark.parametrize('y_std', [5.7, 4.1])
@pytest.mark.parametrize('theta', np.deg2rad([30.0, 45.0]))
@pytest.mark.parametrize('units', [True, False])
def test_centroids(x_std, y_std, theta, units):
    """
    Test the 1D and 2D Gaussian centroid functions on a simple 2D
    Gaussian model.
    """
    xc_ref = 25.7
    yc_ref = 26.2
    data = make_gaussian_source((50, 47), 2.4, xc_ref, yc_ref, x_std, y_std,
                                theta)
    error = np.sqrt(np.abs(data))

    value = 1.0e5
    if units:
        unit = u.nJy
        data = data * unit
        error = error * unit
        value *= unit

    xc, yc = centroid_1dg(data)
    assert_allclose((xc, yc), (xc_ref, yc_ref), rtol=0, atol=1.0e-3)
    xc, yc = centroid_2dg(data)
    assert_allclose((xc, yc), (xc_ref, yc_ref), rtol=0, atol=1.0e-3)

    # Test with errors
    xc, yc = centroid_1dg(data, error=error)
    assert_allclose((xc, yc), (xc_ref, yc_ref), rtol=0, atol=1.0e-3)
    xc, yc = centroid_2dg(data, error=error)
    assert_allclose((xc, yc), (xc_ref, yc_ref), rtol=0, atol=1.0e-3)

    # Test with mask
    mask = np.zeros(data.shape, dtype=bool)
    data[10, 10] = value
    mask[10, 10] = True
    xc, yc = centroid_1dg(data, mask=mask)
    assert_allclose((xc, yc), (xc_ref, yc_ref), rtol=0, atol=1.0e-3)
    xc, yc = centroid_2dg(data, mask=mask)
    assert_allclose((xc, yc), (xc_ref, yc_ref), rtol=0, atol=1.0e-3)


@pytest.mark.parametrize('use_mask', [True, False])
def test_centroids_nan_withmask(use_mask):
    """
    Test that the 1D and 2D Gaussian centroid functions can handle NaN
    values in the input data, both with and without a mask.
    """
    xc_ref = 24.7
    yc_ref = 25.2
    data = make_gaussian_source((50, 50), 2.4, xc_ref, yc_ref, 5.0, 5.0, 0.0)
    data[20, :] = np.nan

    if use_mask:
        mask = np.zeros(data.shape, dtype=bool)
        mask[20, :] = True
        n_warn = 0
        ctx = nullcontext()
    else:
        mask = None
        n_warn = 1
        match = 'Input data contains non-finite values'
        ctx = pytest.warns(AstropyUserWarning, match=match)

    with ctx as warnlist:
        xc, yc = centroid_1dg(data, mask=mask)
        assert_allclose([xc, yc], [xc_ref, yc_ref], rtol=0, atol=1.0e-3)
        if n_warn == 1:
            assert len(warnlist) == n_warn

    with ctx as warnlist:
        xc, yc = centroid_2dg(data, mask=mask)
        assert_allclose([xc, yc], [xc_ref, yc_ref], rtol=0, atol=1.0e-3)
        if n_warn == 1:
            assert len(warnlist) == n_warn


def test_invalid_shapes():
    """
    Test that the 1D and 2D Gaussian centroid functions raise an error
    for invalid data, mask, or error shapes.
    """
    data = np.zeros((4, 4, 4))
    match = 'data must be a 2D array'
    with pytest.raises(ValueError, match=match):
        centroid_1dg(data)
    with pytest.raises(ValueError, match=match):
        centroid_2dg(data)

    data = np.zeros((4, 4))
    mask = np.zeros((2, 2), dtype=bool)
    match = 'data and mask must have the same shape'
    with pytest.raises(ValueError, match=match):
        centroid_1dg(data, mask=mask)
    with pytest.raises(ValueError, match=match):
        centroid_2dg(data, mask=mask)
    data = np.zeros(4)
    mask = np.zeros(2, dtype=bool)
    with pytest.raises(ValueError, match=match):
        _gaussian1d_moments(data, mask=mask)


def test_invalid_error_shape():
    """
    Test that the 1D and 2D Gaussian centroid functions raise an error
    for invalid error shapes.
    """
    error = np.zeros((2, 2), dtype=bool)
    match = 'data and error must have the same shape'
    with pytest.raises(ValueError, match=match):
        centroid_1dg(np.zeros((4, 4)), error=error)
    with pytest.raises(ValueError, match=match):
        centroid_2dg(np.zeros((4, 4)), error=error)


def test_centroid_2dg_dof():
    """
    Test that the 2D Gaussian centroid function raises an error if there
    are not enough unmasked values to fit the model.
    """
    data = np.ones((2, 2))
    match = 'Input data must have at least 6 unmasked values to fit'
    with pytest.raises(ValueError, match=match):
        centroid_2dg(data)


@pytest.mark.parametrize('value', [0.0, 1.0, -3.7])
def test_centroid_2dg_constant_data(value):
    """
    Test that centroid_2dg raises a ValueError for constant (flat) input
    data.

    After subtracting the minimum, a constant array becomes all-zero,
    making the moment sum zero and the Gaussian parameters undefined.
    A clear error is raised rather than silent NaN results.
    """
    data = np.full((10, 10), value)
    match = 'Input data must have non-constant values'
    with pytest.raises(ValueError, match=match):
        centroid_2dg(data)


@pytest.mark.parametrize('value', [0.0, 1.0, -3.7])
def test_centroid_2dg_constant_data_masked(value):
    """
    Test that centroid_2dg raises a ValueError if the unmasked data are
    constant.

    The masked pixels are set to zero internally, so they must not make
    constant data appear to vary.
    """
    data = np.full((10, 10), value)
    data[2, 3] = 100.0
    mask = np.zeros(data.shape, dtype=bool)
    mask[2, 3] = True
    match = 'Input data must have non-constant values'
    with pytest.raises(ValueError, match=match):
        centroid_2dg(data, mask=mask)

    data[2, 3] = np.nan
    with (pytest.warns(AstropyUserWarning, match='non-finite values'),
          pytest.raises(ValueError, match=match)):
        centroid_2dg(data)


def test_centroid_2dg_initial_guess_masked():
    """
    Test that masked pixels do not contribute to the initial parameter
    estimates of centroid_2dg when the data minimum is negative.

    The minimum of the unmasked data is subtracted before the moments
    are calculated, and the masked pixels must remain zero.
    """
    data = make_gaussian_source((31, 31), 100.0, 12.3, 17.6, 2.0, 2.0, 0)
    data -= 5.0
    mask = np.zeros(data.shape, dtype=bool)
    mask[26:28, :10] = True
    data[mask] = 1000.0

    target = 'photutils.centroids.gaussian._gaussian2d_moments'
    with patch(target, wraps=_gaussian2d_moments) as mock_moments:
        xycen = centroid_2dg(data, mask=mask)
    shifted = mock_moments.call_args.args[0]
    assert_array_equal(shifted[mask], 0.0)
    assert np.min(shifted[~mask]) == 0.0
    assert_allclose(_gaussian2d_moments(shifted)[1:3], (12.3, 17.6),
                    atol=0.01)
    assert_allclose(xycen, (12.3, 17.6), atol=1.0e-4)


@pytest.mark.parametrize('value', [0.0, 1.0, -3.7])
def test_centroid_1dg_constant_data(value):
    """
    Test that centroid_1dg raises a ValueError for constant (flat)
    input data.

    A constant marginal distribution has undefined Gaussian moment
    estimates. This previously raised a cryptic error from inside the
    fitter.
    """
    data = np.full((10, 10), value)
    match = 'Input data must have a nonzero sum and non-constant values'
    with pytest.raises(ValueError, match=match):
        centroid_1dg(data)


def test_centroid_1dg_zero_sum_data():
    """
    Test that centroid_1dg raises a ValueError for non-constant data
    whose sum is zero.

    A zero-sum marginal distribution makes the moment estimates 0/0.
    """
    data = np.zeros((10, 10))
    data[3, 3] = 5.0
    data[7, 7] = -5.0
    match = 'Input data must have a nonzero sum and non-constant values'
    with pytest.raises(ValueError, match=match):
        centroid_1dg(data)


def test_gaussian1d_moments():
    """
    Test the _gaussian1d_moments function on a simple 1D Gaussian model.
    """
    x = np.arange(100)
    desired = (75, 50, 5)
    g = Gaussian1D(*desired)
    data = g(x)
    result = _gaussian1d_moments(data)
    assert_allclose(result, desired, rtol=0, atol=1.0e-6)

    data[0] = 1.0e5
    mask = np.zeros(data.shape, dtype=bool)
    mask[0] = True
    result = _gaussian1d_moments(data, mask=mask)
    assert_allclose(result, desired, rtol=0, atol=1.0e-6)

    # Test that masked NaNs do not raise a warning
    data[0] = np.nan
    mask = np.zeros(data.shape, dtype=bool)
    mask[0] = True
    result = _gaussian1d_moments(data, mask=mask)
    assert_allclose(result, desired, rtol=0, atol=1.0e-6)

    # Test that unmasked NaNs raise a warning
    data[0] = np.nan
    mask = np.zeros(data.shape, dtype=bool)
    mask[0] = False
    match = 'Input data contains non-finite values'
    with pytest.warns(AstropyUserWarning, match=match):
        result = _gaussian1d_moments(data, mask=mask)
    assert_allclose(result, desired, rtol=0, atol=1.0e-6)


def test_gaussian2d_warning():
    """
    Test that the 2D Gaussian centroid function emits only the warning
    of the fitter if the fit did not converge.
    """
    data = make_gaussian_source((51, 51), 1.0, 24.17, 25.87, 1.7, 4.7, 0.0)

    match = 'The fit may be unsuccessful'
    with pytest.warns(AstropyUserWarning, match=match) as record:
        centroid_2dg(data + 100000)
    assert len(record) == 1


def test_no_input_mutation():
    """
    Test that input mask and error arrays are not mutated by
    centroid_1dg or centroid_2dg.
    """
    data = make_gaussian_source((50, 50), 2.4, 25.0, 25.0, 5.0, 5.0, 0.0)

    # Add a masked position and a NaN in error to exercise all
    # copy-on-write paths without triggering data-NaN warnings
    mask = np.zeros(data.shape, dtype=bool)
    mask[10, 10] = True
    error = np.sqrt(np.abs(data))
    error[15, 15] = np.nan

    mask_orig = mask.copy()
    error_orig = error.copy()

    centroid_1dg(data, error=error, mask=mask)
    assert_array_equal(mask, mask_orig)
    assert_array_equal(error, error_orig)

    centroid_2dg(data, error=error, mask=mask)
    assert_array_equal(mask, mask_orig)
    assert_array_equal(error, error_orig)


def test_masked_array_input():
    """
    Test that MaskedArray inputs to centroid_1dg and centroid_2dg give
    the same results as equivalent plain array and mask inputs.
    """
    data = make_gaussian_source((50, 50), 2.4, 25.0, 25.0, 5.0, 5.0, 0.0)

    mask = np.zeros(data.shape, dtype=bool)
    mask[10, 10] = True

    # Plain array with mask keyword
    xc1, yc1 = centroid_1dg(data, mask=mask)
    xc2, yc2 = centroid_2dg(data, mask=mask)

    # MaskedArray (no mask keyword)
    masked_data = np.ma.array(data, mask=mask)
    xc1_ma, yc1_ma = centroid_1dg(masked_data)
    xc2_ma, yc2_ma = centroid_2dg(masked_data)

    assert_allclose([xc1_ma, yc1_ma], [xc1, yc1])
    assert_allclose([xc2_ma, yc2_ma], [xc2, yc2])
