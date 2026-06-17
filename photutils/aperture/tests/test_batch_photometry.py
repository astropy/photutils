# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the batch Cython aperture photometry driver, which must give
results identical to the mask-based code path.
"""

from concurrent.futures import ThreadPoolExecutor

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.aperture.circle import CircularAnnulus, CircularAperture
from photutils.aperture.core import PixelAperture
from photutils.aperture.ellipse import EllipticalAnnulus, EllipticalAperture
from photutils.aperture.polygon import PolygonAperture
from photutils.aperture.rectangle import (RectangularAnnulus,
                                          RectangularAperture)

RNG = np.random.default_rng(0)
SHAPE = (157, 163)
DATA = RNG.normal(100.0, 5.0, SHAPE)
ERROR = RNG.uniform(1.0, 2.0, SHAPE)
MASK = RNG.uniform(size=SHAPE) < 0.05
DATA_NAN = DATA.copy()
DATA_NAN[::11, ::13] = np.nan

# Positions covering interior, subpixel offsets, partial overlap with
# every data edge, no overlap, and far outside the data
POSITIONS = np.array([[70.0, 80.0], [50.3, 71.8], [20.49999, 30.50001],
                      [0.0, 0.0], [162.0, 156.0], [-3.0, 80.0],
                      [166.0, 80.0], [80.0, -3.0], [80.0, 160.0],
                      [-50.0, -50.0], [250.0, 250.0], [1e9, -1e9]])

APERTURES = [
    CircularAperture(POSITIONS, r=5.5),
    CircularAperture(POSITIONS, r=0.4),
    CircularAnnulus(POSITIONS, r_in=4.0, r_out=6.0),
    EllipticalAperture(POSITIONS, a=5.0, b=3.0, theta=0.7),
    EllipticalAnnulus(POSITIONS, a_in=3.0, a_out=6.0, b_out=4.0, theta=-1.1),
    RectangularAperture(POSITIONS, w=7.0, h=4.0, theta=0.4),
    RectangularAnnulus(POSITIONS, w_in=3.0, w_out=8.0, h_out=5.0, theta=2.2),
    PolygonAperture.from_regular_polygon(POSITIONS, 6, radius=5.5),
    PolygonAperture(POSITIONS, [[-4.0, -3.0], [6.0, -2.0], [3.0, 5.0],
                                [-5.0, 4.0]]),
    # A non-convex (concave arrowhead) polygon offset shape
    PolygonAperture(POSITIONS, [[-5.0, -4.0], [5.0, -4.0], [0.0, 6.0],
                                [0.0, 0.0]]),
]

METHODS = [('exact', 5), ('center', 5), ('subpixel', 1), ('subpixel', 7)]


def assert_batch_matches_legacy(aperture, data, *, error=None, mask=None,
                                method='exact', subpixels=5):
    """
    Test that the batch photometry driver gives results identical to the
    legacy mask-based code path for the given inputs.
    """
    batch = aperture._do_batch_photometry(data, error=error, mask=mask,
                                          method=method, subpixels=subpixels)
    assert batch is not None
    legacy = aperture._do_mask_photometry(data, error=error, mask=mask,
                                          method=method, subpixels=subpixels)
    for batch_arr, legacy_arr in zip(batch, legacy, strict=True):
        assert batch_arr.shape == legacy_arr.shape
        assert_allclose(batch_arr, legacy_arr, rtol=1e-12, atol=0,
                        equal_nan=True)


@pytest.mark.parametrize('aperture', APERTURES)
@pytest.mark.parametrize(('method', 'subpixels'), METHODS)
def test_matches_legacy(aperture, method, subpixels):
    """
    Test that the batch photometry driver gives results identical to the
    legacy mask-based code path for the given inputs.
    """
    assert_batch_matches_legacy(aperture, DATA, method=method,
                                subpixels=subpixels)


@pytest.mark.parametrize('aperture', APERTURES)
@pytest.mark.parametrize(('method', 'subpixels'), METHODS)
def test_matches_legacy_error_mask(aperture, method, subpixels):
    """
    Test that the batch photometry driver gives results identical to the
    legacy mask-based code path for the given inputs, including error
    and mask.
    """
    assert_batch_matches_legacy(aperture, DATA, error=ERROR, mask=MASK,
                                method=method, subpixels=subpixels)


@pytest.mark.parametrize('aperture', APERTURES)
def test_matches_legacy_nan_data(aperture):
    """
    Test that the batch photometry driver gives results identical to the
    legacy mask-based code path for data containing NaN values.
    """
    assert_batch_matches_legacy(aperture, DATA_NAN, error=ERROR)


@pytest.mark.parametrize('dtype', ['int32', 'uint16', 'float32', 'bool'])
def test_data_dtypes(dtype):
    """
    Test that the batch photometry driver gives results identical to the
    legacy mask-based code path for different data dtypes, including
    bool.
    """
    rtol = 1e-5 if dtype == 'float32' else 1e-12
    aperture = CircularAperture(POSITIONS, r=5.5)
    data = (DATA > 100.0) if dtype == 'bool' else DATA.astype(dtype)
    batch = aperture._do_batch_photometry(data, error=None, mask=None,
                                          method='exact', subpixels=5)
    legacy = aperture._do_mask_photometry(data, error=None, mask=None,
                                          method='exact', subpixels=5)
    assert batch is not None
    assert_allclose(batch[0], legacy[0], rtol=rtol, equal_nan=True)


def test_scalar_position():
    """
    Test that the batch photometry driver gives results identical to the
    legacy mask-based code path for a single scalar position.
    """
    aperture = CircularAperture((70.0, 80.0), r=5.5)
    assert_batch_matches_legacy(aperture, DATA, error=ERROR)

    aperture = EllipticalAperture((70.0, 80.0), 10.124, 5.073)
    assert_batch_matches_legacy(aperture, DATA, error=ERROR)


def test_units():
    """
    Test that the batch photometry driver gives results with the same
    units as the input data and error, and that the values match the
    legacy code path.
    """
    unit = u.Jy
    aperture = CircularAperture(POSITIONS, r=5.5)
    sums, errs = aperture.do_photometry(DATA << unit, error=ERROR << unit)
    assert sums.unit == unit
    assert errs.unit == unit
    sums2, errs2 = aperture.do_photometry(DATA, error=ERROR)
    assert_allclose(sums.value, sums2, equal_nan=True)
    assert_allclose(errs.value, errs2, equal_nan=True)


def test_no_overlap_nan_and_err_shape():
    """
    Test that sources with no overlap with the data give NaN sums.

    Also, test that the error array has the same shape as the sums array
    and contains NaN values for the non-overlapping sources when error
    is not input.
    """
    aperture = CircularAperture(POSITIONS, r=5.5)
    n_outside = 3

    sums, errs = aperture._do_batch_photometry(DATA, error=None, mask=None,
                                               method='exact', subpixels=5)
    assert np.isnan(sums).sum() == n_outside
    assert errs.shape == (n_outside,)
    assert np.all(np.isnan(errs))

    sums, errs = aperture._do_batch_photometry(DATA, error=ERROR, mask=None,
                                               method='exact', subpixels=5)
    assert errs.shape == sums.shape
    assert np.isnan(errs).sum() == n_outside


def test_fallback_inputs():
    """
    Test that inputs not supported by the batch photometry driver cause
    it to return None so that the caller falls back to the mask-based
    code path.
    """
    aperture = CircularAperture(POSITIONS, r=5.5)

    # Masked-array data
    data = np.ma.MaskedArray(DATA, mask=MASK)
    assert aperture._do_batch_photometry(data, error=None, mask=None,
                                         method='exact', subpixels=5) is None

    # Non-bool mask
    assert aperture._do_batch_photometry(DATA, error=None,
                                         mask=MASK.astype(int),
                                         method='exact', subpixels=5) is None

    # Mask with mismatched shape
    assert aperture._do_batch_photometry(DATA, error=None, mask=MASK[1:, :],
                                         method='exact', subpixels=5) is None

    # Unsupported dtype
    assert aperture._do_batch_photometry(DATA.astype(complex), error=None,
                                         mask=None, method='exact',
                                         subpixels=5) is None


def test_fallback_subclass():
    """
    Test that aperture subclasses that do not themselves define
    _batch_shape_params fall back to the mask-based code path in
    do_photometry, so that any overridden behavior (e.g., to_mask) is
    honored.
    """

    class MyAperture(CircularAperture):
        # Inherits _batch_shape_params, but does not define it in its
        # own class, so the batch driver must not be used.
        pass

    aperture = MyAperture(POSITIONS, r=5.5)
    assert aperture._do_batch_photometry(DATA, error=None, mask=None,
                                         method='exact', subpixels=5) is None
    sums, _ = aperture.do_photometry(DATA)
    sums2, _ = CircularAperture(POSITIONS, r=5.5).do_photometry(DATA)
    assert_allclose(sums, sums2, rtol=1e-12, equal_nan=True)


def test_optin_subclass():
    """
    Test that aperture subclasses that define the _batch_shape_params
    hook in their own class opt in to the batch code path.
    """

    class MyAperture(CircularAperture):
        def _batch_shape_params(self):
            return super()._batch_shape_params()

    aperture = MyAperture(POSITIONS, r=5.5)
    assert aperture._do_batch_photometry(DATA, error=None, mask=None,
                                         method='exact', subpixels=5) \
        is not None
    assert_batch_matches_legacy(aperture, DATA, error=ERROR, mask=MASK)


def test_optin_subclass_base_hook_returns_none():
    """
    Test that when a subclass defines _batch_shape_params (opting in)
    but the method returns None, _do_batch_photometry falls back to
    None.

    This covers:
    - PixelAperture._batch_shape_params (the base ``return`` on its own
      line, which is the hook's default no-op implementation), and
    - the ``if spec is None: return None`` branch in _do_batch_photometry.
    """

    class MyAperture(CircularAperture):
        def _batch_shape_params(self):
            # Explicitly delegate to the base PixelAperture hook, which
            # returns None, signalling that the batch driver is not
            # supported for this subclass.
            return PixelAperture._batch_shape_params(self)

    aperture = MyAperture(POSITIONS, r=5.5)
    assert aperture._do_batch_photometry(DATA, error=None, mask=None,
                                         method='exact', subpixels=5) is None


def test_thread_safety():
    """
    Test that the batch driver releases the GIL and is therefore
    thread-safe by running multiple photometry calls in parallel threads
    and checking that they all give the same results.
    """
    aperture = CircularAperture(POSITIONS, r=5.5)
    expected, _ = aperture.do_photometry(DATA, error=ERROR)

    def run(_):
        return aperture.do_photometry(DATA, error=ERROR)[0]

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run, range(8)))

    for result in results:
        assert_allclose(result, expected, rtol=0, atol=0, equal_nan=True)
