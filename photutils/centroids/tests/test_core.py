# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the core module.
"""

import itertools
import warnings

from astropy.modeling.models import Gaussian2D
from astropy.utils.exceptions import AstropyUserWarning
import numpy as np
from numpy.testing import assert_allclose
import pytest

from ..core import centroid_com, centroid_epsf
from ...psf import IntegratedGaussianPRF

try:
    # the fitting routines in astropy use scipy.optimize
    import scipy  # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


XCEN = 25.7
YCEN = 26.2
XSTDS = [3.2, 4.0]
YSTDS = [5.7, 4.1]
THETAS = np.array([30., 45.]) * np.pi / 180.

DATA = np.zeros((3, 3))
DATA[0:2, 1] = 1.
DATA[1, 0:2] = 1.
DATA[1, 1] = 2.


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.parametrize(('x_std', 'y_std', 'theta'),
                         list(itertools.product(XSTDS, YSTDS, THETAS)))
def test_centroid_com(x_std, y_std, theta):
    model = Gaussian2D(2.4, XCEN, YCEN, x_stddev=x_std, y_stddev=y_std,
                       theta=theta)
    y, x = np.mgrid[0:50, 0:47]
    data = model(x, y)
    xc, yc = centroid_com(data)
    assert_allclose((xc, yc), (XCEN, YCEN), rtol=0, atol=1.e-3)

    # test with mask
    mask = np.zeros(data.shape, dtype=bool)
    data[10, 10] = 1.e5
    mask[10, 10] = True
    xc, yc = centroid_com(data, mask=mask)
    assert_allclose((xc, yc), (XCEN, YCEN), rtol=0, atol=1.e-3)

    # test with oversampling
    for oversampling in [4, (4, 6)]:
        if not hasattr(oversampling, '__len__'):
            _oversampling = (oversampling, oversampling)
        else:
            _oversampling = oversampling
        xc, yc = centroid_com(data, mask=mask, oversampling=oversampling)

        desired = [XCEN / _oversampling[0], YCEN / _oversampling[1]]
        assert_allclose((xc, yc), desired, rtol=0, atol=1.e-3)


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.parametrize('use_mask', [True, False])
def test_centroid_com_nan_withmask(use_mask):
    xc_ref = 24.7
    yc_ref = 25.2
    model = Gaussian2D(2.4, xc_ref, yc_ref, x_stddev=5.0, y_stddev=5.0)
    y, x = np.mgrid[0:50, 0:50]
    data = model(x, y)
    data[20, :] = np.nan
    if use_mask:
        mask = np.zeros(data.shape, dtype=bool)
        mask[20, :] = True
        nwarn = 0
    else:
        mask = None
        nwarn = 1

    with warnings.catch_warnings(record=True) as warnlist:
        xc, yc = centroid_com(data, mask=mask)
        assert_allclose(xc, xc_ref, rtol=0, atol=1.e-3)
        assert yc > yc_ref
    assert len(warnlist) == nwarn
    if nwarn == 1:
        assert issubclass(warnlist[0].category, AstropyUserWarning)


@pytest.mark.skipif('not HAS_SCIPY')
def test_centroid_com_invalid_inputs():
    data = np.zeros((4, 4))
    mask = np.zeros((2, 2), dtype=bool)
    with pytest.raises(ValueError):
        centroid_com(data, mask=mask)
    with pytest.raises(ValueError):
        centroid_com(data, oversampling=-1)


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.parametrize('oversampling', (4, (4, 6)))
def test_centroid_epsf(oversampling):
    sigma = 0.5
    psf = IntegratedGaussianPRF(sigma=sigma)
    offsets = np.array((0.1, 0.03))

    if not hasattr(oversampling, '__len__'):
        _oversampling = (oversampling, oversampling)
    else:
        _oversampling = oversampling

    x = np.arange(1 + 25 * _oversampling[0]) / _oversampling[0]
    y = np.arange(1 + 25 * _oversampling[1]) / _oversampling[1]
    x0 = x[-1] / 2
    y0 = y[-1] / 2
    x -= x0
    y -= y0

    data = psf.evaluate(x=x.reshape(1, -1), y=y.reshape(-1, 1), flux=1.,
                        x_0=offsets[0], y_0=offsets[1], sigma=sigma)

    mask = np.zeros(data.shape, dtype=bool)
    data[25, 25] = 1000.
    mask[25, 25] = True
    xc, yc = centroid_epsf(data, mask=mask, oversampling=oversampling)
    desired = np.array((x0, y0)) + offsets
    assert_allclose((xc, yc), desired, rtol=1e-3, atol=1e-2)


def test_centroid_epsf_exceptions():
    data = np.ones((5, 5), dtype=float)
    mask = np.zeros((4, 5), dtype=int)
    mask[2, 2] = 1

    with pytest.raises(ValueError):
        centroid_epsf(data, mask)
    with pytest.raises(ValueError):
        centroid_epsf(data, shift_val=-1)
    with pytest.raises(ValueError):
        centroid_epsf(data, oversampling=-1)

    data = np.ones((21, 21), dtype=float)
    data[10, 10] = np.inf
    with pytest.raises(ValueError):
        centroid_epsf(data)
