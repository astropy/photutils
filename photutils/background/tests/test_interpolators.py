# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the interpolators module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.background.background_2d import Background2D
from photutils.background.interpolators import (BkgIDWInterpolator,
                                                _BkgZoomInterpolator)


@pytest.fixture
def test_data():
    """
    Create test data for interpolator tests.
    """
    return np.ones((300, 300))


@pytest.fixture
def test_mesh():
    """
    Create test mesh for interpolator tests.
    """
    return np.array([[0.01, 0.01, 0.02],
                     [0.01, 0.02, 0.03],
                     [0.03, 0.03, 12.9]])


def test_zoom_interp(test_data, test_mesh):
    bkg = Background2D(test_data, 100)

    interp = _BkgZoomInterpolator(clip=False)
    zoom = interp(test_mesh, **bkg._interp_kwargs)
    assert zoom.shape == (300, 300)

    # test with units
    unit = u.nJy
    bkg = Background2D(test_data << unit, 100)
    interp = _BkgZoomInterpolator(clip=False)
    zoom = interp(test_mesh << unit, **bkg._interp_kwargs)
    assert zoom.shape == (300, 300)

    # test repr
    cls_repr = repr(interp)
    assert cls_repr.startswith(f'{interp.__class__.__name__}')


def test_zoom_interp_clip(test_data, test_mesh):
    bkg = Background2D(test_data, 100)

    interp1 = _BkgZoomInterpolator(clip=False)
    zoom1 = interp1(test_mesh, **bkg._interp_kwargs)

    interp2 = _BkgZoomInterpolator(clip=True)
    zoom2 = interp2(test_mesh, **bkg._interp_kwargs)

    minval = np.min(test_mesh)
    maxval = np.max(test_mesh)
    assert np.min(zoom1) < minval
    assert np.max(zoom1) > maxval
    assert np.min(zoom2) == minval
    assert np.max(zoom2) == maxval


def test_idw_interp(test_data, test_mesh):
    with pytest.warns(AstropyDeprecationWarning):
        interp = BkgIDWInterpolator()
    with pytest.warns(AstropyDeprecationWarning):
        bkg = Background2D(test_data, 100, interpolator=interp)

    zoom = interp(test_mesh, **bkg._interp_kwargs)
    assert zoom.shape == (300, 300)

    # test constant mesh data
    zoom = interp(np.ones_like(test_mesh), **bkg._interp_kwargs)
    assert np.all(zoom == 1)

    # test with units
    unit = u.nJy
    with pytest.warns(AstropyDeprecationWarning):
        bkg = Background2D(test_data << unit, 100, interpolator=interp)
    zoom = interp(test_mesh << unit, **bkg._interp_kwargs)
    assert zoom.shape == (300, 300)

    # test repr
    cls_repr = repr(interp)
    assert cls_repr.startswith(f'{interp.__class__.__name__}')
