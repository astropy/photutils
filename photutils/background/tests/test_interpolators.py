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
                                                BkgZoomInterpolator)


def test_zoom_interp():
    data = np.ones((300, 300))
    bkg = Background2D(data, 100)
    mesh = np.array([[0.01, 0.01, 0.02],
                     [0.01, 0.02, 0.03],
                     [0.03, 0.03, 12.9]])

    interp = BkgZoomInterpolator(clip=False)
    zoom = interp(mesh, **bkg._interp_kwargs)
    assert zoom.shape == (300, 300)

    with pytest.warns(AstropyDeprecationWarning):
        bkg = Background2D(data, 100, edge_method='crop')
    zoom2 = interp(mesh, **bkg._interp_kwargs)
    assert zoom2.shape == (300, 300)

    # test with units
    unit = u.nJy
    bkg = Background2D(data << unit, 100)
    interp = BkgZoomInterpolator(clip=False)
    zoom = interp(mesh << unit, **bkg._interp_kwargs)
    assert zoom.shape == (300, 300)

    # test repr
    cls_repr = repr(interp)
    assert cls_repr.startswith(f'{interp.__class__.__name__}')


def test_zoom_interp_clip():
    bkg = Background2D(np.ones((300, 300)), 100)
    mesh = np.array([[0.01, 0.01, 0.02],
                     [0.01, 0.02, 0.03],
                     [0.03, 0.03, 12.9]])

    interp1 = BkgZoomInterpolator(clip=False)
    zoom1 = interp1(mesh, **bkg._interp_kwargs)

    interp2 = BkgZoomInterpolator(clip=True)
    zoom2 = interp2(mesh, **bkg._interp_kwargs)

    minval = np.min(mesh)
    maxval = np.max(mesh)
    assert np.min(zoom1) < minval
    assert np.max(zoom1) > maxval
    assert np.min(zoom2) == minval
    assert np.max(zoom2) == maxval


def test_idw_interp():
    data = np.ones((300, 300))
    interp = BkgIDWInterpolator()
    bkg = Background2D(data, 100, interpolator=interp)
    mesh = np.array([[0.01, 0.01, 0.02],
                     [0.01, 0.02, 0.03],
                     [0.03, 0.03, 12.9]])

    zoom = interp(mesh, **bkg._interp_kwargs)
    assert zoom.shape == (300, 300)

    # test with units
    unit = u.nJy
    bkg = Background2D(data << unit, 100, interpolator=interp)
    zoom = interp(mesh << unit, **bkg._interp_kwargs)
    assert zoom.shape == (300, 300)

    # test repr
    cls_repr = repr(interp)
    assert cls_repr.startswith(f'{interp.__class__.__name__}')
