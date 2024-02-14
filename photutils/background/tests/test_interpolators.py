# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the interpolators module.
"""

import numpy as np
import pytest

from photutils.background.background_2d import Background2D
from photutils.background.interpolators import (BkgIDWInterpolator,
                                                BkgZoomInterpolator)
from photutils.utils._optional_deps import HAS_SCIPY


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_zoom_interp():
    bkg = Background2D(np.ones((300, 300)), 100)
    mesh = np.array([[0.01, 0.01, 0.02],
                     [0.01, 0.02, 0.03],
                     [0.03, 0.03, 12.9]])

    interp = BkgZoomInterpolator(clip=False)
    zoom = interp(mesh, bkg)
    assert zoom.shape == (300, 300)

    cls_repr = repr(interp)
    assert cls_repr.startswith(f'{interp.__class__.__name__}')


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_idw_interp():
    bkg = Background2D(np.ones((300, 300)), 100)
    mesh = np.array([[0.01, 0.01, 0.02],
                     [0.01, 0.02, 0.03],
                     [0.03, 0.03, 12.9]])

    interp = BkgIDWInterpolator()
    zoom = interp(mesh, bkg)
    assert zoom.shape == (300, 300)

    cls_repr = repr(interp)
    assert cls_repr.startswith(f'{interp.__class__.__name__}')
