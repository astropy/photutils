# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import pytest

from ..circle import CircularAperture

try:
    import matplotlib    # noqa
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


POSITIONS = [(-20, -20), (-20, 20), (20, -20)]


@pytest.mark.parametrize('position', POSITIONS)
def test_mask_cutout_no_overlap(position):
    data = np.ones((50, 50))
    aper = CircularAperture(position, r=10.)
    mask = aper.to_mask()[0]

    cutout = mask.cutout(data)
    assert cutout is None

    weighted_data = mask.multiply(data)
    assert weighted_data is None

    image = mask.to_image(data.shape)
    assert image is None


@pytest.mark.parametrize('position', POSITIONS)
def test_mask_cutout_partial_overlap(position):
    data = np.ones((50, 50))
    aper = CircularAperture(position, r=30.)
    mask = aper.to_mask()[0]

    cutout = mask.cutout(data)
    assert cutout.shape == mask.shape

    weighted_data = mask.multiply(data)
    assert weighted_data.shape == mask.shape

    image = mask.to_image(data.shape)
    assert image.shape == data.shape
