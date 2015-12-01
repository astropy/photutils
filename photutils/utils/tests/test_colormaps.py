# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest
from ..colormaps import random_cmap


def test_colormap():
    cmap = random_cmap(100, random_state=12345)
    assert cmap(0) == (0., 0., 0., 1.0)


def test_colormap_background():
    cmap = random_cmap(100, background_color='white', random_state=12345)
    assert cmap(0) == (1., 1., 1., 1.0)


def test_invalid_background():
    with pytest.raises(ValueError):
        random_cmap(100, background_color='invalid', random_state=12345)
