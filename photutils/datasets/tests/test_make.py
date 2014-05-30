# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest, remote_data
from .. import make


def test_make_example_image():
    shape = (100, 100)
    n_sources = 42
    psf_width = 3
    image = make.make_example_image(shape, n_sources, psf_width)
    assert image.shape == shape
