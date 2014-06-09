# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.table import Table
from .. import make_gaussian_image

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_make_gaussian_image():
    table = Table()
    table['amplitude'] = [1, 2, 3]
    table['x_0'] = [30, 50, 70.5]
    table['y_0'] = [50, 50, 50.5]
    table['sigma'] = [1, 2, 3.5]

    shape = (100, 100)
    image = make_gaussian_image(shape, table)

    assert image.shape == shape
    assert_allclose(image.sum(), table['amplitude'].sum())
