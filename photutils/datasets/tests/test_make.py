# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.table import Table
from .. import make_gaussian_sources

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_make_gaussian_image():
    table = Table()
    table['flux'] = [1, 2, 3]
    table['x_mean'] = [30, 50, 70.5]
    table['y_mean'] = [50, 50, 50.5]
    table['x_stddev'] = [1, 2, 3.5]
    table['y_stddev'] = [2, 1, 3.5]
    table['theta'] = np.array([0., 30, 50]) * np.pi / 180.

    shape = (100, 100)
    image = make_gaussian_sources(shape, table)

    assert image.shape == shape
    assert_allclose(image.sum(), table['flux'].sum())
