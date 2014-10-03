# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest
import os.path as op
import itertools
import numpy as np
from astropy.table import Table
from numpy.testing import assert_allclose
from ..findstars import daofind, irafstarfind
from photutils.datasets import make_100gaussians_image

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import skimage
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


DATA = make_100gaussians_image()
THRESHOLDS = [8.0, 10.0]
FWHMS = [1.0, 1.5, 2.0]


@pytest.mark.skipif('not HAS_SCIPY')
class TestDAOFind(object):
    @pytest.mark.parametrize(('threshold', 'fwhm'),
                             list(itertools.product(THRESHOLDS, FWHMS)))
    def test_daofind(self, threshold, fwhm):
        t = daofind(DATA, threshold, fwhm, sigma_radius=1.5)
        datafn = ('daofind_test_thresh{0:04.1f}_fwhm{1:04.1f}'
                  '.txt'.format(threshold, fwhm))
        datafn = op.join(op.dirname(op.abspath(__file__)), 'data', datafn)
        t_ref = Table.read(datafn, format='ascii')
        assert_allclose(np.array(t).astype(np.float),
                        np.array(t_ref).astype(np.float))


@pytest.mark.skipif('not HAS_SCIPY')
@pytest.mark.skipif('not HAS_SKIMAGE')
class TestIRAFStarFind(object):
    @pytest.mark.parametrize(('threshold', 'fwhm'),
                             list(itertools.product(THRESHOLDS, FWHMS)))
    def test_irafstarfind(self, threshold, fwhm):
        t = irafstarfind(DATA, threshold, fwhm, sigma_radius=1.5)
        datafn = ('irafstarfind_test_thresh{0:04.1f}_fwhm{1:04.1f}'
                  '.txt'.format(threshold, fwhm))
        datafn = op.join(op.dirname(op.abspath(__file__)), 'data', datafn)
        t_ref = Table.read(datafn, format='ascii')
        assert_allclose(np.array(t).astype(np.float),
                        np.array(t_ref).astype(np.float))
