# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
import itertools
from ..background import Background

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


DATA = np.ones((100, 100))
BKG_RMS = np.zeros((100, 100))
BKG_MESH = np.ones((4, 4))
BKG_RMS_MESH = np.zeros((4, 4))
PADBKG_MESH = np.ones((5, 5))
PADBKG_RMS_MESH = np.zeros((5, 5))
FILTER_SHAPES = [(1, 1), (3, 3)]
METHODS = ['mean', 'median', 'sextractor', 'mode_estimate']


def custom_backfunc(data):
    """Same as method='mean'."""
    bkg = np.ma.mean(data, axis=2)
    return np.ma.filled(bkg, np.ma.median(bkg))


@pytest.mark.skipif('not HAS_SCIPY')
class TestBackground(object):
    def test_mask_badshape(self):
        with pytest.raises(ValueError):
            Background(DATA, (25, 25), (1, 1), mask=np.zeros((2, 2)))

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            Background(DATA, (25, 25), (1, 1), method='not_valid')

    @pytest.mark.parametrize(('filter_shape', 'method'),
                             list(itertools.product(FILTER_SHAPES, METHODS)))
    def test_background(self, filter_shape, method):
        b = Background(DATA, (25, 25), filter_shape, method=method)
        assert_allclose(b.background, DATA)
        assert_allclose(b.background_rms, BKG_RMS)
        assert_allclose(b.background_mesh, BKG_MESH)
        assert_allclose(b.background_rms_mesh, BKG_RMS_MESH)
        assert b.background_median == 1.0
        assert b.background_rms_median == 0.0

    def test_background_nonconstant(self):
        data = np.copy(DATA)
        data[25:50, 50:75] = 10.
        mesh = np.copy(BKG_MESH)
        mesh[1, 2] = 10.
        b = Background(data, (25, 25), (1, 1), method='mean')
        assert_allclose(b.background_mesh, mesh)
        assert b.background.shape == data.shape

    @pytest.mark.parametrize(('filter_shape', 'method'),
                             list(itertools.product(FILTER_SHAPES, METHODS)))
    def test_background_padding(self, filter_shape, method):
        b = Background(DATA, (22, 22), filter_shape, method=method)
        assert_allclose(b.background, DATA)
        assert_allclose(b.background_rms, BKG_RMS)
        assert_allclose(b.background_mesh, PADBKG_MESH)
        assert_allclose(b.background_rms_mesh, PADBKG_RMS_MESH)

    @pytest.mark.parametrize('box_shape', ([(25, 25), (22, 22)]))
    def test_background_mask(self, box_shape):
        mask = np.zeros_like(DATA, dtype=np.bool)
        mask[0:50, :] = True
        b = Background(DATA, box_shape, (1, 1), mask=mask)
        maskdata = np.copy(DATA)
        maskdata[0:50, :] = 0.0
        assert_allclose(b.background, maskdata)
        assert_allclose(b.background_rms, BKG_RMS)

    def test_filter_threshold(self):
        """Only mesh > filter_threshold is filtered."""
        data = np.copy(DATA)
        data[25:50, 50:75] = 10.
        b = Background(data, (25, 25), (3, 3), filter_threshold=1.)
        assert_allclose(b.background, DATA)
        assert_allclose(b.background_mesh, BKG_MESH)

    def test_filter_threshold_high(self):
        """No filtering because filter_threshold is too large."""
        data = np.copy(DATA)
        data[25:50, 50:75] = 10.
        ref_data = np.copy(BKG_MESH)
        ref_data[1, 2] = 10.
        b = Background(data, (25, 25), (3, 3), filter_threshold=100.)
        assert_allclose(b.background_mesh, ref_data)

    def test_filter_threshold_nofilter(self):
        """No filtering because filter_shape is (1, 1)."""
        data = np.copy(DATA)
        data[25:50, 50:75] = 10.
        ref_data = np.copy(BKG_MESH)
        ref_data[1, 2] = 10.
        b = Background(data, (25, 25), (1, 1), filter_threshold=1.)
        assert_allclose(b.background_mesh, ref_data)

    def test_custom_method(self):
        b0 = Background(DATA, (25, 25), (3, 3), method='mean')
        b1 = Background(DATA, (25, 25), (3, 3), method='custom',
                        backfunc=custom_backfunc)
        assert_allclose(b0.background, b1.background)
        assert_allclose(b0.background_rms, b1.background_rms)

    def test_custom_return_nonarray(self):
        def backfunc(data):
            return 1
        with pytest.raises(ValueError):
            b = Background(DATA, (25, 25), (3, 3), method='custom',
                           backfunc=backfunc)
            b.background_mesh

    def test_custom_return_masked_array(self):
        def backfunc(data):
            return np.ma.masked_array([1], mask=False)
        with pytest.raises(ValueError):
            b = Background(DATA, (25, 25), (3, 3), method='custom',
                           backfunc=backfunc)
            b.background_mesh

    def test_custom_return_badshape(self):
        def backfunc(data):
            return np.ones((2, 2))
        with pytest.raises(ValueError):
            b = Background(DATA, (25, 25), (3, 3), method='custom',
                           backfunc=backfunc)
            b.background_mesh
