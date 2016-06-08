# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
import itertools
from ..background import Background2D, BackgroundIDW2D, std_blocksum
from ..datasets import make_noise_image

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
FILTER_SIZES = [(1, 1), (3, 3)]
METHODS = ['mean', 'median', 'sextractor', 'mode_estimator']
EDGE_METHODS = ['pad', 'crop']


def custom_backfunc(data):
    """Same as method='mean'."""
    return np.ma.mean(data, axis=1)


@pytest.mark.skipif('not HAS_SCIPY')
class TestBackground2D(object):
    @pytest.mark.parametrize(('filter_size', 'method'),
                             list(itertools.product(FILTER_SIZES, METHODS)))
    def test_background(self, filter_size, method):
        b = Background2D(DATA, (25, 25), filter_size=filter_size,
                         method=method)
        assert_allclose(b.background, DATA)
        assert_allclose(b.background_rms, BKG_RMS)
        assert_allclose(b.background_mesh2d, BKG_MESH)
        assert_allclose(b.background_rms_mesh2d, BKG_RMS_MESH)
        assert b.background_median == 1.0
        assert b.background_rms_median == 0.0

    def test_background_nonconstant(self):
        data = np.copy(DATA)
        data[25:50, 50:75] = 10.
        bkg_low_res = np.copy(BKG_MESH)
        bkg_low_res[1, 2] = 10.
        b1 = Background2D(data, (25, 25), filter_size=(1, 1), method='mean')
        assert_allclose(b1.background_mesh2d, bkg_low_res)
        assert b1.background.shape == data.shape
        b2 = Background2D(data, (25, 25), filter_size=(1, 1), method='mean',
                          edge_method='pad')
        assert_allclose(b2.background_mesh2d, bkg_low_res)
        assert b2.background.shape == data.shape

    @pytest.mark.parametrize(('filter_size', 'method'),
                             list(itertools.product(FILTER_SIZES, METHODS)))
    def test_resizing(self, filter_size, method):
        b1 = Background2D(DATA, (22, 22), filter_size=filter_size,
                          edge_method='crop', method=method)
        b2 = Background2D(DATA, (22, 22), filter_size=filter_size,
                          edge_method='pad', method=method)
        assert_allclose(b1.background, b2.background)
        assert_allclose(b1.background_rms, b2.background_rms)

    @pytest.mark.parametrize('box_size', ([(25, 25), (22, 22)]))
    def test_background_mask(self, box_size):
        """
        Test with an input mask.  Note that box_size=(22, 22) tests the
        resizing of the image and mask.
        """

        data = np.copy(DATA)
        data[25:50, 25:50] = 100.
        mask = np.zeros_like(DATA, dtype=np.bool)
        mask[25:50, 25:50] = True
        b = Background2D(data, box_size, filter_size=(1, 1), mask=mask)
        assert_allclose(b.background, DATA)
        assert_allclose(b.background_rms, BKG_RMS)

        # test mask padding
        b2 = Background2D(data, (22, 22), filter_size=(1, 1), mask=mask,
                          edge_method='pad')
        assert_allclose(b2.background, DATA)

    @pytest.mark.parametrize('remove_masked', (['any', 'all', 'threshold']))
    def test_remove_masked(self, remove_masked):
        b = Background2D(DATA, (25, 25), remove_masked=remove_masked)
        assert_allclose(b.background, DATA)
        b2 = Background2D(DATA, (25, 25), remove_masked='_none')
        assert_allclose(b2.background, DATA)

        # test if data is completely masked
        with pytest.raises(ValueError):
            mask = np.ones_like(DATA, dtype=np.bool)
            Background2D(DATA, (25, 25), mask=mask,
                         remove_masked=remove_masked)

    def test_filter_threshold(self):
        """Only meshes greater than filter_threshold are filtered."""
        data = np.copy(DATA)
        data[25:50, 50:75] = 10.
        b = Background2D(data, (25, 25), filter_size=(3, 3),
                         filter_threshold=9.)
        assert_allclose(b.background, DATA)
        assert_allclose(b.background_mesh2d, BKG_MESH)
        b2 = Background2D(data, (25, 25), filter_size=(3, 3),
                          filter_threshold=11.)   # no filtering
        assert b2.background_mesh2d[1, 2] == 10

    def test_filter_threshold_high(self):
        """No filtering because filter_threshold is too large."""
        data = np.copy(DATA)
        data[25:50, 50:75] = 10.
        ref_data = np.copy(BKG_MESH)
        ref_data[1, 2] = 10.
        b = Background2D(data, (25, 25), filter_size=(3, 3),
                         filter_threshold=100.)
        assert_allclose(b.background_mesh2d, ref_data)

    def test_filter_threshold_nofilter(self):
        """No filtering because filter_size is (1, 1)."""
        data = np.copy(DATA)
        data[25:50, 50:75] = 10.
        ref_data = np.copy(BKG_MESH)
        ref_data[1, 2] = 10.
        b = Background2D(data, (25, 25), filter_size=(1, 1),
                         filter_threshold=1.)
        assert_allclose(b.background_mesh2d, ref_data)

    def test_scalar_sizes(self):
        b1 = Background2D(DATA, (25, 25), filter_size=(3, 3), method='mean')
        b2 = Background2D(DATA, 25, filter_size=3, method='mean')
        assert_allclose(b1.background, b2.background)
        assert_allclose(b1.background_rms, b2.background_rms)

    def test_meshpix_threshold(self):
        with pytest.raises(ValueError):
            Background2D(DATA, (5, 5), remove_masked='threshold',
                         meshpix_threshold=26)

    def test_custom_method(self):
        b0 = Background2D(DATA, (25, 25), filter_size=(3, 3), method='mean')
        b1 = Background2D(DATA, (25, 25), filter_size=(3, 3), method='custom',
                          backfunc=custom_backfunc)
        assert_allclose(b0.background, b1.background)
        assert_allclose(b0.background_rms, b1.background_rms)

    def test_mask_badshape(self):
        with pytest.raises(ValueError):
            Background2D(DATA, (25, 25), filter_size=(1, 1),
                         mask=np.zeros((2, 2)))

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            Background2D(DATA, (25, 25), filter_size=(1, 1),
                         method='not_valid')

    def test_invalid_edge_method(self):
        with pytest.raises(ValueError):
            Background2D(DATA, (22, 22), filter_size=(1, 1),
                         edge_method='not_valid')

    def test_invalid_remove_masked(self):
        with pytest.raises(ValueError):
            Background2D(DATA, (22, 22), filter_size=(1, 1),
                         remove_masked='not_valid')

    def test_custom_return_nonarray(self):
        def backfunc(data):
            return 1
        with pytest.raises(ValueError):
            Background2D(DATA, (25, 25), filter_size=(3, 3),
                         method='custom', backfunc=backfunc)

    def test_custom_return_masked_array(self):
        def backfunc(data):
            return np.ma.masked_array([1], mask=False)
        with pytest.raises(ValueError):
            Background2D(DATA, (25, 25), filter_size=(3, 3),
                         method='custom', backfunc=backfunc)

    def test_custom_return_badshape(self):
        def backfunc(data):
            return np.ones((2, 2))
        with pytest.raises(ValueError):
            Background2D(DATA, (25, 25), filter_size=(3, 3),
                         method='custom', backfunc=backfunc)

    def test_plot_meshes(self):
        """
        This test should run without any errors, but there is no return
        value.
        """

        b = Background2D(DATA, (25, 25))
        b.plot_meshes(outlines=True)


@pytest.mark.skipif('not HAS_SCIPY')
class TestBackgroundIDW2D(object):
    def test_background_idw(self):
        b = BackgroundIDW2D(DATA, (25, 25), filter_size=(1, 1), method='mean')
        assert_allclose(b.background, DATA)
        assert_allclose(b.background_rms, BKG_RMS)
        assert_allclose(b.background_mesh2d, BKG_MESH)
        assert_allclose(b.background_rms_mesh2d, BKG_RMS_MESH)
        assert b.background_median == 1.0
        assert b.background_rms_median == 0.0

    def test_background_idw_nonconstant(self):
        data = np.copy(DATA)
        data[25:50, 50:75] = 10.
        bkg_low_res = np.copy(BKG_MESH)
        bkg_low_res[1, 2] = 10.
        b = BackgroundIDW2D(data, (25, 25), filter_size=(1, 1), method='mean')
        assert_allclose(b.background_mesh2d, bkg_low_res)
        assert b.background.shape == data.shape


class TestStdBlocksum(object):
    stddev = 5
    data = make_noise_image((100, 100), mean=0, stddev=stddev,
                            random_state=12345)
    block_sizes = np.array([5, 7, 10])
    stds = std_blocksum(data, block_sizes)
    expected = np.array([stddev, stddev, stddev])
    assert_allclose(stds / block_sizes, expected, atol=0.2)

    mask = np.zeros_like(data, dtype=np.bool)
    mask[25:50, 25:50] = True
    stds2 = std_blocksum(data, block_sizes, mask=mask)
    assert_allclose(stds2 / block_sizes, expected, atol=0.3)
