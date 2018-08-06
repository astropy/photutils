# Licensed under a 3-clause BSD style license - see LICENSE.rst

import itertools

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

from ..core import MeanBackground
from ..background_2d import (BkgZoomInterpolator, BkgIDWInterpolator,
                             Background2D)

try:
    import scipy    # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib    # noqa
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


DATA = np.ones((100, 100))
BKG_RMS = np.zeros((100, 100))
BKG_MESH = np.ones((4, 4))
BKG_RMS_MESH = np.zeros((4, 4))
PADBKG_MESH = np.ones((5, 5))
PADBKG_RMS_MESH = np.zeros((5, 5))
FILTER_SIZES = [(1, 1), (3, 3)]
INTERPOLATORS = [BkgZoomInterpolator(), BkgIDWInterpolator()]


@pytest.mark.skipif('not HAS_SCIPY')
class TestBackground2D:
    @pytest.mark.parametrize(('filter_size', 'interpolator'),
                             list(itertools.product(FILTER_SIZES,
                                                    INTERPOLATORS)))
    def test_background(self, filter_size, interpolator):
        b = Background2D(DATA, (25, 25), filter_size=filter_size,
                         interpolator=interpolator)
        assert_allclose(b.background, DATA)
        assert_allclose(b.background_rms, BKG_RMS)
        assert_allclose(b.background_mesh, BKG_MESH)
        assert_allclose(b.background_rms_mesh, BKG_RMS_MESH)
        assert b.background_median == 1.0
        assert b.background_rms_median == 0.0

    @pytest.mark.parametrize('interpolator', INTERPOLATORS)
    def test_background_nonconstant(self, interpolator):
        data = np.copy(DATA)
        data[25:50, 50:75] = 10.
        bkg_low_res = np.copy(BKG_MESH)
        bkg_low_res[1, 2] = 10.
        b1 = Background2D(data, (25, 25), filter_size=(1, 1),
                          interpolator=interpolator)
        assert_allclose(b1.background_mesh, bkg_low_res)
        assert b1.background.shape == data.shape
        b2 = Background2D(data, (25, 25), filter_size=(1, 1),
                          edge_method='pad', interpolator=interpolator)
        assert_allclose(b2.background_mesh, bkg_low_res)
        assert b2.background.shape == data.shape

    def test_no_sigma_clipping(self):
        data = np.copy(DATA)
        data[10, 10] = 100.
        b1 = Background2D(data, (25, 25), filter_size=(1, 1),
                          bkg_estimator=MeanBackground())
        b2 = Background2D(data, (25, 25), filter_size=(1, 1), sigma_clip=None,
                          bkg_estimator=MeanBackground())

        assert b2.background_mesh[0, 0] > b1.background_mesh[0, 0]

    @pytest.mark.parametrize('filter_size', FILTER_SIZES)
    def test_resizing(self, filter_size):
        b1 = Background2D(DATA, (23, 22), filter_size=filter_size,
                          bkg_estimator=MeanBackground(), edge_method='crop')
        b2 = Background2D(DATA, (23, 22), filter_size=filter_size,
                          bkg_estimator=MeanBackground(), edge_method='pad')
        assert_allclose(b1.background, b2.background)
        assert_allclose(b1.background_rms, b2.background_rms)

    @pytest.mark.parametrize('box_size', ([(25, 25), (23, 22)]))
    def test_background_mask(self, box_size):
        """
        Test with an input mask.  Note that box_size=(23, 22) tests the
        resizing of the image and mask.
        """

        data = np.copy(DATA)
        data[25:50, 25:50] = 100.
        mask = np.zeros_like(DATA, dtype=np.bool)
        mask[25:50, 25:50] = True
        b = Background2D(data, box_size, filter_size=(1, 1), mask=mask,
                         bkg_estimator=MeanBackground())
        assert_allclose(b.background, DATA)
        assert_allclose(b.background_rms, BKG_RMS)

        # test edge crop with
        b2 = Background2D(data, box_size, filter_size=(1, 1), mask=mask,
                          bkg_estimator=MeanBackground(), edge_method='crop')
        assert_allclose(b2.background, DATA)

    def test_mask(self):
        data = np.copy(DATA)
        data[25:50, 25:50] = 100.
        mask = np.zeros_like(DATA, dtype=np.bool)
        mask[25:50, 25:50] = True
        b1 = Background2D(data, (25, 25), filter_size=(1, 1), mask=None,
                          bkg_estimator=MeanBackground())

        assert_equal(b1.background_mesh, b1.background_mesh_ma)
        assert_equal(b1.background_rms_mesh, b1.background_rms_mesh_ma)
        assert not np.ma.is_masked(b1.mesh_nmasked)

        b2 = Background2D(data, (25, 25), filter_size=(1, 1), mask=mask,
                          bkg_estimator=MeanBackground())

        assert np.ma.count(b2.background_mesh_ma) < b2.nboxes
        assert np.ma.count(b2.background_rms_mesh_ma) < b2.nboxes
        assert np.ma.is_masked(b2.mesh_nmasked)

    def test_completely_masked(self):
        with pytest.raises(ValueError):
            mask = np.ones_like(DATA, dtype=np.bool)
            Background2D(DATA, (25, 25), mask=mask)

    def test_zero_padding(self):
        """Test case where padding is added only on one axis."""

        b = Background2D(DATA, (25, 22), filter_size=(1, 1))
        assert_allclose(b.background, DATA)
        assert_allclose(b.background_rms, BKG_RMS)
        assert b.background_median == 1.0
        assert b.background_rms_median == 0.0

    def test_filter_threshold(self):
        """Only meshes greater than filter_threshold are filtered."""

        data = np.copy(DATA)
        data[25:50, 50:75] = 10.
        b = Background2D(data, (25, 25), filter_size=(3, 3),
                         filter_threshold=9.)
        assert_allclose(b.background, DATA)
        assert_allclose(b.background_mesh, BKG_MESH)
        b2 = Background2D(data, (25, 25), filter_size=(3, 3),
                          filter_threshold=11.)   # no filtering
        assert b2.background_mesh[1, 2] == 10

    def test_filter_threshold_high(self):
        """No filtering because filter_threshold is too large."""

        data = np.copy(DATA)
        data[25:50, 50:75] = 10.
        ref_data = np.copy(BKG_MESH)
        ref_data[1, 2] = 10.
        b = Background2D(data, (25, 25), filter_size=(3, 3),
                         filter_threshold=100.)
        assert_allclose(b.background_mesh, ref_data)

    def test_filter_threshold_nofilter(self):
        """No filtering because filter_size is (1, 1)."""

        data = np.copy(DATA)
        data[25:50, 50:75] = 10.
        ref_data = np.copy(BKG_MESH)
        ref_data[1, 2] = 10.
        b = Background2D(data, (25, 25), filter_size=(1, 1),
                         filter_threshold=1.)
        assert_allclose(b.background_mesh, ref_data)

    def test_scalar_sizes(self):
        b1 = Background2D(DATA, (25, 25), filter_size=(3, 3))
        b2 = Background2D(DATA, 25, filter_size=3)
        assert_allclose(b1.background, b2.background)
        assert_allclose(b1.background_rms, b2.background_rms)

    def test_exclude_percentile(self):
        with pytest.raises(ValueError):
            Background2D(DATA, (5, 5), exclude_percentile=-1)

        with pytest.raises(ValueError):
            Background2D(DATA, (5, 5), exclude_percentile=101)

    def test_mask_badshape(self):
        with pytest.raises(ValueError):
            Background2D(DATA, (25, 25), filter_size=(1, 1),
                         mask=np.zeros((2, 2)))

    def test_invalid_edge_method(self):
        with pytest.raises(ValueError):
            Background2D(DATA, (23, 22), filter_size=(1, 1),
                         edge_method='not_valid')

    def test_invalid_mesh_idx_len(self):
        with pytest.raises(ValueError):
            bkg = Background2D(DATA, (25, 25), filter_size=(1, 1))
            bkg._make_2d_array(np.arange(3))

    @pytest.mark.skipif('not HAS_MATPLOTLIB')
    def test_plot_meshes(self):
        """
        This test should run without any errors, but there is no return
        value.
        """

        b = Background2D(DATA, (25, 25))
        b.plot_meshes(outlines=True)
