# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the background_2d module.
"""

import itertools

import astropy.units as u
import numpy as np
import pytest
from astropy.nddata import CCDData, NDData
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_equal

from photutils.background.background_2d import Background2D
from photutils.background.core import MeanBackground
from photutils.background.interpolators import (BkgIDWInterpolator,
                                                BkgZoomInterpolator)
from photutils.utils._optional_deps import HAS_MATPLOTLIB, HAS_SCIPY

DATA = np.ones((100, 100))
BKG_RMS = np.zeros((100, 100))
BKG_MESH = np.ones((4, 4))
BKG_RMS_MESH = np.zeros((4, 4))
PADBKG_MESH = np.ones((5, 5))
PADBKG_RMS_MESH = np.zeros((5, 5))
FILTER_SIZES = [(1, 1), (3, 3)]
INTERPOLATORS = [BkgZoomInterpolator(), BkgIDWInterpolator()]

DATA1 = DATA << u.ct
DATA2 = NDData(DATA, unit=None)
DATA3 = NDData(DATA, unit=u.ct)
DATA4 = CCDData(DATA, unit=u.ct)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
class TestBackground2D:
    @pytest.mark.parametrize(('filter_size', 'interpolator'),
                             list(itertools.product(FILTER_SIZES,
                                                    INTERPOLATORS)))
    def test_background(self, filter_size, interpolator):
        bkg = Background2D(DATA, (25, 25), filter_size=filter_size,
                           interpolator=interpolator)
        assert_allclose(bkg.background, DATA)
        assert_allclose(bkg.background_rms, BKG_RMS)
        assert_allclose(bkg.background_mesh, BKG_MESH)
        assert_allclose(bkg.background_rms_mesh, BKG_RMS_MESH)
        assert bkg.background_median == 1.0
        assert bkg.background_rms_median == 0.0

    @pytest.mark.parametrize('data', [DATA1, DATA3, DATA4])
    def test_background_nddata(self, data):
        """Test with NDData and CCDData, and also test units."""
        bkg = Background2D(data, (25, 25), filter_size=3)
        assert isinstance(bkg.background, u.Quantity)
        assert isinstance(bkg.background_rms, u.Quantity)
        assert isinstance(bkg.background_median, u.Quantity)
        assert isinstance(bkg.background_rms_median, u.Quantity)

        bkg = Background2D(DATA2, (25, 25), filter_size=3)
        assert_allclose(bkg.background, DATA)
        assert_allclose(bkg.background_rms, BKG_RMS)
        assert_allclose(bkg.background_mesh, BKG_MESH)
        assert_allclose(bkg.background_rms_mesh, BKG_RMS_MESH)
        assert bkg.background_median == 1.0
        assert bkg.background_rms_median == 0.0

    @pytest.mark.parametrize('interpolator', INTERPOLATORS)
    def test_background_rect(self, interpolator):
        """
        Regression test for interpolators with non-square input data.
        """
        data = np.arange(12).reshape(3, 4)
        rms = np.zeros((3, 4))
        bkg = Background2D(data, (1, 1), filter_size=1,
                           interpolator=interpolator)
        assert_allclose(bkg.background, data, atol=0.005)
        assert_allclose(bkg.background_rms, rms)
        assert_allclose(bkg.background_mesh, data)
        assert_allclose(bkg.background_rms_mesh, rms)
        assert bkg.background_median == 5.5
        assert bkg.background_rms_median == 0.0

    @pytest.mark.parametrize('interpolator', INTERPOLATORS)
    def test_background_nonconstant(self, interpolator):
        data = np.copy(DATA)
        data[25:50, 50:75] = 10.0
        bkg_low_res = np.copy(BKG_MESH)
        bkg_low_res[1, 2] = 10.0
        bkg1 = Background2D(data, (25, 25), filter_size=(1, 1),
                            interpolator=interpolator)
        assert_allclose(bkg1.background_mesh, bkg_low_res)
        assert bkg1.background.shape == data.shape
        bkg2 = Background2D(data, (25, 25), filter_size=(1, 1),
                            edge_method='pad', interpolator=interpolator)
        assert_allclose(bkg2.background_mesh, bkg_low_res)
        assert bkg2.background.shape == data.shape

    def test_no_sigma_clipping(self):
        data = np.copy(DATA)
        data[10, 10] = 100.0
        bkg1 = Background2D(data, (25, 25), filter_size=(1, 1),
                            bkg_estimator=MeanBackground())
        bkg2 = Background2D(data, (25, 25), filter_size=(1, 1),
                            sigma_clip=None, bkg_estimator=MeanBackground())

        assert bkg2.background_mesh[0, 0] > bkg1.background_mesh[0, 0]

    @pytest.mark.parametrize('filter_size', FILTER_SIZES)
    def test_resizing(self, filter_size):
        bkg1 = Background2D(DATA, (23, 22), filter_size=filter_size,
                            bkg_estimator=MeanBackground(), edge_method='crop')
        bkg2 = Background2D(DATA, (23, 22), filter_size=filter_size,
                            bkg_estimator=MeanBackground(), edge_method='pad')
        assert_allclose(bkg1.background, bkg2.background, rtol=2e-6)
        assert_allclose(bkg1.background_rms, bkg2.background_rms)

        shape1 = (128, 256)
        shape2 = (129, 256)
        box_size = (16, 16)
        data1 = np.ones(shape1)
        data2 = np.ones(shape2)
        bkg1 = Background2D(data1, box_size)
        bkg2 = Background2D(data2, box_size)
        assert bkg1.background_mesh.shape == (8, 16)
        assert bkg2.background_mesh.shape == (9, 16)
        assert bkg1.background.shape == shape1
        assert bkg2.background.shape == shape2

    @pytest.mark.parametrize('box_size', ([(25, 25), (23, 22)]))
    def test_background_mask(self, box_size):
        """
        Test with an input mask.  Note that box_size=(23, 22) tests the
        resizing of the image and mask.
        """
        data = np.copy(DATA)
        data[25:50, 25:50] = 100.0
        mask = np.zeros(DATA.shape, dtype=bool)
        mask[25:50, 25:50] = True
        bkg = Background2D(data, box_size, filter_size=(1, 1), mask=mask,
                           bkg_estimator=MeanBackground())
        assert_allclose(bkg.background, DATA, rtol=2.0e-5)
        assert_allclose(bkg.background_rms, BKG_RMS)

        # test edge crop with mask
        bkg2 = Background2D(data, box_size, filter_size=(1, 1), mask=mask,
                            bkg_estimator=MeanBackground(), edge_method='crop')
        assert_allclose(bkg2.background, DATA, rtol=2.0e-5)

    def test_mask(self):
        data = np.copy(DATA)
        data[25:50, 25:50] = 100.0
        mask = np.zeros(DATA.shape, dtype=bool)
        mask[25:50, 25:50] = True
        bkg1 = Background2D(data, (25, 25), filter_size=(1, 1), mask=None,
                            bkg_estimator=MeanBackground())

        assert_equal(bkg1.background_mesh, bkg1.background_mesh_masked)
        assert_equal(bkg1.background_rms_mesh, bkg1.background_rms_mesh_masked)
        assert np.count_nonzero(np.isnan(bkg1.mesh_nmasked)) == 0

        bkg2 = Background2D(data, (25, 25), filter_size=(1, 1), mask=mask,
                            bkg_estimator=MeanBackground())

        assert (np.count_nonzero(~np.isnan(bkg2.background_mesh_masked))
                < bkg2.nboxes_tot)
        assert (np.count_nonzero(~np.isnan(bkg2.background_rms_mesh_masked))
                < bkg2.nboxes_tot)
        assert np.count_nonzero(np.isnan(bkg2.mesh_nmasked)) == 1

    @pytest.mark.parametrize('fill_value', [0.0, np.nan, -1.0])
    def test_coverage_mask(self, fill_value):
        data = np.copy(DATA)
        data[:50, :50] = np.nan
        mask = np.isnan(data)

        with pytest.warns(AstropyUserWarning,
                          match='Input data contains invalid values'):
            bkg1 = Background2D(data, (25, 25), filter_size=(1, 1),
                                coverage_mask=mask, fill_value=fill_value,
                                bkg_estimator=MeanBackground())
        assert_equal(bkg1.background[:50, :50], fill_value)
        assert_equal(bkg1.background_rms[:50, :50], fill_value)

        # test combination of masks
        mask = np.zeros(DATA.shape, dtype=bool)
        coverage_mask = np.zeros(DATA.shape, dtype=bool)
        mask[:50, :25] = True
        coverage_mask[:50, 25:50] = True
        with pytest.warns(AstropyUserWarning,
                          match='Input data contains invalid values'):
            bkg2 = Background2D(data, (25, 25), filter_size=(1, 1), mask=mask,
                                coverage_mask=mask, fill_value=0.0,
                                bkg_estimator=MeanBackground())
        assert_equal(bkg1.background_mesh, bkg2.background_mesh)
        assert_equal(bkg1.background_rms_mesh, bkg2.background_rms_mesh)

    def test_mask_nonfinite(self):
        data = DATA.copy()
        data[0, 0:50] = np.nan
        with pytest.warns(AstropyUserWarning,
                          match='Input data contains invalid values'):
            bkg = Background2D(data, (25, 25), filter_size=(1, 1))
        assert_allclose(bkg.background, DATA, rtol=1e-5)

    def test_masked_array(self):
        data = DATA.copy()
        data[0, 0:50] = True
        mask = np.zeros(DATA.shape, dtype=bool)
        mask[0, 0:50] = True
        data_ma1 = np.ma.MaskedArray(DATA, mask=mask)
        data_ma2 = np.ma.MaskedArray(data, mask=mask)

        bkg1 = Background2D(data, (25, 25), filter_size=(1, 1))
        bkg2 = Background2D(data_ma1, (25, 25), filter_size=(1, 1))
        bkg3 = Background2D(data_ma2, (25, 25), filter_size=(1, 1))
        assert_allclose(bkg1.background, bkg2.background, rtol=1e-5)
        assert_allclose(bkg2.background, bkg3.background, rtol=1e-5)

    def test_completely_masked(self):
        with pytest.raises(ValueError):
            mask = np.ones(DATA.shape, dtype=bool)
            Background2D(DATA, (25, 25), mask=mask)

    def test_zero_padding(self):
        """Test case where padding is added only on one axis."""
        bkg = Background2D(DATA, (25, 22), filter_size=(1, 1))
        assert_allclose(bkg.background, DATA, rtol=1e-5)
        assert_allclose(bkg.background_rms, BKG_RMS)
        assert bkg.background_median == 1.0
        assert bkg.background_rms_median == 0.0

    def test_exclude_percentile(self):
        """Only meshes greater than filter_threshold are filtered."""
        data = np.copy(DATA)
        data[0:50, 0:50] = np.nan
        with pytest.warns(AstropyUserWarning,
                          match='Input data contains invalid values'):
            bkg = Background2D(data, (25, 25), filter_size=(1, 1),
                               exclude_percentile=100.0)
        assert len(bkg._box_idx) == 12

    def test_filter_threshold(self):
        """Only meshes greater than filter_threshold are filtered."""
        data = np.copy(DATA)
        data[25:50, 50:75] = 10.0
        bkg = Background2D(data, (25, 25), filter_size=(3, 3),
                           filter_threshold=9.0)
        assert_allclose(bkg.background, DATA)
        assert_allclose(bkg.background_mesh, BKG_MESH)
        bkg2 = Background2D(data, (25, 25), filter_size=(3, 3),
                            filter_threshold=11.0)  # no filtering
        assert bkg2.background_mesh[1, 2] == 10

    def test_filter_threshold_high(self):
        """No filtering because filter_threshold is too large."""
        data = np.copy(DATA)
        data[25:50, 50:75] = 10.0
        ref_data = np.copy(BKG_MESH)
        ref_data[1, 2] = 10.0
        b = Background2D(data, (25, 25), filter_size=(3, 3),
                         filter_threshold=100.0)
        assert_allclose(b.background_mesh, ref_data)

    def test_filter_threshold_nofilter(self):
        """No filtering because filter_size is (1, 1)."""
        data = np.copy(DATA)
        data[25:50, 50:75] = 10.0
        ref_data = np.copy(BKG_MESH)
        ref_data[1, 2] = 10.0
        b = Background2D(data, (25, 25), filter_size=(1, 1),
                         filter_threshold=1.0)
        assert_allclose(b.background_mesh, ref_data)

    def test_scalar_sizes(self):
        bkg1 = Background2D(DATA, (25, 25), filter_size=(3, 3))
        bkg2 = Background2D(DATA, 25, filter_size=3)
        assert_allclose(bkg1.background, bkg2.background)
        assert_allclose(bkg1.background_rms, bkg2.background_rms)

    def test_invalid_box_size(self):
        with pytest.raises(ValueError):
            Background2D(DATA, (5, 5, 3))

    def test_invalid_filter_size(self):
        with pytest.raises(ValueError):
            Background2D(DATA, (5, 5), filter_size=(3, 3, 3))

    def test_invalid_exclude_percentile(self):
        with pytest.raises(ValueError):
            Background2D(DATA, (5, 5), exclude_percentile=-1)

        with pytest.raises(ValueError):
            Background2D(DATA, (5, 5), exclude_percentile=101)

    def test_mask_nomask(self):
        bkg = Background2D(DATA, (25, 25), filter_size=(1, 1),
                           mask=np.ma.nomask)
        assert bkg.mask is None

        bkg = Background2D(DATA, (25, 25), filter_size=(1, 1),
                           coverage_mask=np.ma.nomask)
        assert bkg.coverage_mask is None

    def test_invalid_mask(self):
        with pytest.raises(ValueError):
            Background2D(DATA, (25, 25), filter_size=(1, 1),
                         mask=np.zeros((2, 2)))

        with pytest.raises(ValueError):
            Background2D(DATA, (25, 25), filter_size=(1, 1),
                         mask=np.zeros((2, 2, 2)))

    def test_invalid_coverage_mask(self):
        with pytest.raises(ValueError):
            Background2D(DATA, (25, 25), filter_size=(1, 1),
                         coverage_mask=np.zeros((2, 2)))

        with pytest.raises(ValueError):
            Background2D(DATA, (25, 25), filter_size=(1, 1),
                         coverage_mask=np.zeros((2, 2, 2)))

    def test_invalid_edge_method(self):
        with pytest.raises(ValueError):
            Background2D(DATA, (23, 22), filter_size=(1, 1),
                         edge_method='not_valid')

    def test_invalid_mesh_idx_len(self):
        with pytest.raises(ValueError):
            bkg = Background2D(DATA, (25, 25), filter_size=(1, 1))
            bkg._make_2d_array(np.arange(3))

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot_meshes(self):
        """
        This test should run without any errors, but there is no return
        value.
        """
        bkg = Background2D(DATA, (25, 25))
        bkg.plot_meshes(outlines=True)

    def test_crop(self):
        data = np.ones((300, 500))
        bkg = Background2D(data, (74, 99), edge_method='crop')
        assert_allclose(bkg.background_median, 1.0)
        assert_allclose(bkg.background_rms_median, 0.0)
        assert_allclose(bkg.background_mesh.shape, (4, 5))

    def test_repr(self):
        data = np.ones((300, 500))
        bkg = Background2D(data, (74, 99), edge_method='crop')
        cls_repr = repr(bkg)
        assert cls_repr.startswith(f'{bkg.__class__.__name__}')


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
def test_bkgzoominterp_clip():
    bkg = Background2D(np.ones((300, 300)), 100)
    mesh = np.array([[0.01, 0.01, 0.02],
                     [0.01, 0.02, 0.03],
                     [0.03, 0.03, 12.9]])

    interp1 = BkgZoomInterpolator(clip=False)
    zoom1 = interp1(mesh, bkg)

    interp2 = BkgZoomInterpolator(clip=True)
    zoom2 = interp2(mesh, bkg)

    minval = np.min(mesh)
    maxval = np.max(mesh)
    assert np.min(zoom1) < minval
    assert np.max(zoom1) > maxval
    assert np.min(zoom2) == minval
    assert np.max(zoom2) == maxval
