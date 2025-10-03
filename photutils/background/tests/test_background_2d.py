# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the background_2d module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.nddata import CCDData, NDData
from astropy.utils.exceptions import (AstropyDeprecationWarning,
                                      AstropyUserWarning)
from numpy.testing import assert_allclose, assert_equal

from photutils.background import (Background2D, BkgZoomInterpolator,
                                  MeanBackground, SExtractorBackground)
from photutils.utils._optional_deps import HAS_MATPLOTLIB


@pytest.fixture
def test_data():
    """
    Create test data for Background2D tests.
    """
    return np.ones((100, 100))


@pytest.fixture
def bkg_rms(test_data):
    """
    Expected background RMS for test data.
    """
    return np.zeros(test_data.shape)


@pytest.fixture
def bkg_mesh():
    """
    Expected background mesh for test data.
    """
    return np.ones((4, 4))


@pytest.fixture
def bkg_rms_mesh():
    """
    Expected background RMS mesh for test data.
    """
    return np.zeros((4, 4))


@pytest.fixture(params=['quantity', 'nddata_with_unit', 'ccddata'])
def nddata_variant(request, test_data):
    """
    Create different variants of input data with units for testing.
    """
    if request.param == 'quantity':
        return test_data << u.ct
    if request.param == 'nddata_with_unit':
        return NDData(test_data, unit=u.ct)
    return CCDData(test_data, unit=u.ct)


@pytest.fixture
def nddata_no_unit(test_data):
    """
    Create NDData without units for testing.
    """
    return NDData(test_data, unit=None)


class TestBackground2D:
    @pytest.mark.parametrize('filter_size', [(1, 1), (3, 3)])
    def test_background(self, filter_size, test_data, bkg_rms, bkg_mesh,
                        bkg_rms_mesh):
        bkg = Background2D(test_data, (25, 25), filter_size=filter_size)
        assert_allclose(bkg.background, test_data)
        assert_allclose(bkg.background_rms, bkg_rms)
        assert_allclose(bkg.background_mesh, bkg_mesh)
        assert_allclose(bkg.background_rms_mesh, bkg_rms_mesh)
        assert bkg.background_median == 1.0
        assert bkg.background_rms_median == 0.0
        assert bkg.npixels_mesh.shape == (4, 4)
        assert bkg.npixels_map.shape == test_data.shape

    @pytest.mark.parametrize('box_size', [(25, 25), (23, 22)])
    @pytest.mark.parametrize('dtype', ['int', 'int32', 'float32'])
    def test_background_dtype(self, box_size, dtype, test_data, bkg_rms):
        filter_size = 3
        data2 = test_data.copy().astype(dtype)
        bkg = Background2D(data2, box_size, filter_size=filter_size)
        assert bkg.background.dtype == data2.dtype
        assert bkg.background_rms.dtype == data2.dtype
        assert bkg.background_mesh.dtype == data2.dtype
        assert bkg.background_rms_mesh.dtype == data2.dtype
        assert bkg.npixels_map.dtype == int
        assert bkg.npixels_mesh.dtype == int
        assert_allclose(bkg.background, data2)
        assert_allclose(bkg.background_rms, bkg_rms)
        assert bkg.background_median == 1.0
        assert bkg.background_rms_median == 0.0
        assert bkg.npixels_map.shape == test_data.shape

    def test_background_nddata(self, test_data, bkg_rms, bkg_mesh,
                               bkg_rms_mesh, nddata_variant,
                               nddata_no_unit):
        """
        Test with NDData and CCDData, and also test units.
        """
        bkg = Background2D(nddata_variant, (25, 25), filter_size=3)
        assert isinstance(bkg.background, u.Quantity)
        assert isinstance(bkg.background_rms, u.Quantity)
        assert isinstance(bkg.background_median, u.Quantity)
        assert isinstance(bkg.background_rms_median, u.Quantity)

        bkg = Background2D(nddata_no_unit, (25, 25), filter_size=3)
        assert_allclose(bkg.background, test_data)
        assert_allclose(bkg.background_rms, bkg_rms)
        assert_allclose(bkg.background_mesh, bkg_mesh)
        assert_allclose(bkg.background_rms_mesh, bkg_rms_mesh)
        assert bkg.background_median == 1.0
        assert bkg.background_rms_median == 0.0

    def test_background_rect(self):
        """
        Regression test for interpolators with non-square input data.
        """
        data = np.arange(12).reshape(3, 4)
        rms = np.zeros((3, 4))
        bkg = Background2D(data, (1, 1), filter_size=1)
        assert_allclose(bkg.background, data, atol=0.005)
        assert_allclose(bkg.background_rms, rms)
        assert_allclose(bkg.background_mesh, data)
        assert_allclose(bkg.background_rms_mesh, rms)
        assert bkg.background_median == 5.5
        assert bkg.background_rms_median == 0.0

    def test_background_nonconstant(self, test_data, bkg_mesh):
        data = np.copy(test_data)
        data[25:50, 50:75] = 10.0
        bkg_low_res = np.copy(bkg_mesh)
        bkg_low_res[1, 2] = 10.0
        bkg1 = Background2D(data, (25, 25), filter_size=(1, 1))
        assert_allclose(bkg1.background_mesh, bkg_low_res)
        assert bkg1.background.shape == data.shape
        bkg2 = Background2D(data, (25, 25), filter_size=(1, 1))
        assert_allclose(bkg2.background_mesh, bkg_low_res)
        assert bkg2.background.shape == data.shape

        rng = np.random.default_rng(0)
        data = rng.normal(1.0, 0.1, (121, 289))
        mask = np.zeros(data.shape, dtype=bool)
        mask[50:100, 50:100] = True
        bkg = Background2D(data, (25, 25), mask=mask)
        assert np.mean(bkg.background) < 1.0
        assert np.mean(bkg.background_rms) < 1.0
        assert bkg.background_median < 1.0
        assert bkg.background_rms_median < 0.1
        assert bkg.npixels_mesh.shape == (5, 12)
        assert bkg.npixels_map.shape == data.shape

    def test_no_sigma_clipping(self, test_data):
        data = np.copy(test_data)
        data[10, 10] = 100.0
        bkg1 = Background2D(data, (25, 25), filter_size=(1, 1),
                            bkg_estimator=MeanBackground())
        bkg2 = Background2D(data, (25, 25), filter_size=(1, 1),
                            sigma_clip=None, bkg_estimator=MeanBackground())

        assert bkg2.background_mesh[0, 0] > bkg1.background_mesh[0, 0]

    def test_resizing(self):
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
    def test_background_mask(self, box_size, test_data, bkg_rms):
        """
        Test with an input mask.

        Note that box_size=(23, 22) tests the resizing of the image and
        mask.
        """
        data = np.copy(test_data)
        data[25:50, 25:50] = 100.0
        mask = np.zeros(test_data.shape, dtype=bool)
        mask[25:50, 25:50] = True
        bkg = Background2D(data, box_size, filter_size=(1, 1), mask=mask,
                           bkg_estimator=MeanBackground())
        assert_allclose(bkg.background, test_data, rtol=2.0e-5)
        assert_allclose(bkg.background_rms, bkg_rms)

    def test_mask(self, test_data):
        data = np.copy(test_data)
        data[25:50, 25:50] = 100.0
        mask = np.zeros(test_data.shape, dtype=bool)
        mask[25:50, 25:50] = True
        bkg1 = Background2D(data, (25, 25), filter_size=(1, 1), mask=None,
                            bkg_estimator=MeanBackground())
        assert np.all(bkg1.npixels_map == 625)
        assert np.all(bkg1.npixels_mesh == 625)
        assert bkg1.background.shape == data.shape
        assert_allclose(bkg1.background_mesh[0, 0], 1.0)
        assert_allclose(bkg1.background_mesh[1, 1], 100.0)
        assert np.all(bkg1.background_rms_mesh == 0.0)

        bkg2 = Background2D(data, (25, 25), filter_size=(1, 1), mask=mask,
                            bkg_estimator=MeanBackground())

        ngoodpix = test_data.size - 625
        assert np.count_nonzero(bkg2.npixels_map == 625) == ngoodpix
        assert np.count_nonzero(bkg2.npixels_mesh == 625) == 15
        assert bkg2.background.shape == data.shape
        assert_allclose(bkg2.background_mesh, 1.0)
        assert np.all(bkg2.background_rms_mesh == 0.0)

    @pytest.mark.parametrize('fill_value', [0.0, np.nan, -1.0])
    def test_coverage_mask(self, fill_value, test_data):
        data = np.copy(test_data)
        data[:50, :50] = np.nan
        mask = np.isnan(data)

        bkg1 = Background2D(data, (25, 25), filter_size=(1, 1),
                            coverage_mask=mask, fill_value=fill_value,
                            bkg_estimator=MeanBackground())
        assert_equal(bkg1.background[:50, :50], fill_value)
        assert_equal(bkg1.background_rms[:50, :50], fill_value)

        # test that combined mask and coverage_mask gives the same
        # results
        mask = np.zeros(test_data.shape, dtype=bool)
        coverage_mask = np.zeros(test_data.shape, dtype=bool)
        mask[:50, :25] = True
        coverage_mask[:50, 25:50] = True
        match = r'Input data contains non-finite \(NaN or infinity\) values'
        with pytest.warns(AstropyUserWarning, match=match):
            bkg2 = Background2D(data, (25, 25), filter_size=(1, 1), mask=mask,
                                coverage_mask=mask, fill_value=0.0,
                                bkg_estimator=MeanBackground())
        assert_allclose(bkg1.background_mesh, bkg2.background_mesh)
        assert_allclose(bkg1.background_rms_mesh, bkg2.background_rms_mesh)

    def test_mask_nonfinite(self, test_data):
        data = test_data.copy()
        data[0, 0:50] = np.nan
        match = r'Input data contains non-finite \(NaN or infinity\) values'
        with pytest.warns(AstropyUserWarning, match=match):
            bkg = Background2D(data, (25, 25), filter_size=(1, 1))
        assert_allclose(bkg.background, test_data, rtol=1e-5)

    def test_mask_with_already_masked_nans(self, test_data):
        """
        Test masked invalid values.

        These tests should not issue a warning.
        """
        data = test_data.copy()
        data[50, 25:50] = np.nan
        mask = np.isnan(data)

        bkg = Background2D(data, (25, 25), filter_size=(1, 1), mask=mask)
        assert_allclose(bkg.background, test_data, rtol=1e-5)

        bkg = Background2D(data, (25, 25), filter_size=(1, 1),
                           coverage_mask=mask)
        assert bkg.background.shape == data.shape

        mask = np.zeros(data.shape, dtype=bool)
        coverage_mask = np.zeros(data.shape, dtype=bool)
        mask[50, 25:30] = True
        coverage_mask[50, 30:50] = True
        bkg = Background2D(data, (25, 25), filter_size=(1, 1), mask=mask,
                           coverage_mask=coverage_mask)
        assert bkg.background.shape == data.shape

    def test_masked_array(self, test_data):
        data = test_data.copy()
        data[0, 0:50] = True
        mask = np.zeros(test_data.shape, dtype=bool)
        mask[0, 0:50] = True
        data_ma1 = np.ma.MaskedArray(test_data, mask=mask)
        data_ma2 = np.ma.MaskedArray(data, mask=mask)

        bkg1 = Background2D(data, (25, 25), filter_size=(1, 1))
        bkg2 = Background2D(data_ma1, (25, 25), filter_size=(1, 1))
        bkg3 = Background2D(data_ma2, (25, 25), filter_size=(1, 1))
        assert_allclose(bkg1.background, bkg2.background, rtol=1e-5)
        assert_allclose(bkg2.background, bkg3.background, rtol=1e-5)

    def test_completely_masked(self, test_data):
        mask = np.ones(test_data.shape, dtype=bool)
        match = 'All input pixels are masked. Cannot compute a background.'
        with pytest.raises(ValueError, match=match):
            Background2D(test_data, (25, 25), mask=mask)
        with pytest.raises(ValueError, match=match):
            Background2D(test_data, (25, 25), coverage_mask=mask)

        mask = np.zeros(test_data.shape, dtype=bool)
        coverage_mask = np.zeros(test_data.shape, dtype=bool)
        mask[:, 0:40] = True
        coverage_mask[:, 40:] = True
        with pytest.raises(ValueError, match=match):
            Background2D(test_data, (25, 25), mask=mask,
                         coverage_mask=coverage_mask)

        data = test_data.copy()
        data[:] = np.nan
        match = r'Input data contains all non-finite \(NaN or infinity\)'
        with pytest.raises(ValueError, match=match):
            Background2D(data, (25, 25))

    def test_zero_padding(self, test_data, bkg_rms):
        """
        Test case where padding is added only on one axis.
        """
        bkg = Background2D(test_data, (25, 22), filter_size=(1, 1))
        assert_allclose(bkg.background, test_data, rtol=1e-5)
        assert_allclose(bkg.background_rms, bkg_rms)
        assert bkg.background_median == 1.0
        assert bkg.background_rms_median == 0.0

        bkg = Background2D(test_data, (22, 25), filter_size=(1, 1))
        assert_allclose(bkg.background, test_data, rtol=1e-5)
        assert_allclose(bkg.background_rms, bkg_rms)
        assert bkg.background_median == 1.0
        assert bkg.background_rms_median == 0.0

    def test_exclude_percentile(self, test_data):
        """
        Only meshes greater than filter_threshold are filtered.
        """
        data = np.copy(test_data)
        data[0:50, 0:50] = np.nan
        match = r'Input data contains non-finite \(NaN or infinity\) values'
        with pytest.warns(AstropyUserWarning, match=match):
            bkg = Background2D(data, (25, 25), filter_size=(1, 1),
                               exclude_percentile=100.0)
        assert_equal(bkg.npixels_mesh[0:2, 0:2], np.zeros((2, 2)))
        assert bkg.npixels_mesh[-1, -1] == 625

        data = np.ones((111, 121))
        bkg = Background2D(data, box_size=10, exclude_percentile=100)
        assert_allclose(bkg.background_mesh, np.ones((12, 13)))

        data[:] = np.nan
        data[0, 0] = 1.0
        match1 = r'Input data contains non-finite \(NaN or infinity\) values'
        match2 = r'All boxes contain .* unmasked or finite pixels'
        ctx1 = pytest.warns(AstropyUserWarning, match=match1)
        ctx2 = pytest.raises(ValueError, match=match2)
        with ctx1, ctx2:
            Background2D(data, (10, 10))

    def test_filter_threshold(self, test_data, bkg_mesh):
        """
        Only meshes greater than filter_threshold are filtered.
        """
        data = np.copy(test_data)
        data[25:50, 50:75] = 10.0
        bkg = Background2D(data, (25, 25), filter_size=(3, 3),
                           filter_threshold=9.0)
        assert_allclose(bkg.background, test_data)
        assert_allclose(bkg.background_mesh, bkg_mesh)
        bkg2 = Background2D(data, (25, 25), filter_size=(3, 3),
                            filter_threshold=11.0)  # no filtering
        assert bkg2.background_mesh[1, 2] == 10

    def test_filter_threshold_high(self, test_data, bkg_mesh):
        """
        No filtering because filter_threshold is too large.
        """
        data = np.copy(test_data)
        data[25:50, 50:75] = 10.0
        ref_data = np.copy(bkg_mesh)
        ref_data[1, 2] = 10.0
        bkg = Background2D(data, (25, 25), filter_size=(3, 3),
                           filter_threshold=100.0)
        assert_allclose(bkg.background_mesh, ref_data)

    def test_filter_threshold_nofilter(self, test_data, bkg_mesh):
        """
        No filtering because filter_size is (1, 1).
        """
        data = np.copy(test_data)
        data[25:50, 50:75] = 10.0
        ref_data = np.copy(bkg_mesh)
        ref_data[1, 2] = 10.0
        b = Background2D(data, (25, 25), filter_size=(1, 1),
                         filter_threshold=1.0)
        assert_allclose(b.background_mesh, ref_data)

    def test_scalar_sizes(self, test_data):
        bkg1 = Background2D(test_data, (25, 25), filter_size=(3, 3))
        bkg2 = Background2D(test_data, 25, filter_size=3)
        assert_allclose(bkg1.background, bkg2.background)
        assert_allclose(bkg1.background_rms, bkg2.background_rms)

    def test_invalid_box_size(self, test_data):
        match = 'box_size must have 1 or 2 elements'
        with pytest.raises(ValueError, match=match):
            Background2D(test_data, (5, 5, 3))

    def test_invalid_filter_size(self, test_data):
        match = 'filter_size must have 1 or 2 elements'
        with pytest.raises(ValueError, match=match):
            Background2D(test_data, (5, 5), filter_size=(3, 3, 3))

    def test_invalid_exclude_percentile(self, test_data):
        match = 'exclude_percentile must be between 0 and 100'
        with pytest.raises(ValueError, match=match):
            Background2D(test_data, (5, 5), exclude_percentile=-1)
        with pytest.raises(ValueError, match=match):
            Background2D(test_data, (5, 5), exclude_percentile=101)

    def test_mask_nomask(self, test_data):
        bkg = Background2D(test_data, (25, 25), filter_size=(1, 1),
                           mask=np.ma.nomask)
        assert bkg._mask is None

        bkg = Background2D(test_data, (25, 25), filter_size=(1, 1),
                           coverage_mask=np.ma.nomask)
        assert bkg.coverage_mask is None

    def test_invalid_mask(self, test_data):
        match = 'data and mask must have the same shape'
        with pytest.raises(ValueError, match=match):
            Background2D(test_data, (25, 25), filter_size=(1, 1),
                         mask=np.zeros((2, 2)))

        match = 'mask must be a 2D array'
        with pytest.raises(ValueError, match=match):
            Background2D(test_data, (25, 25), filter_size=(1, 1),
                         mask=np.zeros((2, 2, 2)))

    def test_invalid_coverage_mask(self, test_data):
        match = 'data and coverage_mask must have the same shape'
        with pytest.raises(ValueError, match=match):
            Background2D(test_data, (25, 25), filter_size=(1, 1),
                         coverage_mask=np.zeros((2, 2)))

        match = 'coverage_mask must be a 2D array'
        with pytest.raises(ValueError, match=match):
            Background2D(test_data, (25, 25), filter_size=(1, 1),
                         coverage_mask=np.zeros((2, 2, 2)))

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot_meshes(self, test_data):
        """
        This test should run without any errors, but there is no return
        value.
        """
        bkg = Background2D(test_data, (25, 25))
        bkg.plot_meshes(outlines=True)

    def test_repr(self):
        data = np.ones((300, 500))
        bkg = Background2D(data, (74, 99))
        cls_repr = repr(bkg)
        assert cls_repr.startswith(f'{bkg.__class__.__name__}')

        mask = np.zeros(data.shape, dtype=bool)
        mask[0:10, 0:10] = True
        bkg = Background2D(data, (74, 99), mask=mask)
        cls_repr = repr(bkg)
        assert cls_repr.startswith(f'{bkg.__class__.__name__}')
        assert 'mask' in cls_repr

        bkg = Background2D(data, (74, 99), coverage_mask=mask)
        cls_repr = repr(bkg)
        assert cls_repr.startswith(f'{bkg.__class__.__name__}')
        assert 'coverage_mask' in cls_repr

    def test_str(self):
        data = np.ones((300, 500))
        bkg = Background2D(data, (74, 99))
        cls_str = str(bkg)
        cls_name = bkg.__class__.__name__
        cls_name = f'{bkg.__class__.__module__}.{cls_name}'
        assert cls_str.startswith(f'<{cls_name}>')

    def test_masks(self):
        arr = np.arange(25.0).reshape(5, 5)
        arr_orig = arr.copy()
        mask = np.zeros(arr.shape, dtype=bool)
        mask[0, 0] = np.nan
        mask[-1, 0] = np.nan
        mask[-1, -1] = np.nan
        mask[0, -1] = np.nan

        box_size = (2, 2)
        exclude_percentile = 100
        filter_size = 1
        bkg_estimator = MeanBackground()
        bkg1 = Background2D(arr, box_size, mask=mask,
                            exclude_percentile=exclude_percentile,
                            filter_size=filter_size,
                            bkg_estimator=bkg_estimator)
        bkgimg1 = bkg1.background
        assert_equal(arr, arr_orig)

        arr2 = arr.copy()
        arr2[mask] = np.nan
        arr3 = arr2.copy()
        match = r'Input data contains non-finite \(NaN or infinity\) values'
        with pytest.warns(AstropyUserWarning, match=match):
            bkg2 = Background2D(arr2, box_size, mask=None,
                                exclude_percentile=exclude_percentile,
                                filter_size=filter_size,
                                bkg_estimator=bkg_estimator)
        bkgimg2 = bkg2.background
        assert_equal(arr2, arr3)

        assert_allclose(bkgimg1, bkgimg2)

    @pytest.mark.parametrize('bkg_est', [MeanBackground(),
                                         SExtractorBackground()])
    def test_large_boxsize(self, bkg_est):
        """
        Regression test to ensure that when boxsize is the same as the
        image size that the input data left unchanged.
        """
        shape = (103, 107)
        data = np.ones(shape)
        data[50:55, 50:55] = 1000.0
        data[20:25, 20:25] = 1000.0
        box_size = data.shape
        filter_size = (3, 3)
        data_orig = data.copy()
        bkg = Background2D(data, box_size, filter_size=filter_size,
                           bkg_estimator=bkg_est)
        bkgim = bkg.background
        assert bkgim.shape == shape
        assert_equal(data, data_orig)

    def test_interpolator_keyword_deprecation(self, test_data):
        """
        Test that the interpolator keyword is deprecated.
        """
        match = 'BkgZoomInterpolator is deprecated'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            interp = BkgZoomInterpolator()

        match = '"interpolator" was deprecated'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            bkg = Background2D(test_data, (25, 25), interpolator=interp)

        assert_allclose(bkg.background, test_data)

        bkg = Background2D(test_data, (25, 25))  # Should not raise
        assert_allclose(bkg.background, test_data)

    def test_background_box_size_one(self, test_data):
        """
        Test with box_size=1 (no smoothing).
        """
        bkg = Background2D(test_data, (1, 1), filter_size=(1, 1))
        # Should be close to input data
        assert_allclose(bkg.background, test_data, rtol=1e-5)

    def test_background_prime_dimensions(self):
        """
        Test with prime-number dimensions.
        """
        data = np.ones((97, 101))  # Prime dimensions
        bkg = Background2D(data, (10, 10))
        assert bkg.background.shape == data.shape
        assert_allclose(bkg.background, data, rtol=1e-5)

    def test_background_box_size_larger_than_image(self):
        """
        Test when box_size exceeds image dimensions.
        """
        data = np.ones((50, 60))
        bkg = Background2D(data, (100, 100))
        assert bkg.background.shape == data.shape
        # With box size larger than image, should get single mesh value
        assert bkg.background_mesh.shape == (1, 1)

        # Test with one box dimension larger than image
        bkg = Background2D(data, (100, 30))
        assert bkg.background.shape == data.shape
        assert bkg.background_mesh.shape == (1, 2)

        bkg = Background2D(data, (25, 100))
        assert bkg.background.shape == data.shape
        assert bkg.background_mesh.shape == (2, 1)

    def test_background_properties_relationship(self, test_data):
        """
        Test that properties are computed correctly and maintain
        relationships.
        """
        bkg = Background2D(test_data, (25, 25))

        # Check relationships between properties
        assert bkg.background_mesh.shape[0] * 25 >= test_data.shape[0]
        assert bkg.background_mesh.shape[1] * 25 >= test_data.shape[1]
        assert_allclose(bkg.background_median,
                        np.median(bkg.background_mesh))
        assert_allclose(bkg.background_rms_median,
                        np.median(bkg.background_rms_mesh))
