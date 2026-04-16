# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the catalog module.
"""

from io import StringIO
from unittest.mock import patch

import astropy.units as u
import numpy as np
import pytest
from astropy.convolution import convolve
from astropy.coordinates import SkyCoord
from astropy.modeling.models import Gaussian2D
from astropy.table import QTable
from astropy.utils.exceptions import AstropyDeprecationWarning
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import root_scalar

from photutils.aperture import (BoundingBox, CircularAperture,
                                EllipticalAperture)
from photutils.background import Background2D, MedianBackground
from photutils.datasets import (make_100gaussians_image, make_gwcs,
                                make_noise_image, make_wcs)
from photutils.segmentation.catalog import SourceCatalog
from photutils.segmentation.core import SegmentationImage
from photutils.segmentation.detect import detect_sources
from photutils.segmentation.finder import SourceFinder
from photutils.segmentation.utils import make_2dgaussian_kernel
from photutils.utils._optional_deps import (HAS_GWCS, HAS_MATPLOTLIB,
                                            HAS_SKIMAGE)
from photutils.utils.cutouts import CutoutImage


@pytest.fixture
def progress_bar_catalog():
    """
    A two-source SourceCatalog on a 101x101 grid with progress_bar=True.
    """
    yy, xx = np.mgrid[0:101, 0:101]
    g1 = Gaussian2D(100, 50, 50, 5, 5)
    g2 = Gaussian2D(80, 30, 30, 4, 4)
    data = g1(xx, yy) + g2(xx, yy)
    segm = detect_sources(data, 10.0, n_pixels=5)
    return SourceCatalog(data, segm, progress_bar=True)


@pytest.fixture
def single_source_catalog():
    """
    A single-source SourceCatalog from a Gaussian on a 51x51 grid.

    Returns ``(data, segm, cat)``.
    """
    yy, xx = np.mgrid[0:51, 0:51]
    g1 = Gaussian2D(100, 25, 25, 5, 5)
    data = g1(xx, yy)
    segm = detect_sources(data, 10.0, n_pixels=5)
    cat = SourceCatalog(data, segm)
    return data, segm, cat


@pytest.fixture
def gauss_101_data():
    """
    A single-source Gaussian on a 101x101 grid.

    Returns ``(data, segm)``.
    """
    yy, xx = np.mgrid[0:101, 0:101]
    g1 = Gaussian2D(100, 50, 50, 5, 5)
    data = g1(xx, yy)
    segm = detect_sources(data, 10.0, n_pixels=5)
    return data, segm


@pytest.fixture
def gauss_101_catalog(gauss_101_data):
    """
    A single-source SourceCatalog from a Gaussian on a 101x101 grid.

    Returns ``(data, segm, cat)``.
    """
    data, segm = gauss_101_data
    cat = SourceCatalog(data, segm)
    return data, segm, cat


@pytest.fixture
def centroid_win_data():
    """
    Two-source data on a 21x21 grid for centroid_win tests.

    Returns ``(data, segment_map, convolved_data)``.
    """
    g1 = Gaussian2D(1621, 6.29, 10.95, 1.55, 1.29, 0.296706)
    g2 = Gaussian2D(3596, 13.81, 8.29, 1.44, 1.27, 0.628319)
    m = g1 + g2
    yy, xx = np.mgrid[0:21, 0:21]
    data = m(xx, yy)
    noise = make_noise_image(data.shape, mean=0, stddev=65.0, seed=123)
    data += noise

    kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data = convolve(data, kernel)
    n_pixels = 10
    finder = SourceFinder(n_pixels=n_pixels, progress_bar=False)
    threshold = 107.9
    segment_map = finder(convolved_data, threshold)
    return data, segment_map, convolved_data


class TestSourceCatalog:
    @pytest.fixture(autouse=True)
    def setup(self):
        xcen = 51.0
        ycen = 52.7
        major_sigma = 8.0
        minor_sigma = 3.0
        theta = np.pi / 6.0
        g1 = Gaussian2D(111.0, xcen, ycen, major_sigma, minor_sigma,
                        theta=theta)
        g2 = Gaussian2D(50, 20, 80, 5.1, 4.5)
        g3 = Gaussian2D(70, 75, 18, 9.2, 4.5)
        g4 = Gaussian2D(111.0, 11.1, 12.2, major_sigma, minor_sigma,
                        theta=theta)
        g5 = Gaussian2D(81.0, 61, 42.7, major_sigma, minor_sigma, theta=theta)
        g6 = Gaussian2D(107.0, 75, 61, major_sigma, minor_sigma, theta=-theta)
        g7 = Gaussian2D(107.0, 90, 90, 4, 2, theta=-theta)

        yy, xx = np.mgrid[0:101, 0:101]
        self.data = (g1(xx, yy) + g2(xx, yy) + g3(xx, yy) + g4(xx, yy)
                     + g5(xx, yy) + g6(xx, yy) + g7(xx, yy))
        threshold = 27.0
        self.segm = detect_sources(self.data, threshold, n_pixels=5)
        self.error = make_noise_image(self.data.shape, mean=0, stddev=2.0,
                                      seed=123)
        self.background = np.ones(self.data.shape) * 5.1
        self.mask = np.zeros(self.data.shape, dtype=bool)
        self.mask[0:30, 0:30] = True

        self.wcs = make_wcs(self.data.shape)
        self.cat = SourceCatalog(self.data, self.segm, error=self.error,
                                 background=self.background, mask=self.mask,
                                 wcs=self.wcs, local_bkg_width=24)
        unit = u.nJy
        self.unit = unit
        self.cat_units = SourceCatalog(self.data << unit, self.segm,
                                       error=self.error << unit,
                                       background=self.background << unit,
                                       mask=self.mask, wcs=self.wcs,
                                       local_bkg_width=24)

    @pytest.mark.parametrize('with_units', [True, False])
    def test_catalog(self, with_units):
        """
        Test catalog.
        """
        if with_units:
            cat1 = self.cat_units.copy()
            cat2 = self.cat_units.copy()
        else:
            cat1 = self.cat.copy()
            cat2 = self.cat.copy()

        props = self.cat.properties

        # Add extra properties
        cat1.circular_photometry(5.0, name='circ5')
        cat1.kron_photometry((2.5, 1.4), name='kron2')
        cat1.flux_radius(0.5, name='r_hl')
        segment_snr = cat1.segment_flux / cat1.segment_flux_err
        cat1.add_property('segment_snr', segment_snr)
        props = list(props)
        props.extend(cat1.custom_properties)

        idx = 1  # no NaN values

        # Evaluate (cache) catalog properties before slice
        obj = cat1[idx]
        for prop in props:
            assert_equal(getattr(cat1, prop)[idx], getattr(obj, prop))

        # Slice catalog before evaluating catalog properties
        obj = cat2[idx]
        obj.circular_photometry(5.0, name='circ5')
        obj.kron_photometry((2.5, 1.4), name='kron2')
        obj.flux_radius(0.5, name='r_hl')
        segment_snr = obj.segment_flux / obj.segment_flux_err
        obj.add_property('segment_snr', segment_snr)
        for prop in props:
            assert_equal(getattr(obj, prop), getattr(cat1, prop)[idx])

        match = 'Both units and masked cannot be True'
        with pytest.raises(ValueError, match=match):
            cat1._prepare_cutouts(cat1._segmentation_image_cutouts, units=True,
                                  masked=True)

    @pytest.mark.parametrize('with_units', [True, False])
    def test_catalog_detection_catalog(self, with_units):
        """
        Test aperture-based properties with an input detection catalog.
        """
        error = 2.0 * self.error
        data2 = self.data + error

        if with_units:
            cat1 = self.cat_units.copy()
            cat2 = SourceCatalog(data2 << self.unit, self.segm,
                                 error=error << self.unit,
                                 background=self.background << self.unit,
                                 mask=self.mask, wcs=self.wcs,
                                 local_bkg_width=24,
                                 detection_catalog=None)
            cat3 = SourceCatalog(data2 << self.unit, self.segm,
                                 error=error << self.unit,
                                 background=self.background << self.unit,
                                 mask=self.mask, wcs=self.wcs,
                                 local_bkg_width=24,
                                 detection_catalog=cat1)
        else:
            cat1 = self.cat.copy()
            cat2 = SourceCatalog(data2, self.segm, error=error,
                                 background=self.background, mask=self.mask,
                                 wcs=self.wcs, local_bkg_width=24,
                                 detection_catalog=None)
            cat3 = SourceCatalog(data2, self.segm, error=error,
                                 background=self.background, mask=self.mask,
                                 wcs=self.wcs, local_bkg_width=24,
                                 detection_catalog=cat1)

        assert_equal(cat1.kron_radius, cat3.kron_radius)
        # Assert not equal
        match = 'Arrays are not equal'
        with pytest.raises(AssertionError, match=match):
            assert_equal(cat1.kron_radius, cat2.kron_radius)
        with pytest.raises(AssertionError, match=match):
            assert_equal(cat2.kron_flux, cat3.kron_flux)
        with pytest.raises(AssertionError, match=match):
            assert_equal(cat2.kron_flux_err, cat3.kron_flux_err)
        with pytest.raises(AssertionError, match=match):
            assert_equal(cat1.kron_flux, cat3.kron_flux)
        with pytest.raises(AssertionError, match=match):
            assert_equal(cat1.kron_flux_err, cat3.kron_flux_err)

        flux1, flux_err1 = cat1.circular_photometry(1.0)
        flux2, flux_err2 = cat2.circular_photometry(1.0)
        flux3, flux_err3 = cat3.circular_photometry(1.0)
        with pytest.raises(AssertionError, match=match):
            assert_equal(flux2, flux3)
        with pytest.raises(AssertionError, match=match):
            assert_equal(flux_err2, flux_err3)
        with pytest.raises(AssertionError, match=match):
            assert_equal(flux1, flux2)
        with pytest.raises(AssertionError, match=match):
            assert_equal(flux_err1, flux_err2)

        flux1, flux_err1 = cat1.kron_photometry((2.5, 1.4))
        flux2, flux_err2 = cat2.kron_photometry((2.5, 1.4))
        flux3, flux_err3 = cat3.kron_photometry((2.5, 1.4))
        with pytest.raises(AssertionError, match=match):
            assert_equal(flux2, flux3)
        with pytest.raises(AssertionError, match=match):
            assert_equal(flux_err2, flux_err3)
        with pytest.raises(AssertionError, match=match):
            assert_equal(flux1, flux2)
        with pytest.raises(AssertionError, match=match):
            assert_equal(flux_err1, flux_err2)

        radius1 = cat1.flux_radius(0.5)
        radius2 = cat2.flux_radius(0.5)
        radius3 = cat3.flux_radius(0.5)
        with pytest.raises(AssertionError, match=match):
            assert_equal(radius2, radius3)
        with pytest.raises(AssertionError, match=match):
            assert_equal(radius1, radius2)

        cat4 = cat3[0:1]
        assert len(cat4.kron_radius) == 1

    def test_minimal_catalog(self):
        """
        Test minimal catalog.
        """
        cat = SourceCatalog(self.data, self.segm)
        obj = cat[4]
        props = ('background_cutout', 'background_cutout_masked',
                 'error_cutout', 'error_cutout_masked')
        for prop in props:
            assert getattr(obj, prop) is None

        arr_props = ('_background_cutouts', '_error_cutouts')
        for prop in arr_props:
            assert getattr(obj, prop)[0] is None

        props = ('background_mean', 'background_sum', 'background_centroid',
                 'segment_flux_err', 'kron_flux_err')
        for prop in props:
            assert np.isnan(getattr(obj, prop))

        assert obj.local_background_aperture is None
        assert obj.local_background == 0.0

    def test_slicing(self):
        """
        Test slicing.
        """
        self.cat.to_table()  # evaluate and cache several properties

        obj1 = self.cat[0]
        assert obj1.n_labels == 1
        obj1b = self.cat.select_label(1)
        assert obj1b.n_labels == 1

        obj2 = self.cat[0:1]
        assert obj2.n_labels == 1
        assert len(obj2) == 1

        obj3 = self.cat[0:3]
        obj3b = self.cat.select_labels((1, 2, 3))
        assert_equal(obj3.label, obj3b.label)
        obj4 = self.cat[[0, 1, 2]]
        assert obj3.n_labels == 3
        assert obj3b.n_labels == 3
        assert obj4.n_labels == 3
        assert len(obj3) == 3
        assert len(obj4) == 3

        obj5 = self.cat[[3, 2, 1]]
        labels = [4, 3, 2]
        obj5b = self.cat.select_labels(labels)
        assert_equal(obj5.label, obj5b.label)
        assert obj5.n_labels == 3
        assert len(obj5) == 3
        assert_equal(obj5.label, labels)

        # Test select_labels when labels are not sorted
        obj5 = self.cat[[3, 2, 1]]
        labels2 = (3, 4)
        obj5b = obj5.select_labels(labels2)
        assert_equal(obj5b.label, labels2)

        obj6 = obj5[0]
        assert obj6.label == labels[0]

        mask = self.cat.label > 3
        obj7 = self.cat[mask]
        assert obj7.n_labels == 4
        assert len(obj7) == 4

        obj1 = self.cat[0]
        match = "A scalar 'SourceCatalog' object cannot be indexed"
        with pytest.raises(TypeError, match=match):
            obj2 = obj1[0]

        match = 'is invalid'
        with pytest.raises(ValueError, match=match):
            self.cat.select_label(1000)
        with pytest.raises(ValueError, match=match):
            self.cat.select_labels([1, 2, 1000])

    def test_iter(self):
        """
        Test iter.
        """
        labels = [obj.label for obj in self.cat]
        assert len(labels) == len(self.cat)

    def test_table(self):
        """
        Test table.
        """
        columns = ['label', 'x_centroid', 'y_centroid']
        tbl = self.cat.to_table(columns=columns)
        assert len(tbl) == 7
        assert tbl.colnames == columns

        tbl = self.cat.to_table(columns=self.cat.default_columns)
        for col in tbl.columns:
            assert isinstance(col, str)
            assert not isinstance(col, np.str_)

        tbl = self.cat.to_table(columns='label')
        for col in tbl.columns:
            assert isinstance(col, str)
            assert not isinstance(col, np.str_)

    def test_invalid_inputs(self):
        """
        Test invalid inputs.
        """
        segm = SegmentationImage(np.zeros(self.data.shape, dtype=int))
        match = 'segmentation_image must have at least one non-zero label'
        with pytest.raises(ValueError, match=match):
            SourceCatalog(self.data, segm)

        # Test 1D arrays
        img1d = np.arange(4)
        segm = SegmentationImage(img1d)
        match = 'data must be a 2D array'
        with pytest.raises(ValueError, match=match):
            SourceCatalog(img1d, segm)

        wrong_shape = np.ones((3, 3), dtype=int)
        match = 'segmentation_image and data must have the same shape'
        with pytest.raises(ValueError, match=match):
            SourceCatalog(wrong_shape, self.segm)

        match = 'data and error must have the same shape'
        with pytest.raises(ValueError, match=match):
            SourceCatalog(self.data, self.segm, error=wrong_shape)

        match = 'data and background must have the same shape'
        with pytest.raises(ValueError, match=match):
            SourceCatalog(self.data, self.segm, background=wrong_shape)

        match = 'data and mask must have the same shape'
        with pytest.raises(ValueError, match=match):
            SourceCatalog(self.data, self.segm, mask=wrong_shape)

        segm = SegmentationImage(wrong_shape)
        match = 'segmentation_image and data must have the same shape'
        with pytest.raises(ValueError, match=match):
            SourceCatalog(self.data, segm)

        match = 'segmentation_image must be a SegmentationImage'
        with pytest.raises(TypeError, match=match):
            SourceCatalog(self.data, wrong_shape)

        obj = SourceCatalog(self.data, self.segm)[0]
        match = "Scalar 'SourceCatalog' object has no len()"
        with pytest.raises(TypeError, match=match):
            len(obj)

        match = 'local_bkg_width must be >= 0'
        with pytest.raises(ValueError, match=match):
            SourceCatalog(self.data, self.segm, local_bkg_width=-1)
        match = 'local_bkg_width must be an integer'
        with pytest.raises(ValueError, match=match):
            SourceCatalog(self.data, self.segm, local_bkg_width=3.4)

        aperture_mask_method = 'invalid'
        match = 'Invalid aperture_mask_method value'
        with pytest.raises(ValueError, match=match):
            SourceCatalog(self.data, self.segm,
                          aperture_mask_method=aperture_mask_method)

        kron_params = (0.0, 1.0)
        match = r'kron_params\[0\] must be > 0'
        with pytest.raises(ValueError, match=match):
            SourceCatalog(self.data, self.segm, kron_params=kron_params)
        kron_params = (-2.5, 0.0)
        with pytest.raises(ValueError, match=match):
            SourceCatalog(self.data, self.segm, kron_params=kron_params)

        kron_params = (2.5, 0.0)
        match = r'kron_params\[1\] must be > 0'
        with pytest.raises(ValueError, match=match):
            SourceCatalog(self.data, self.segm, kron_params=kron_params)
        kron_params = (2.5, -4.0)
        with pytest.raises(ValueError, match=match):
            SourceCatalog(self.data, self.segm, kron_params=kron_params)

        kron_params = (2.5, 1.4, -2.0)
        match = r'kron_params\[2\] must be >= 0'
        with pytest.raises(ValueError, match=match):
            SourceCatalog(self.data, self.segm, kron_params=kron_params)

    def test_invalid_units(self):
        """
        Test invalid units.
        """
        unit = u.uJy
        wrong_unit = u.km

        match = 'must all have the same units'
        with pytest.raises(ValueError, match=match):
            SourceCatalog(self.data << unit, self.segm,
                          error=self.error << wrong_unit)

        with pytest.raises(ValueError, match=match):
            SourceCatalog(self.data << unit, self.segm,
                          background=self.background << wrong_unit)

        # All array inputs must have the same unit
        with pytest.raises(ValueError, match=match):
            SourceCatalog(self.data << unit, self.segm, error=self.error)

        with pytest.raises(ValueError, match=match):
            SourceCatalog(self.data, self.segm,
                          background=self.background << unit)

    def test_wcs(self):
        """
        Test wcs.
        """
        mywcs = make_wcs(self.data.shape)
        cat = SourceCatalog(self.data, self.segm, wcs=mywcs)
        obj = cat[0]
        assert obj.sky_centroid is not None
        assert obj.sky_centroid_icrs is not None
        assert obj.sky_centroid_win is not None
        assert obj.sky_bbox_ll is not None
        assert obj.sky_bbox_ul is not None
        assert obj.sky_bbox_lr is not None
        assert obj.sky_bbox_ur is not None

    @pytest.mark.skipif(not HAS_GWCS, reason='gwcs is required')
    def test_gwcs(self):
        """
        Test gwcs.
        """
        mywcs = make_gwcs(self.data.shape)
        cat = SourceCatalog(self.data, self.segm, wcs=mywcs)
        obj = cat[1]
        assert obj.sky_centroid is not None
        assert obj.sky_centroid_icrs is not None
        assert obj.sky_centroid_win is not None
        assert obj.sky_bbox_ll is not None
        assert obj.sky_bbox_ul is not None
        assert obj.sky_bbox_lr is not None
        assert obj.sky_bbox_ur is not None

    def test_nowcs(self):
        """
        Test nowcs.
        """
        cat = SourceCatalog(self.data, self.segm, wcs=None)
        obj = cat[2]
        assert obj.sky_centroid is None
        assert obj.sky_centroid_icrs is None
        assert obj.sky_centroid_win is None
        assert obj.sky_bbox_ll is None
        assert obj.sky_bbox_ul is None
        assert obj.sky_bbox_lr is None
        assert obj.sky_bbox_ur is None

    def test_to_table(self):
        """
        Test to table.
        """
        cat = SourceCatalog(self.data, self.segm)
        assert len(cat) == 7
        tbl = cat.to_table()
        assert isinstance(tbl, QTable)
        assert len(tbl) == 7
        obj = cat[0]
        assert obj.n_labels == 1
        tbl = obj.to_table()
        assert len(tbl) == 1

    def test_masks(self):
        """
        Test masks, including automatic masking of all non-finite (e.g.,
        NaN, inf) values in the data array.
        """
        data = np.copy(self.data)
        error = np.copy(self.error)
        background = np.copy(self.background)
        data[:, 55] = np.nan
        data[16, :] = np.inf
        error[:, 55] = np.nan
        error[16, :] = np.inf
        background[:, 55] = np.nan
        background[16, :] = np.inf

        cat = SourceCatalog(data, self.segm, error=error,
                            background=background, mask=self.mask)

        props = ('x_centroid', 'y_centroid', 'area', 'orientation',
                 'segment_flux', 'segment_flux_err', 'kron_flux',
                 'kron_flux_err', 'background_mean')
        obj = cat[0]
        for prop in props:
            assert np.isnan(getattr(obj, prop))
        objs = cat[1:]
        for prop in props:
            assert np.all(np.isfinite(getattr(objs, prop)))

        # Test that mask=None is the same as mask=np.ma.nomask
        cat1 = SourceCatalog(data, self.segm, mask=None)
        cat2 = SourceCatalog(data, self.segm, mask=np.ma.nomask)
        assert cat1[0].x_centroid == cat2[0].x_centroid

    def test_repr_str(self):
        """
        Test repr str.
        """
        cat = SourceCatalog(self.data, self.segm)
        assert repr(cat) == str(cat)

        lines = ('Length: 7', 'labels: [1 2 3 4 5 6 7]')
        for line in lines:
            assert line in repr(cat)

    def test_detection_catalog(self):
        """
        Test detection catalog.
        """
        data2 = self.data - 5
        cat1 = SourceCatalog(data2, self.segm)
        cat2 = SourceCatalog(data2, self.segm, detection_catalog=self.cat)
        assert len(cat2.kron_aperture) == len(cat2)
        assert not np.array_equal(cat1.kron_radius, cat2.kron_radius)
        assert not np.array_equal(cat1.kron_flux, cat2.kron_flux)
        assert_allclose(cat2.kron_radius, self.cat.kron_radius)
        assert not np.array_equal(cat2.kron_flux, self.cat.kron_flux)

        with pytest.raises(TypeError):
            SourceCatalog(data2, self.segm, detection_catalog=np.arange(4))

        segm = self.segm.copy()
        segm.remove_labels((6, 7))
        cat = SourceCatalog(self.data, segm)

        match = ('detection_catalog must have same'
                 ' segmentation_image as the input')
        with pytest.raises(ValueError, match=match):
            SourceCatalog(self.data, self.segm, detection_catalog=cat)

    def test_kron_minradius(self):
        """
        Test kron minradius.
        """
        kron_params = (2.5, 2.5)
        cat = SourceCatalog(self.data, self.segm, mask=self.mask,
                            aperture_mask_method='none',
                            kron_params=kron_params)
        assert cat.kron_aperture[0] is None
        assert np.isnan(cat.kron_radius[0])
        kronrad = cat.kron_radius.value
        kronrad = kronrad[~np.isnan(kronrad)]
        assert np.min(kronrad) == kron_params[1]
        assert isinstance(cat.kron_aperture[2], EllipticalAperture)
        assert isinstance(cat.kron_aperture[4], EllipticalAperture)
        assert isinstance(cat.kron_params, tuple)

    def test_kron_masking(self):
        """
        Test kron masking.
        """
        aperture_mask_method = 'none'
        cat1 = SourceCatalog(self.data, self.segm,
                             aperture_mask_method=aperture_mask_method)
        aperture_mask_method = 'mask'
        cat2 = SourceCatalog(self.data, self.segm,
                             aperture_mask_method=aperture_mask_method)
        aperture_mask_method = 'correct'
        cat3 = SourceCatalog(self.data, self.segm,
                             aperture_mask_method=aperture_mask_method)
        idx = 2  # source with close neighbors
        assert cat1[idx].kron_flux > cat2[idx].kron_flux
        assert cat3[idx].kron_flux > cat2[idx].kron_flux
        assert cat1[idx].kron_flux > cat3[idx].kron_flux

    def test_kron_negative(self):
        """
        Test kron negative.
        """
        cat = SourceCatalog(self.data - 10, self.segm)
        assert_allclose(cat.kron_radius.value, cat.kron_params[1])

    def test_kron_photometry(self):
        """
        Test kron photometry.
        """
        flux0, flux_err0 = self.cat.kron_photometry((2.5, 1.4))
        assert_allclose(flux0, self.cat.kron_flux)
        assert_allclose(flux_err0, self.cat.kron_flux_err)

        flux1, flux_err1 = self.cat.kron_photometry((1.0, 1.4), name='kron1')
        flux2, flux_err2 = self.cat.kron_photometry((2.0, 1.4), name='kron2')
        assert_allclose(flux1, self.cat.kron1_flux)
        assert_allclose(flux_err1, self.cat.kron1_flux_err)
        assert_allclose(flux2, self.cat.kron2_flux)
        assert_allclose(flux_err2, self.cat.kron2_flux_err)

        assert np.all((flux2 > flux1) | (np.isnan(flux2) & np.isnan(flux1)))
        assert np.all((flux_err2 > flux_err1)
                      | (np.isnan(flux_err2) & np.isnan(flux_err1)))

        # Test different min Kron radius
        flux3, flux_err3 = self.cat.kron_photometry((2.5, 2.5))
        assert np.all((flux3 > flux0) | (np.isnan(flux3) & np.isnan(flux0)))
        assert np.all((flux_err3 > flux_err0)
                      | (np.isnan(flux_err3) & np.isnan(flux_err0)))

        obj = self.cat[1]
        flux1, flux_err1 = obj.kron_photometry((1.0, 1.4), name='kron0')
        assert np.isscalar(flux1)
        assert np.isscalar(flux_err1)
        assert_allclose(flux1, obj.kron0_flux)
        assert_allclose(flux_err1, obj.kron0_flux_err)

        cat = SourceCatalog(self.data, self.segm)
        _, flux_err = cat.kron_photometry((2.0, 1.4))
        assert np.all(np.isnan(flux_err))

        match = 'kron_params must be 1D'
        with pytest.raises(ValueError, match=match):
            self.cat.kron_photometry(2.5)

        match = r'kron_params\[0\] must be > 0'
        with pytest.raises(ValueError, match=match):
            self.cat.kron_photometry((0.0, 1.4))

        match = r'kron_params\[1\] must be > 0'
        with pytest.raises(ValueError, match=match):
            self.cat.kron_photometry((2.5, 0.0))
        with pytest.raises(ValueError, match=match):
            self.cat.kron_photometry((2.5, 0.0, 1.5))

    def test_circular_photometry(self):
        """
        Test circular photometry.
        """
        flux1, flux_err1 = self.cat.circular_photometry(1.0, name='circ1')
        flux2, flux_err2 = self.cat.circular_photometry(5.0, name='circ5')
        assert_allclose(flux1, self.cat.circ1_flux)
        assert_allclose(flux_err1, self.cat.circ1_flux_err)
        assert_allclose(flux2, self.cat.circ5_flux)
        assert_allclose(flux_err2, self.cat.circ5_flux_err)

        assert np.all((flux2 > flux1) | (np.isnan(flux2) & np.isnan(flux1)))
        assert np.all((flux_err2 > flux_err1)
                      | (np.isnan(flux_err2) & np.isnan(flux_err1)))

        obj = self.cat[1]
        assert obj.isscalar
        flux1, flux_err1 = obj.circular_photometry(1.0, name='circ0')
        assert np.isscalar(flux1)
        assert np.isscalar(flux_err1)
        assert_allclose(flux1, obj.circ0_flux)
        assert_allclose(flux_err1, obj.circ0_flux_err)

        cat = SourceCatalog(self.data, self.segm)
        _, flux_err = cat.circular_photometry(1.0)
        assert np.all(np.isnan(flux_err))

        # With "center" mode, tiny apertures that do not overlap any
        # center should return NaN
        cat2 = self.cat.copy()
        cat2._set_semode()  # sets "center" mode
        flux1, flux_err1 = cat2.circular_photometry(0.1)
        assert np.all(np.isnan(flux1[2:4]))
        assert np.all(np.isnan(flux_err1[2:4]))

        match = 'radius must be > 0'
        with pytest.raises(ValueError, match=match):
            self.cat.circular_photometry(0.0)
        with pytest.raises(ValueError, match=match):
            self.cat.circular_photometry(-1.0)
        with pytest.raises(ValueError, match=match):
            self.cat.make_circular_apertures(0.0)
        with pytest.raises(ValueError, match=match):
            self.cat.make_circular_apertures(-1.0)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plots(self):
        """
        Test plots.
        """
        from matplotlib.patches import Patch

        patches = self.cat.plot_circular_apertures(5.0)
        assert isinstance(patches, list)
        for patch_ in patches:
            assert isinstance(patch_, Patch)

        patches = self.cat.plot_kron_apertures()
        assert isinstance(patches, list)
        for patch_ in patches:
            assert isinstance(patch_, Patch)

        patches2 = self.cat.plot_kron_apertures(kron_params=(2.0, 1.2))
        assert isinstance(patches2, list)
        for patch_ in patches2:
            assert isinstance(patch_, Patch)

        # Test scalar
        obj = self.cat[1]
        patch1 = obj.plot_kron_apertures()
        assert isinstance(patch1, Patch)
        patch2 = obj.plot_kron_apertures(kron_params=(2.0, 1.2))
        assert isinstance(patch2, Patch)

    def test_flux_radius_cache(self):
        """
        Test that flux_radius caches results and reuses them on repeated
        calls with the same flux_radius value.
        """
        cat = SourceCatalog(self.data, self.segm)

        # Cache must start empty
        assert cat._flux_radius_cache == {}

        # First call computes and stores in cache
        r1 = cat.flux_radius(0.5)
        assert 0.5 in cat._flux_radius_cache
        assert r1 is cat._flux_radius_cache[0.5]

        # Second call returns the identical cached object
        r2 = cat.flux_radius(0.5)
        assert r2 is r1

        # Different flux_radius values are cached independently
        r3 = cat.flux_radius(0.3)
        assert 0.3 in cat._flux_radius_cache
        assert np.all(r1.value >= r3.value)

        # Test "name"= still works on a cache hit and stores the
        # attribute
        cat2 = SourceCatalog(self.data, self.segm)
        cat2.flux_radius(0.5)  # populate cache
        cat2.flux_radius(0.5, name='r_hl')  # cache hit with name
        assert hasattr(cat2, 'r_hl')
        assert_allclose(cat2.r_hl, cat2._flux_radius_cache[0.5])

    def test_flux_radius_cache_getitem(self):
        """
        Test that sliced SourceCatalog objects preserve the flux_radius
        cache and produce correct results.
        """
        cat = SourceCatalog(self.data, self.segm)

        # Sliced object without a populated parent cache has an empty
        # cache
        obj = cat[1]
        assert obj._flux_radius_cache == {}

        # Populate the parent cache with two flux_radius values
        r_parent_05 = cat.flux_radius(0.5)
        r_parent_03 = cat.flux_radius(0.3)
        assert 0.5 in cat._flux_radius_cache
        assert 0.3 in cat._flux_radius_cache

        # Scalar slice preserves the cache
        obj = cat[1]
        assert 0.5 in obj._flux_radius_cache
        assert 0.3 in obj._flux_radius_cache
        r_sliced = obj.flux_radius(0.5)
        assert_allclose(r_sliced.value, r_parent_05[1].value)
        r_sliced_03 = obj.flux_radius(0.3)
        assert_allclose(r_sliced_03.value, r_parent_03[1].value)

        # Range slice preserves the cache
        sub = cat[1:3]
        assert 0.5 in sub._flux_radius_cache
        assert 0.3 in sub._flux_radius_cache
        r_sub = sub.flux_radius(0.5)
        assert_allclose(r_sub.value, r_parent_05[1:3].value)

        # Fancy index slice preserves the cache
        sub2 = cat[[0, 2]]
        assert 0.5 in sub2._flux_radius_cache
        r_sub2 = sub2.flux_radius(0.5)
        assert_allclose(r_sub2.value, r_parent_05[[0, 2]].value)

        # Boolean mask slice preserves the cache
        mask = np.array([True, False, True, False, True, False, True])
        sub3 = cat[mask]
        assert 0.5 in sub3._flux_radius_cache
        r_sub3 = sub3.flux_radius(0.5)
        assert_allclose(r_sub3.value, r_parent_05[mask].value)

        # Modifying the parent cache does not affect the sliced cache
        cat.flux_radius(0.7)
        assert 0.7 not in obj._flux_radius_cache

    def test_flux_radius_max_radius_delta(self):
        """
        Test that the max_radius_delta fallback loop reduces max_radius
        by 10 percent on each failed bracketing attempt and still
        returns a valid result when the second (reduced) bracket
        succeeds.
        """
        # Use a single-source scalar catalog to keep the mock simple
        cat = SourceCatalog(self.data, self.segm)[1]
        assert cat.isscalar

        brackets_seen = []
        call_count = [0]

        def mock_root_scalar(fcn, args, bracket, method):
            call_count[0] += 1
            brackets_seen.append(list(bracket))
            if call_count[0] == 1:
                # Simulate a bracket with no sign change
                msg = 'no sign change in bracket'
                raise ValueError(msg)
            return root_scalar(fcn, args=args, bracket=bracket, method=method)

        with patch('photutils.segmentation.catalog.root_scalar',
                   mock_root_scalar):
            r = cat.flux_radius(0.5)

        # Fallback triggered once then succeeded
        assert call_count[0] == 2

        # Second bracket max_radius must be 10% smaller than the first
        assert_allclose(brackets_seen[1][1], 0.9 * brackets_seen[0][1],
                        rtol=1e-10)

        # Result is a valid radius (not NaN)
        assert np.isfinite(r.value)

    def test_flux_radius(self):
        """
        Test flux_radius.
        """
        radius1 = self.cat.flux_radius(0.1, name='flux_radius_r1')
        radius2 = self.cat.flux_radius(0.5, name='flux_radius_r5')
        assert_allclose(radius1, self.cat.flux_radius_r1)
        assert_allclose(radius2, self.cat.flux_radius_r5)
        assert np.all((radius2 > radius1)
                      | (np.isnan(radius2) & np.isnan(radius1)))

        cat = SourceCatalog(self.data, self.segm)
        obj = cat[1]
        radius = obj.flux_radius(0.5)
        assert radius.isscalar  # Quantity radius - can't use np.isscalar
        assert_allclose(radius.value, 7.899648)

        match = 'fraction must be > 0 and <= 1'
        with pytest.raises(ValueError, match=match):
            radius = self.cat.flux_radius(0)
        with pytest.raises(ValueError, match=match):
            radius = self.cat.flux_radius(-1)

        cat = SourceCatalog(self.data - 50.0, self.segm, error=self.error,
                            background=self.background, mask=self.mask,
                            wcs=self.wcs, local_bkg_width=24)
        radius_hl = cat.flux_radius(0.5)
        assert np.isnan(radius_hl[0])

    def test_cutout_units(self):
        """
        Test cutout units.
        """
        obj = self.cat_units[0]
        quantities = (obj.data_cutout, obj.error_cutout,
                      obj.background_cutout)
        ndarray = (obj.segment_cutout, obj.segment_cutout_masked,
                   obj.data_cutout_masked, obj.error_cutout_masked,
                   obj.background_cutout_masked)
        for arr in quantities:
            assert isinstance(arr, u.Quantity)
        for arr in ndarray:
            assert not isinstance(arr, u.Quantity)

    @pytest.mark.parametrize('scalar', [True, False])
    def test_custom_properties(self, scalar):
        """
        Test extra properties.
        """
        cat = SourceCatalog(self.data, self.segm)
        if scalar:
            cat = cat[1]

        segment_snr = cat.segment_flux / cat.segment_flux_err

        match = 'cannot be set because it is a built-in attribute'
        with pytest.raises(ValueError, match=match):
            # Built-in attribute
            cat.add_property('_data', segment_snr)
        with pytest.raises(ValueError, match=match):
            # Built-in property
            cat.add_property('label', segment_snr)
        with pytest.raises(ValueError, match=match):
            # Built-in lazyproperty
            cat.add_property('area', segment_snr)

        cat.add_property('segment_snr', segment_snr)

        match = 'already exists as an attribute'
        with pytest.raises(ValueError, match=match):
            # Already exists
            cat.add_property('segment_snr', segment_snr)

        cat.add_property('segment_snr', 2.0 * segment_snr,
                         overwrite=True)
        assert len(cat.custom_properties) == 1
        assert_equal(cat.segment_snr, 2.0 * segment_snr)

        match = 'is not a defined property'
        with pytest.raises(ValueError, match=match):
            cat.remove_property('invalid')

        cat.remove_property(cat.custom_properties)
        assert len(cat.custom_properties) == 0

        cat.add_property('segment_snr', segment_snr)
        cat.add_property('segment_snr2', segment_snr)
        cat.add_property('segment_snr3', segment_snr)
        assert len(cat.custom_properties) == 3

        cat.remove_properties(cat.custom_properties)
        assert len(cat.custom_properties) == 0

        cat.add_property('segment_snr', segment_snr)
        new_name = 'segment_snr0'
        cat.rename_property('segment_snr', new_name)
        assert new_name in cat.custom_properties

        # Key in extra_properties, but not a defined attribute
        cat._custom_properties.append('invalid')
        match = 'already exists in the custom_properties attribute'
        with pytest.raises(ValueError, match=match):
            cat.add_property('invalid', segment_snr)
        cat._custom_properties.remove('invalid')

        assert cat._has_len([1, 2, 3])
        assert not cat._has_len('test_string')

        cat.add_property('segment_snr4', segment_snr, overwrite=True)
        cat.add_property('segment_snr4', segment_snr, overwrite=True)

    def test_custom_properties_invalid(self):
        """
        Test extra properties invalid.
        """
        cat = SourceCatalog(self.data, self.segm)
        match = 'value must have the same number of elements as the catalog'
        with pytest.raises(ValueError, match=match):
            cat.add_property('invalid', 1.0)
        with pytest.raises(ValueError, match=match):
            cat.add_property('invalid', (1.0, 2.0))

        obj = cat[1]
        with pytest.raises(ValueError, match=match):
            obj.add_property('invalid', (1.0, 2.0))

        val = np.arange(2) << u.km
        with pytest.raises(ValueError, match=match):
            obj.add_property('invalid', val)

        coord = SkyCoord([42, 43], [44, 45], unit='deg')
        with pytest.raises(ValueError, match=match):
            obj.add_property('invalid', coord)

    def test_properties(self):
        """
        Test properties.
        """
        attrs = ('label', 'labels', 'slices', 'x_centroid',
                 'segment_flux', 'kron_flux')
        for attr in attrs:
            assert attr in self.cat.properties

    def test_lazyproperties_class_cache(self):
        """
        Test that _lazyproperties is cached on the class and shared
        across instances.
        """
        cat2 = SourceCatalog(self.data, self.segm)
        result1 = self.cat._lazyproperties
        result2 = cat2._lazyproperties
        assert result1 is result2

    def test_properties_class_cache(self):
        """
        Test that _properties is cached on the class and shared across
        instances.
        """
        cat2 = SourceCatalog(self.data, self.segm)
        result1 = self.cat._properties
        result2 = cat2._properties
        assert result1 is result2

    def test_copy(self):
        """
        Test copy.
        """
        cat = SourceCatalog(self.data, self.segm)
        cat2 = cat.copy()
        _ = cat.kron_flux
        assert 'kron_flux' not in cat2.__dict__
        tbl = cat2.to_table()
        assert len(tbl) == 7

    def test_data_dtype(self):
        """
        Test that input ``data`` with int dtype does not raise
        UFuncTypeError due to subtraction of float array from int
        array.
        """
        data = np.zeros((25, 25), dtype=np.uint16)
        data[8:16, 8:16] = 10
        segmdata = np.zeros((25, 25), dtype=int)
        segmdata[8:16, 8:16] = 1
        segm = SegmentationImage(segmdata)
        cat = SourceCatalog(data, segm, local_bkg_width=3)
        assert cat.min_value == 10
        assert cat.max_value == 10

    def test_make_circular_apertures(self):
        """
        Test make circular apertures.
        """
        radius = 10
        aper = self.cat.make_circular_apertures(radius)
        assert len(aper) == len(self.cat)
        assert isinstance(aper[1], CircularAperture)
        assert aper[1].r == radius

        obj = self.cat[1]
        aper = obj.make_circular_apertures(radius)
        assert isinstance(aper, CircularAperture)
        assert aper.r == radius

    def test_make_kron_apertures(self):
        """
        Test make kron apertures.
        """
        aper = self.cat.make_kron_apertures()
        assert len(aper) == len(self.cat)
        assert isinstance(aper[1], EllipticalAperture)

        aper2 = self.cat.make_kron_apertures(kron_params=(2.0, 1.4))
        assert len(aper2) == len(self.cat)

        obj = self.cat[1]
        aper = obj.make_kron_apertures()
        assert isinstance(aper, EllipticalAperture)

    @pytest.mark.skipif(not HAS_SKIMAGE, reason='skimage is required')
    def test_make_cutouts(self):
        """
        Test make cutouts.
        """
        data = make_100gaussians_image()
        bkg_estimator = MedianBackground()
        bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                           bkg_estimator=bkg_estimator)
        data -= bkg.background  # subtract the background
        threshold = 1.5 * bkg.background_rms
        kernel = make_2dgaussian_kernel(3.0, size=5)
        convolved_data = convolve(data, kernel)
        n_pixels = 10
        finder = SourceFinder(n_pixels=n_pixels, progress_bar=False)
        segment_map = finder(convolved_data, threshold)
        cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)

        shape = (100, 100)

        match = "mode must be 'partial' or 'trim'"
        with pytest.raises(ValueError, match=match):
            cat.make_cutouts(shape, mode='strict')

        cutouts = cat.make_cutouts(shape, mode='trim')
        assert cutouts[0].data.shape != shape
        assert_equal(cutouts[0].xyorigin, np.array((186, 0)))
        assert (cutouts[0].bbox_original
                == BoundingBox(ixmin=186, ixmax=286, iymin=0, iymax=52))

        cutouts = cat.make_cutouts(shape, mode='partial')
        assert_equal(cutouts[0].xyorigin, np.array((186, -48)))
        assert (cutouts[0].bbox_original
                == BoundingBox(ixmin=186, ixmax=286, iymin=0, iymax=52))

        assert len(cutouts) == len(cat)
        assert isinstance(cutouts[1], CutoutImage)
        for cutout in cutouts:
            assert cutout.data.shape == shape

        # Test making cutouts from an input image
        image = np.ones(data.shape)
        cutouts = cat.make_cutouts(shape, array=image, mode='partial')
        for cutout in cutouts:
            assert np.all(cutout.data[np.isfinite(cutout.data)] == 1)
            assert cutout.data.shape == shape

        match = 'array must have the same shape as data'
        with pytest.raises(ValueError, match=match):
            cat.make_cutouts(shape, array=np.ones((3, 3)), mode='partial')

        obj = cat[1]
        cut = obj.make_cutouts(shape)
        assert isinstance(cut, CutoutImage)
        assert cut.data.shape == shape

        cutouts = cat.make_cutouts(shape, mode='partial', fill_value=-100)
        assert cutouts[0].data[0, 0] == -100

        # Cutout will be None if source is completely masked
        cutouts = self.cat.make_cutouts(shape)
        assert cutouts[0] is None

    def test_meta(self):
        """
        Test meta.
        """
        meta = self.cat.meta
        attrs = ['local_bkg_width', 'aperture_mask_method',
                 'kron_params']
        for attr in attrs:
            assert attr in meta

        deprecated_attrs = {
            'localbkg_width': 'local_bkg_width',
            'apermask_method': 'aperture_mask_method',
        }
        for old_name, new_name in deprecated_attrs.items():
            assert old_name in meta
            assert meta[old_name] == meta[new_name]

        tbl = self.cat.to_table()
        assert tbl.meta == self.cat.meta

        out = StringIO()
        tbl.write(out, format='ascii.ecsv')
        tbl2 = QTable.read(out.getvalue(), format='ascii.ecsv')
        # Check order of meta keys
        assert list(tbl2.meta.keys()) == list(tbl.meta.keys())

    def test_meta_deprecated_kwargs(self):
        """
        Test meta aliases for deprecated keyword names.
        """
        with pytest.warns(AstropyDeprecationWarning) as record:
            cat = SourceCatalog(self.data, self.segm, error=self.error,
                                background=self.background, mask=self.mask,
                                wcs=self.wcs, localbkg_width=24,
                                apermask_method='mask')

        assert len(record) == 2
        messages = [str(item.message) for item in record]
        assert any('localbkg_width' in message for message in messages)
        assert any('apermask_method' in message for message in messages)
        assert cat.meta['localbkg_width'] == cat.meta['local_bkg_width']
        assert cat.meta['apermask_method'] == cat.meta['aperture_mask_method']

    def test_meta_future_column_names(self, monkeypatch):
        """
        Test meta with future_column_names enabled.
        """
        import photutils

        monkeypatch.setattr(photutils, 'future_column_names', True)

        with pytest.warns(AstropyDeprecationWarning) as record:
            cat = SourceCatalog(self.data, self.segm, error=self.error,
                                background=self.background, mask=self.mask,
                                wcs=self.wcs, localbkg_width=24,
                                apermask_method='mask')

        assert len(record) == 2
        meta = cat.meta
        attrs = ['local_bkg_width', 'aperture_mask_method', 'kron_params']
        for attr in attrs:
            assert attr in meta

        assert 'localbkg_width' not in meta
        assert 'apermask_method' not in meta

        tbl = cat.to_table()
        assert tbl.meta == meta

    def test_semode(self):
        """
        Test semode.
        """
        self.cat._set_semode()
        tbl = self.cat.to_table()
        assert len(tbl) == 7

    def test_tiny_sources(self):
        """
        Test tiny sources.
        """
        data = np.zeros((11, 11))
        data[5, 5] = 1.0
        data[8, 8] = 1.0
        segm = detect_sources(data, 0.1, 1)
        data[8, 8] = 0
        cat = SourceCatalog(data, segm)
        assert_allclose(cat[0].covariance,
                        [(1 / 12, 0), (0, 1 / 12)] * u.pix**2)
        assert_allclose(cat[1].covariance,
                        [(np.nan, np.nan), (np.nan, np.nan)] * u.pix**2)
        assert_allclose(cat.fwhm, [0.67977799, np.nan] * u.pix)


@pytest.mark.skipif(not HAS_SKIMAGE, reason='skimage is required')
def test_kron_params():
    """
    Test kron params.
    """
    data = make_100gaussians_image()
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                       bkg_estimator=bkg_estimator)
    data -= bkg.background  # subtract the background

    threshold = 1.5 * bkg.background_rms

    kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data = convolve(data, kernel)

    n_pixels = 10
    finder = SourceFinder(n_pixels=n_pixels, progress_bar=False)
    segm = finder(convolved_data, threshold)

    minrad = 1.4
    kron_params = (2.5, minrad, 0.0)
    cat = SourceCatalog(data, segm, convolved_data=convolved_data,
                        kron_params=kron_params)
    assert cat.kron_radius.value.min() == minrad
    assert_allclose(cat.kron_flux.min(), 264.775307)
    rh = cat.flux_radius(0.5)
    assert_allclose(rh.value.min(), 1.293722, rtol=1e-6)

    minrad = 1.2
    kron_params = (2.5, minrad, 0.0)
    cat = SourceCatalog(data, segm, convolved_data=convolved_data,
                        kron_params=kron_params)
    assert cat.kron_radius.value.min() == minrad
    assert_allclose(cat.kron_flux.min(), 264.775307)
    rh = cat.flux_radius(0.5)
    assert_allclose(rh.value.min(), 1.312618, rtol=1e-6)

    minrad = 0.2
    kron_params = (2.5, minrad, 0.0)
    cat = SourceCatalog(data, segm, convolved_data=convolved_data,
                        kron_params=kron_params)
    assert_allclose(cat.kron_radius.value.min(), 0.677399, rtol=1e-6)
    assert_allclose(cat.kron_flux.min(), 264.775307)
    rh = cat.flux_radius(0.5)
    assert_allclose(rh.value.min(), 1.232554)

    kron_params = (2.5, 1.4, 7.0)
    cat = SourceCatalog(data, segm, convolved_data=convolved_data,
                        kron_params=kron_params)
    assert cat.kron_radius.value.min() == 0.0
    assert_allclose(cat.kron_flux.min(), 264.775307)
    rh = cat.flux_radius(0.5)
    assert_allclose(rh.value.min(), 1.288211, rtol=1e-6)
    assert isinstance(cat.kron_aperture[0], CircularAperture)


@pytest.mark.skipif(not HAS_SKIMAGE, reason='skimage is required')
def test_centroid_win(centroid_win_data):
    """
    Test centroid win.
    """
    data, segment_map, convolved_data = centroid_win_data
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data,
                        aperture_mask_method='none')

    assert cat.x_centroid[0] != cat.x_centroid_win[0]
    assert cat.y_centroid[0] != cat.y_centroid_win[0]
    # centroid_win moved beyond 1-sigma ellipse and was reset to
    # isophotal centroid
    assert cat.x_centroid[1] == cat.x_centroid_win[1]
    assert cat.y_centroid[1] == cat.y_centroid_win[1]


def test_centroid_win_migrate():
    """
    Test that when the windowed centroid moves the aperture completely
    off the image the isophotal centroid is returned.
    """
    g1 = Gaussian2D(1621, 76.29, 185.95, 1.55, 1.29, 0.296706)
    g2 = Gaussian2D(3596, 83.81, 182.29, 1.44, 1.27, 0.628319)
    m = g1 + g2
    yy, xx = np.mgrid[0:256, 0:256]
    data = m(xx, yy)
    noise = make_noise_image(data.shape, mean=0, stddev=65.0, seed=123)
    data += noise
    segm = detect_sources(data, 98.0, n_pixels=5)
    cat = SourceCatalog(data, segm)

    indices = (0, 3, 14, 30)
    for idx in indices:
        assert_equal(cat.centroid_win[idx], cat.centroid[idx])


def test_background_centroid_coordinate_order():
    """
    Test that the background_centroid property correctly passes (y, x)
    coordinates to map_coordinates.
    """
    yy, xx = np.mgrid[0:101, 0:101]
    # Background varies only along y (rows)
    background = yy.astype(float)

    g1 = Gaussian2D(200, 50, 25, 5, 5)
    g2 = Gaussian2D(200, 50, 75, 5, 5)
    data = g1(xx, yy) + g2(xx, yy)
    segm = detect_sources(data, 30.0, n_pixels=5)

    cat = SourceCatalog(data, segm, background=background)
    bkg_cen = cat.background_centroid

    # The expected value at each centroid is approximately y_centroid
    # (since background = y)
    for i in range(cat.n_labels):
        xcen = cat.x_centroid[i]
        ycen = cat.y_centroid[i]
        if np.isfinite(xcen) and np.isfinite(ycen):
            # The interpolated background at the centroid should be
            # close to ycen (not xcen)
            assert_allclose(bkg_cen[i], ycen, atol=0.5)
            # If x != y, the wrong order would give a value close to
            # xcen instead
            if abs(xcen - ycen) > 5:
                assert abs(bkg_cen[i] - xcen) > 2


def test_aperture_mask_method_none():
    """
    Test that circular_photometry with aperture_mask_method='none' does not
    mask neighboring sources.
    """
    yy, xx = np.mgrid[0:101, 0:101]
    # Two overlapping sources
    g1 = Gaussian2D(200, 40, 50, 8, 8)
    g2 = Gaussian2D(200, 60, 50, 8, 8)
    data = g1(xx, yy) + g2(xx, yy)
    segm = detect_sources(data, 20.0, n_pixels=5)

    cat_none = SourceCatalog(data, segm, aperture_mask_method='none')
    cat_mask = SourceCatalog(data, segm, aperture_mask_method='mask')

    # Use a large aperture that overlaps both sources
    flux_none, _ = cat_none.circular_photometry(20.0)
    flux_mask, _ = cat_mask.circular_photometry(20.0)

    # 'none' should include neighbor flux, so should be >= 'mask'
    for i in range(cat_none.n_labels):
        if np.isfinite(flux_none[i]) and np.isfinite(flux_mask[i]):
            assert flux_none[i] >= flux_mask[i]


def test_flux_radius_nan_fallback():
    """
    Test that flux_radius returns NaN when no root can be found (e.g.,
    when the source has all-negative Kron flux within the search bracket
    or zero Kron flux).
    """
    # Create a source with negative total flux by subtracting a large
    # constant. The Kron flux will be zero/negative, causing flux_radius
    # to return NaN.
    yy, xx = np.mgrid[0:51, 0:51]
    g1 = Gaussian2D(10, 25, 25, 3, 3)
    data = g1(xx, yy) - 50.0  # all negative
    segm_data = np.zeros((51, 51), dtype=int)
    segm_data[20:31, 20:31] = 1
    segm = SegmentationImage(segm_data)

    cat = SourceCatalog(data, segm)
    radius = cat.flux_radius(0.5)
    # Should be NaN since there's no meaningful flux
    assert np.isnan(radius.value)


def test_reduceat_empty_input():
    """
    Test that _reduceat returns empty arrays when given an empty list.
    """
    result, sizes = SourceCatalog._reduceat([], np.add)
    assert len(result) == 0
    assert len(sizes) == 0
    assert sizes.dtype == int


def test_reduceat_negative_data():
    """
    Test that the _reduceat optimization gives correct results for
    min_value, max_value, and segment_flux when data contains negative
    pixel values.
    """
    yy, xx = np.mgrid[0:101, 0:101]
    g1 = Gaussian2D(100, 30, 30, 5, 5)
    g2 = Gaussian2D(80, 70, 70, 4, 4)
    data = g1(xx, yy) + g2(xx, yy) - 20.0  # shift so many pixels negative
    segm = detect_sources(data, 0.5, n_pixels=5)

    cat = SourceCatalog(data, segm)
    for i in range(cat.n_labels):
        obj = cat[i]
        vals = obj._data_values[0]
        expected_min = np.min(vals) - obj._local_background
        expected_max = np.max(vals) - obj._local_background
        expected_flux = np.sum(vals) - obj._local_background * len(vals)
        assert_allclose(obj.min_value, expected_min)
        assert_allclose(obj.max_value, expected_max)
        assert_allclose(obj.segment_flux, expected_flux)


def test_make_cutouts_trim_mode():
    """
    Test that make_cutouts with mode='trim' returns cutouts that are
    correctly trimmed when they extend beyond the array boundary.
    """
    yy, xx = np.mgrid[0:101, 0:101]
    # Source near the edge
    g1 = Gaussian2D(100, 5, 5, 3, 3)
    # Source in the center
    g2 = Gaussian2D(100, 50, 50, 3, 3)
    data = g1(xx, yy) + g2(xx, yy)
    segm = detect_sources(data, 10, n_pixels=5)

    cat = SourceCatalog(data, segm)
    shape = (40, 40)
    cutouts = cat.make_cutouts(shape, mode='trim')

    for cutout in cutouts:
        if cutout is None:
            continue
        # Trim mode: cutout shape should be <= requested shape
        assert cutout.data.shape[0] <= shape[0]
        assert cutout.data.shape[1] <= shape[1]
        assert isinstance(cutout, CutoutImage)

    # At least one near-edge source should be trimmed (smaller than
    # shape)
    shapes = [c.data.shape for c in cutouts if c is not None]
    assert any(s != shape for s in shapes)

    # Center source should be full size
    assert any(s == shape for s in shapes)


def test_progress_bar_centroid_win(progress_bar_catalog):
    """
    Test that centroid_win works with progress_bar=True.
    """
    cat = progress_bar_catalog
    cwin = cat.centroid_win
    assert cwin.shape == (cat.n_labels, 2)
    assert np.all(np.isfinite(cwin))


def test_progress_bar_centroid_quad(progress_bar_catalog):
    """
    Test that centroid_quad works with progress_bar=True.
    """
    cat = progress_bar_catalog
    cquad = cat.centroid_quad
    assert cquad.shape == (cat.n_labels, 2)
    assert np.all(np.isfinite(cquad))


def test_progress_bar_kron_radius(progress_bar_catalog):
    """
    Test that kron_radius works with progress_bar=True.
    """
    cat = progress_bar_catalog
    kr = cat.kron_radius
    assert len(kr) == cat.n_labels


def test_progress_bar_kron_photometry(progress_bar_catalog):
    """
    Test that kron_photometry (aperture photometry) works with
    progress_bar=True.
    """
    cat = progress_bar_catalog
    flux, _flux_err = cat.kron_photometry((2.5, 1.4))
    assert len(flux) == cat.n_labels


def test_progress_bar_flux_radius(progress_bar_catalog):
    """
    Test that flux_radius and its prep work with progress_bar=True.
    """
    cat = progress_bar_catalog
    r = cat.flux_radius(0.5)
    assert len(r) == cat.n_labels


def test_progress_bar_circular_photometry(progress_bar_catalog):
    """
    Test that circular_photometry works with progress_bar=True.
    """
    cat = progress_bar_catalog
    flux, _fluxerr = cat.circular_photometry(5.0)
    assert len(flux) == cat.n_labels


def test_negative_covariance_eigvals(single_source_catalog):
    """
    Test that negative eigenvalues in the covariance matrix are
    replaced with NaN.
    """
    _data, _segm, cat = single_source_catalog

    # Patch np.linalg.eigvalsh to return negative eigenvalues
    real_eigvalsh = np.linalg.eigvalsh

    def mock_eigvalsh(a):
        result = real_eigvalsh(a)
        result[:] = -1.0  # force negative eigenvalues
        return result

    with patch('numpy.linalg.eigvalsh', mock_eigvalsh):
        eigvals = cat.covariance_eigvals
    assert np.all(np.isnan(eigvals.value))


def test_local_background_few_pixels():
    """
    Test that _local_background returns 0 when fewer than 10 unmasked
    pixels are available in the local background annulus.
    """
    # Create a tiny image where the background annulus will have very
    # few unmasked pixels
    data = np.zeros((11, 11))
    data[4:7, 4:7] = 100.0
    segm_data = np.zeros((11, 11), dtype=int)
    segm_data[4:7, 4:7] = 1
    segm = SegmentationImage(segm_data)

    # Use a large mask that leaves fewer than 10 pixels in the annulus
    mask = np.ones((11, 11), dtype=bool)
    # Unmask only the source and a thin border
    mask[3:8, 3:8] = False

    cat = SourceCatalog(data, segm, mask=mask, local_bkg_width=2)
    bkg = cat._local_background
    assert bkg[0] == 0.0


def test_validate_kron_params_wrong_element_count():
    """
    Test that _validate_kron_params raises ValueError for wrong number
    of elements.
    """
    match = 'kron_params must have 2 or 3 elements'
    with pytest.raises(ValueError, match=match):
        SourceCatalog._validate_kron_params([2.5, 1.4, 0.0, 99.0])

    with pytest.raises(ValueError, match=match):
        SourceCatalog._validate_kron_params([2.5])


def test_error_values_with_error(single_source_catalog):
    """
    Test that _error_values returns null objects when error is None.
    """
    _data, _segm, cat = single_source_catalog
    err_vals = cat._error_values
    assert err_vals is cat._null_objects


def test_background_values_with_background(single_source_catalog):
    """
    Test that _background_values returns null objects when background is
    None.
    """
    _data, _segm, cat = single_source_catalog
    bkg_vals = cat._background_values
    assert bkg_vals is cat._null_objects


def test_sky_centroid_quad_with_wcs(single_source_catalog):
    """
    Test that sky_centroid_quad returns a coordinate when wcs is
    provided.
    """
    data, segm, _cat = single_source_catalog
    wcs = make_wcs(data.shape)
    cat = SourceCatalog(data, segm, wcs=wcs)
    sky_quad = cat.sky_centroid_quad
    assert sky_quad is not None


def test_sky_centroid_quad_no_wcs(single_source_catalog):
    """
    Test that sky_centroid_quad returns None when wcs is not provided.
    """
    _data, _segm, cat = single_source_catalog
    sky_quad = cat.sky_centroid_quad
    # Single source returns scalar None (from _null_objects)
    assert sky_quad is None or np.all(sky_quad == np.array(None))


def test_centroid_quad_edge_cases():
    """
    Test cutout_centroid_quad edge cases.
    """
    # Small cutout (< 3x3) triggers NaN fallback
    data = np.zeros((10, 10))
    segm_data = np.zeros((10, 10), dtype=int)
    data[0, 4:6] = [5.0, 3.0]
    segm_data[0, 4:6] = 1
    data[5, 5] = 100.0
    segm_data[4:7, 4:7] = 2
    segm = SegmentationImage(segm_data)
    cat = SourceCatalog(data, segm)
    cquad = cat.cutout_centroid_quad
    assert cquad.shape == (2, 2)
    assert np.all(np.isfinite(cquad))

    # Checkerboard pattern triggers det <= 0 or c20 > 0
    data3 = np.zeros((7, 7))
    data3[3, 3] = 10.0
    data3[2, 2] = 9.0
    data3[4, 4] = 9.0
    data3[2, 4] = 9.0
    data3[4, 2] = 9.0
    data3[3, 2] = 1.0
    data3[3, 4] = 1.0
    data3[2, 3] = 1.0
    data3[4, 3] = 1.0
    segm_data3 = np.zeros((7, 7), dtype=int)
    segm_data3[1:6, 1:6] = 1
    segm3 = SegmentationImage(segm_data3)
    cat3 = SourceCatalog(data3, segm3)
    cquad3 = cat3.cutout_centroid_quad
    assert np.all(np.isfinite(cquad3))

    # Quadratic max falls outside cutout bounds.
    # Use a 3x3 segment so cutout is exactly 3x3 (xidx0=0, yidx0=0), and
    # the relative quadratic max falls outside [0, 2].
    data4 = np.zeros((7, 7))
    box4 = np.array([[7.45, 9.68, 3.26],
                     [3.70, 10.67, 1.89],
                     [1.30, 4.76, 2.27]])
    data4[2:5, 2:5] = box4
    segm_data4 = np.zeros((7, 7), dtype=int)
    segm_data4[2:5, 2:5] = 1
    segm4 = SegmentationImage(segm_data4)
    cat4 = SourceCatalog(data4, segm4)
    cquad4 = cat4.cutout_centroid_quad
    assert np.all(np.isfinite(cquad4))


def test_flux_radius_no_solution(single_source_catalog):
    """
    Test that flux_radius returns NaN when no solution is found
    (root_scalar always raises ValueError).
    """
    _data, _segm, cat = single_source_catalog

    # Make root_scalar always raise ValueError so no solution is found
    def mock_root_scalar(*_args, **_kwargs):
        msg = 'bracket signs'
        raise ValueError(msg)

    with patch('photutils.segmentation.catalog.root_scalar',
               mock_root_scalar):
        result = cat.flux_radius(0.5)
    assert np.isnan(result.value[0])


def test_kron_radius_max(gauss_101_catalog):
    """
    Test that measured kron_radius values exceeding the measurement
    aperture scale (6.0) are set to NaN.

    This flags unphysical Kron radii caused by near-cancellation in the
    denominator of the Kron formula (e.g., due to outlier pixels or
    noise).
    """
    data, segm, cat = gauss_101_catalog

    # The measured kron radius should be reasonable (<= 6.0)
    assert cat.kron_radius.value <= 6.0

    # Artificially test the NaN by patching _measured_kron_radius to
    # return a huge value
    with patch.object(type(cat), '_measured_kron_radius',
                      new_callable=lambda: property(
                          lambda _self: np.array([100.0]))):
        cat2 = SourceCatalog(data, segm)
        assert np.isnan(cat2.kron_radius.value)

        # Downstream properties should also be NaN / None
        assert cat2.kron_aperture[0] is None
        assert np.isnan(cat2.kron_flux[0])
        assert np.isnan(cat2.flux_radius(0.5).value[0])


def test_aperture_to_mask_size_check():
    """
    Test that _aperture_to_mask returns None when the aperture bounding
    box exceeds the allowed size, preventing out-of-memory errors from
    pathologically large apertures (e.g., from huge Kron radii).

    The check happens before to_mask() allocates the mask array.
    """
    data = np.zeros((11, 11))
    data[4:7, 4:7] = 100.0
    segm_data = np.zeros((11, 11), dtype=int)
    segm_data[4:7, 4:7] = 1
    segm = SegmentationImage(segm_data)
    cat = SourceCatalog(data, segm)

    # A small aperture is fine
    small_aper = CircularAperture((5, 5), r=3)
    result = cat._aperture_to_mask(small_aper, method='center')
    assert result is not None

    # An aperture larger than data.size but within the 1M floor is fine
    big_aper = CircularAperture((5, 5), r=100)
    assert big_aper.bbox.shape[0] * big_aper.bbox.shape[1] > data.size
    result = cat._aperture_to_mask(big_aper, method='center')
    assert result is not None

    # An aperture whose bbox exceeds 1_000_000 pixels returns None
    huge_aper = CircularAperture((5, 5), r=600)
    bbox_size = huge_aper.bbox.shape[0] * huge_aper.bbox.shape[1]
    assert bbox_size > 1_000_000
    result = cat._aperture_to_mask(huge_aper, method='center')
    assert result is None


def test_aperture_to_mask_none_branches(gauss_101_catalog):
    """
    Test that _aperture_photometry gracefully returns NaN when
    _aperture_to_mask returns None (i.e., aperture too large to
    allocate).

    This covers the None-guard branch in _aperture_photometry (used by
    the general circle photometry path).
    """
    _, _, cat = gauss_101_catalog

    # _aperture_photometry (general path) with _aperture_to_mask
    # returning None
    with patch.object(type(cat), '_aperture_to_mask', return_value=None):
        circ_aper = [CircularAperture((50, 50), r=10)] * cat.n_labels
        flux, _ = cat._aperture_photometry(circ_aper, method='exact')
        assert np.all(np.isnan(flux))


def test_kron_photometry_oom_guard(gauss_101_catalog):
    """
    Test that _calc_kron_photometry returns NaN when the Kron aperture
    is too large (OOM guard).
    """
    data, segm, cat = gauss_101_catalog
    _ = cat.kron_aperture  # cache

    # Create huge elliptical apertures that exceed the max_size check
    huge_aper = [EllipticalAperture((50, 50), 2000, 2000, theta=0.0)
                 for _ in range(cat.n_labels)]
    cat.__dict__['kron_aperture'] = huge_aper
    assert np.all(np.isnan(cat.kron_flux))

    # Create huge circular apertures that exceed the max_size check
    cat2 = SourceCatalog(data, segm)
    _ = cat2.kron_aperture
    huge_circ = [CircularAperture((50, 50), r=2000)
                 for _ in range(cat2.n_labels)]
    cat2.__dict__['kron_aperture'] = huge_circ
    assert np.all(np.isnan(cat2.kron_flux))

    # Aperture completely off-image triggers data=None guard
    cat3 = SourceCatalog(data, segm)
    _ = cat3.kron_aperture
    off_aper = [EllipticalAperture((-1000, -1000), 5, 3, theta=0.0)
                for _ in range(cat3.n_labels)]
    cat3.__dict__['kron_aperture'] = off_aper
    assert np.all(np.isnan(cat3.kron_flux))

    # All pixels masked in aperture overlap triggers empty values with
    # error branch
    error = np.ones_like(data)
    cat4 = SourceCatalog(data, segm, error=error)
    _ = cat4.kron_aperture  # cache
    original_make = type(cat4)._make_aperture_data

    def _make_all_masked(self, label, xcen, ycen, bbox, bkg, **kwargs):
        result = original_make(self, label, xcen, ycen, bbox, bkg, **kwargs)
        if result[0] is not None:
            # Set mask to all True (all pixels masked)
            return (result[0], result[1], np.ones_like(result[2]),
                    result[3], result[4])
        return result

    with patch.object(type(cat4), '_make_aperture_data', _make_all_masked):
        assert np.all(np.isnan(cat4.kron_flux))
        assert np.all(np.isnan(cat4.kron_flux_err))


def test_flux_radius_cache_not_mutated_by_centroid_win(gauss_101_data):
    """
    Test that calling centroid_win before flux_radius does not corrupt
    the flux_radius cache.

    ``centroid_win`` calls ``flux_radius(0.5)`` internally and then
    modifies the returned half-light radius array in-place (replacing
    NaN with a minimum radius). This must not mutate the cached
    ``flux_radius`` Quantity.

    Regression test for a bug where ``.value`` returned a view of the
    cached Quantity's internal array, causing in-place modification.
    """
    data, segm = gauss_101_data
    cat = SourceCatalog(data, segm)

    # Patch _measured_kron_radius to return a value > 6.0 so that
    # kron_radius is NaN, which makes flux_radius return NaN
    with patch.object(type(cat), '_measured_kron_radius',
                      new_callable=lambda: property(
                          lambda _self: np.array([100.0]))):
        cat2 = SourceCatalog(data, segm)
        assert np.isnan(cat2.kron_radius.value[0])

        # Call centroid_win first (it internally calls flux_radius(0.5))
        _ = cat2.centroid_win

        # flux_radius(0.5) must still return NaN, not 0.5
        result = cat2.flux_radius(0.5)
        assert np.all(np.isnan(result.value))


def test_centroid_win_nan_when_flux_radius_nan(gauss_101_data):
    """
    Test that centroid_win returns NaN when flux_radius(0.5) is NaN
    (e.g., because kron_radius is NaN).
    """
    data, segm = gauss_101_data

    # Patch _measured_kron_radius to return a value > 6.0 so that
    # kron_radius is NaN, which makes flux_radius return NaN
    with patch.object(type(SourceCatalog(data, segm)),
                      '_measured_kron_radius',
                      new_callable=lambda: property(
                          lambda _self: np.array([100.0]))):
        cat = SourceCatalog(data, segm)
        assert np.isnan(cat.kron_radius.value[0])
        assert np.isnan(cat.flux_radius(0.5).value[0])

        cwin = cat.centroid_win
        assert np.all(np.isnan(cwin))


def test_centroid_win_aperture_mask_none_in_loop(gauss_101_catalog):
    """
    Test that centroid_win falls back to the isophotal centroid when
    _aperture_to_mask returns None during the iteration loop (e.g.,
    because the circular aperture exceeds the size threshold).
    """
    _data, _segm, cat = gauss_101_catalog

    # Pre-compute flux_radius before mocking, since it also uses
    # CircularAperture internally
    hl_val = cat.flux_radius(0.5)
    original_method = cat._aperture_to_mask

    def mock_aperture_to_mask(aperture, **kwargs):
        if isinstance(aperture, CircularAperture):
            return None
        return original_method(aperture, **kwargs)

    with patch.object(cat, '_aperture_to_mask',
                      side_effect=mock_aperture_to_mask), \
         patch.object(type(cat), 'flux_radius',
                      return_value=hl_val):
        cwin = cat.centroid_win
        # NaN from the loop resets to isophotal centroid
        # because nan_hl is False (flux_radius was valid)
        assert_allclose(cwin[:, 0], cat.x_centroid)
        assert_allclose(cwin[:, 1], cat.y_centroid)


def test_centroid_win_oom_guard(gauss_101_catalog):
    """
    Test that centroid_win returns NaN for sources whose half-light
    radius would require an aperture larger than max_aper_size.
    """
    _data, _segm, cat = gauss_101_catalog

    # Provide a huge half-light radius so the aperture bbox exceeds
    # max_aper_size (max(data.size, 1_000_000) = 1_000_000).
    huge_radius = np.array([200.0]) * u.pix
    with patch.object(type(cat), 'flux_radius', return_value=huge_radius):
        cwin = cat.centroid_win
    # OOM guard should force NaN, which then resets to isophotal
    # centroid (because nan_hl is False)
    assert_allclose(cwin[:, 0], cat.x_centroid)
    assert_allclose(cwin[:, 1], cat.y_centroid)


@pytest.mark.skipif(not HAS_SKIMAGE, reason='skimage is required')
def test_centroid_win_aperture_mask_mask(centroid_win_data):
    """
    Test centroid_win with aperture_mask_method='mask' to cover the
    ``data_mask = data_mask | segm_mask`` branch.
    """
    data, segment_map, convolved_data = centroid_win_data
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data,
                        aperture_mask_method='mask')

    # Verify it runs without error and returns finite values for at
    # least the first source
    cwin = cat.centroid_win
    assert cwin.shape == (len(cat), 2)
    assert np.isfinite(cwin[0, 0])


def test_make_aperture_data_outside_image(gauss_101_catalog):
    """
    Test that _make_aperture_data returns (None,) * 5 when the aperture
    bbox does not overlap the data.
    """
    _data, _segm, cat = gauss_101_catalog

    # BoundingBox completely outside the 101x101 data
    offimage_bbox = BoundingBox(ixmin=200, ixmax=210,
                                iymin=200, iymax=210)
    result = cat._make_aperture_data(1, 205.0, 205.0, offimage_bbox, 0.0)
    assert result == (None,) * 5


def test_flux_radius_optimizer_args_oom_guard(gauss_101_catalog):
    """
    Test that _flux_radius_optimizer_args returns None for sources whose
    max-radius aperture bbox exceeds max_aper_size.
    """
    data, segm, cat = gauss_101_catalog

    # Cache kron_photometry normally, then patch
    # _max_circular_kron_radius to return a huge radius that triggers
    # the OOM guard
    _ = cat._kron_photometry
    huge = np.array([2000.0])
    with patch.object(type(cat), '_max_circular_kron_radius',
                      new_callable=lambda: property(lambda _self: huge)):
        cat2 = SourceCatalog(data, segm)
        cat2.__dict__['_kron_photometry'] = cat._kron_photometry
        cat2.__dict__['_max_circular_kron_radius'] = huge
        assert np.all(np.isnan(cat2.flux_radius(0.5)))


def test_flux_radius_optimizer_args_off_image(gauss_101_catalog):
    """
    Test that _flux_radius_optimizer_args returns None for sources whose
    max-radius aperture bbox doesn't overlap the data.
    """
    _data, _segm, cat = gauss_101_catalog

    # Cache kron_photometry, then move the centroid way off-image
    _ = cat._kron_photometry
    off = np.array([500.0])
    cat.__dict__['_x_centroid'] = off
    cat.__dict__['_y_centroid'] = off
    assert np.all(np.isnan(cat.flux_radius(0.5)))


def test_flux_radius_optimizer_args_center_method(gauss_101_catalog):
    """
    Test _flux_radius_optimizer_args with method='center' to cover the
    center-method branch in the method translation logic.
    """
    _data, _segm, cat = gauss_101_catalog
    cat._aperture_mask_kwargs['flux_radius'] = {'method': 'center'}
    r50 = cat.flux_radius(0.5)
    assert np.isfinite(r50.value[0])


def test_flux_radius_optimizer_args_subpixel_method(gauss_101_catalog):
    """
    Test _flux_optimizer_args with method='subpixel' to cover the
    subpixel-method branch in the method translation logic.
    """
    _data, _segm, cat = gauss_101_catalog
    cat._aperture_mask_kwargs['flux_radius'] = {'method': 'subpixel',
                                                'subpixels': 5}
    r50 = cat.flux_radius(0.5)
    assert np.isfinite(r50.value[0])


def test_measured_kron_radius_oom_guard(gauss_101_catalog):
    """
    Test that _measured_kron_radius returns NaN for sources whose
    aperture bounding box exceeds max_size (OOM guard).
    """
    _data, _segm, cat = gauss_101_catalog

    # Patch semimajor_axis to a huge value so the bbox triggers OOM
    huge = np.array([1e6]) << u.pix
    with patch.object(type(cat), 'semimajor_axis',
                      new_callable=lambda: property(lambda _self: huge)):
        assert np.all(np.isnan(cat.kron_radius.value))


def test_measured_kron_radius_off_image(gauss_101_catalog):
    """
    Test that _measured_kron_radius returns NaN for sources whose
    aperture bounding box doesn't overlap the data.
    """
    _data, _segm, cat = gauss_101_catalog

    # Move the centroid off the image
    cat.__dict__['_x_centroid'] = np.array([5000.0])
    cat.__dict__['_y_centroid'] = np.array([5000.0])
    assert np.all(np.isnan(cat.kron_radius.value))


def test_measured_kron_radius_circular_fallback(gauss_101_data):
    """
    Test _measured_kron_radius with the circular aperture fallback when
    semimajor/semiminor sigma are zero (kron_params[2] > 0).
    """
    data, segm = gauss_101_data
    cat = SourceCatalog(data, segm, kron_params=(2.5, 1.4, 5.0))

    # Force semimajor/semiminor to zero to trigger circular fallback.
    # Also patch ellipse_cxx/ellipse_cxy/ellipse_cyy to valid values
    # since the real ones depend on covariance eigenvalues.
    zero = np.array([0.0]) << u.pix
    cxx_val = np.array([1.0]) / (u.pix * u.pix)
    cyy_val = np.array([1.0]) / (u.pix * u.pix)
    cxy_val = np.array([0.0]) / (u.pix * u.pix)
    with (patch.object(type(cat), 'semimajor_axis',
                       new_callable=lambda: property(lambda _self: zero)),
          patch.object(type(cat), 'semiminor_axis',
                       new_callable=lambda: property(lambda _self: zero)),
          patch.object(type(cat), 'ellipse_cxx',
                       new_callable=lambda: property(lambda _self: cxx_val)),
          patch.object(type(cat), 'ellipse_cyy',
                       new_callable=lambda: property(lambda _self: cyy_val)),
          patch.object(type(cat), 'ellipse_cxy',
                       new_callable=lambda: property(lambda _self: cxy_val))):
        kr = cat._measured_kron_radius
        assert np.isfinite(kr[0])


def test_measured_kron_radius_circular_no_min_radius(gauss_101_data):
    """
    Test _measured_kron_radius returns NaN for the circular fallback
    when kron_params has only 2 elements (no minimum circular radius).
    """
    data, segm = gauss_101_data
    cat = SourceCatalog(data, segm, kron_params=(2.5, 1.4))

    zero = np.array([0.0]) << u.pix
    cxx_val = np.array([1.0]) / (u.pix * u.pix)
    cyy_val = np.array([1.0]) / (u.pix * u.pix)
    cxy_val = np.array([0.0]) / (u.pix * u.pix)
    with (patch.object(type(cat), 'semimajor_axis',
                       new_callable=lambda: property(lambda _self: zero)),
          patch.object(type(cat), 'semiminor_axis',
                       new_callable=lambda: property(lambda _self: zero)),
          patch.object(type(cat), 'ellipse_cxx',
                       new_callable=lambda: property(lambda _self: cxx_val)),
          patch.object(type(cat), 'ellipse_cyy',
                       new_callable=lambda: property(lambda _self: cyy_val)),
          patch.object(type(cat), 'ellipse_cxy',
                       new_callable=lambda: property(lambda _self: cxy_val))):
        kr = cat._measured_kron_radius
        assert np.all(np.isnan(kr))
