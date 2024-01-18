# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the catalog module.
"""

from io import StringIO

import astropy.units as u
import numpy as np
import pytest
from astropy.convolution import convolve
from astropy.coordinates import SkyCoord
from astropy.modeling.models import Gaussian2D
from astropy.table import QTable
from numpy.testing import assert_allclose, assert_equal

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
                                            HAS_SCIPY, HAS_SKIMAGE)
from photutils.utils.cutouts import CutoutImage


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
class TestSourceCatalog:
    def setup_class(self):
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
        self.segm = detect_sources(self.data, threshold, npixels=5)
        self.error = make_noise_image(self.data.shape, mean=0, stddev=2.0,
                                      seed=123)
        self.background = np.ones(self.data.shape) * 5.1
        self.mask = np.zeros(self.data.shape, dtype=bool)
        self.mask[0:30, 0:30] = True

        self.wcs = make_wcs(self.data.shape)
        self.cat = SourceCatalog(self.data, self.segm, error=self.error,
                                 background=self.background, mask=self.mask,
                                 wcs=self.wcs, localbkg_width=24)
        unit = u.nJy
        self.unit = unit
        self.cat_units = SourceCatalog(self.data << unit, self.segm,
                                       error=self.error << unit,
                                       background=self.background << unit,
                                       mask=self.mask, wcs=self.wcs,
                                       localbkg_width=24)

    @pytest.mark.parametrize('with_units', (True, False))
    def test_catalog(self, with_units):
        if with_units:
            cat1 = self.cat_units.copy()
            cat2 = self.cat_units.copy()
        else:
            cat1 = self.cat.copy()
            cat2 = self.cat.copy()

        props = self.cat.properties

        # add extra properties
        cat1.circular_photometry(5.0, name='circ5')
        cat1.kron_photometry((2.5, 1.4), name='kron2')
        cat1.fluxfrac_radius(0.5, name='r_hl')
        segment_snr = cat1.segment_flux / cat1.segment_fluxerr
        cat1.add_extra_property('segment_snr', segment_snr)
        props = list(props)
        props.extend(cat1.extra_properties)

        idx = 1  # no NaN values

        # evaluate (cache) catalog properties before slice
        obj = cat1[idx]
        for prop in props:
            assert_equal(getattr(cat1, prop)[idx], getattr(obj, prop))

        # slice catalog before evaluating catalog properties
        obj = cat2[idx]
        obj.circular_photometry(5.0, name='circ5')
        obj.kron_photometry((2.5, 1.4), name='kron2')
        obj.fluxfrac_radius(0.5, name='r_hl')
        segment_snr = obj.segment_flux / obj.segment_fluxerr
        obj.add_extra_property('segment_snr', segment_snr)
        for prop in props:
            assert_equal(getattr(obj, prop), getattr(cat1, prop)[idx])

        with pytest.raises(ValueError):
            cat1._prepare_cutouts(cat1._segment_img_cutouts, units=True,
                                  masked=True)

    @pytest.mark.parametrize('with_units', (True, False))
    def test_catalog_detection_cat(self, with_units):
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
                                 localbkg_width=24, detection_cat=None)
            cat3 = SourceCatalog(data2 << self.unit, self.segm,
                                 error=error << self.unit,
                                 background=self.background << self.unit,
                                 mask=self.mask, wcs=self.wcs,
                                 localbkg_width=24, detection_cat=cat1)
        else:
            cat1 = self.cat.copy()
            cat2 = SourceCatalog(data2, self.segm, error=error,
                                 background=self.background, mask=self.mask,
                                 wcs=self.wcs, localbkg_width=24,
                                 detection_cat=None)
            cat3 = SourceCatalog(data2, self.segm, error=error,
                                 background=self.background, mask=self.mask,
                                 wcs=self.wcs, localbkg_width=24,
                                 detection_cat=cat1)

        assert_equal(cat1.kron_radius, cat3.kron_radius)
        # assert not equal
        with pytest.raises(AssertionError):
            assert_equal(cat1.kron_radius, cat2.kron_radius)

        with pytest.raises(AssertionError):
            assert_equal(cat2.kron_flux, cat3.kron_flux)
        with pytest.raises(AssertionError):
            assert_equal(cat2.kron_fluxerr, cat3.kron_fluxerr)
        with pytest.raises(AssertionError):
            assert_equal(cat1.kron_flux, cat3.kron_flux)
        with pytest.raises(AssertionError):
            assert_equal(cat1.kron_fluxerr, cat3.kron_fluxerr)

        flux1, fluxerr1 = cat1.circular_photometry(1.0)
        flux2, fluxerr2 = cat2.circular_photometry(1.0)
        flux3, fluxerr3 = cat3.circular_photometry(1.0)
        with pytest.raises(AssertionError):
            assert_equal(flux2, flux3)
        with pytest.raises(AssertionError):
            assert_equal(fluxerr2, fluxerr3)
        with pytest.raises(AssertionError):
            assert_equal(flux1, flux2)
        with pytest.raises(AssertionError):
            assert_equal(fluxerr1, fluxerr2)

        flux1, fluxerr1 = cat1.kron_photometry((2.5, 1.4))
        flux2, fluxerr2 = cat2.kron_photometry((2.5, 1.4))
        flux3, fluxerr3 = cat3.kron_photometry((2.5, 1.4))
        with pytest.raises(AssertionError):
            assert_equal(flux2, flux3)
        with pytest.raises(AssertionError):
            assert_equal(fluxerr2, fluxerr3)
        with pytest.raises(AssertionError):
            assert_equal(flux1, flux2)
        with pytest.raises(AssertionError):
            assert_equal(fluxerr1, fluxerr2)

        radius1 = cat1.fluxfrac_radius(0.5)
        radius2 = cat2.fluxfrac_radius(0.5)
        radius3 = cat3.fluxfrac_radius(0.5)
        with pytest.raises(AssertionError):
            assert_equal(radius2, radius3)
        with pytest.raises(AssertionError):
            assert_equal(radius1, radius2)

        cat4 = cat3[0:1]
        assert len(cat4.kron_radius) == 1

    def test_minimal_catalog(self):
        cat = SourceCatalog(self.data, self.segm)
        obj = cat[4]
        props = ('background', 'background_ma', 'error', 'error_ma')
        for prop in props:
            assert getattr(obj, prop) is None

        arr_props = ('_background_cutouts', '_error_cutouts')
        for prop in arr_props:
            assert getattr(obj, prop)[0] is None

        props = ('background_mean', 'background_sum', 'background_centroid',
                 'segment_fluxerr', 'kron_fluxerr')
        for prop in props:
            assert np.isnan(getattr(obj, prop))

        assert obj.local_background_aperture is None
        assert obj.local_background == 0.0

    def test_slicing(self):
        self.cat.to_table()  # evaluate and cache several properties

        obj1 = self.cat[0]
        assert obj1.nlabels == 1
        obj1b = self.cat.get_label(1)
        assert obj1b.nlabels == 1

        obj2 = self.cat[0:1]
        assert obj2.nlabels == 1
        assert len(obj2) == 1

        obj3 = self.cat[0:3]
        obj3b = self.cat.get_labels((1, 2, 3))
        assert_equal(obj3.label, obj3b.label)
        obj4 = self.cat[[0, 1, 2]]
        assert obj3.nlabels == 3
        assert obj3b.nlabels == 3
        assert obj4.nlabels == 3
        assert len(obj3) == 3
        assert len(obj4) == 3

        obj5 = self.cat[[3, 2, 1]]
        labels = [4, 3, 2]
        obj5b = self.cat.get_labels(labels)
        assert_equal(obj5.label, obj5b.label)
        assert obj5.nlabels == 3
        assert len(obj5) == 3
        assert_equal(obj5.label, labels)

        # test get_labels when labels are not sorted
        obj5 = self.cat[[3, 2, 1]]
        labels2 = (3, 4)
        obj5b = obj5.get_labels(labels2)
        assert_equal(obj5b.label, labels2)

        obj6 = obj5[0]
        assert obj6.label == labels[0]

        mask = self.cat.label > 3
        obj7 = self.cat[mask]
        assert obj7.nlabels == 4
        assert len(obj7) == 4

        with pytest.raises(TypeError):
            obj1 = self.cat[0]
            obj2 = obj1[0]

        match = 'is invalid'
        with pytest.raises(ValueError, match=match):
            self.cat.get_label(1000)

        with pytest.raises(ValueError, match=match):
            self.cat.get_labels([1, 2, 1000])

    def test_iter(self):
        labels = []
        for obj in self.cat:
            labels.append(obj.label)
        assert len(labels) == len(self.cat)

    def test_table(self):
        columns = ['label', 'xcentroid', 'ycentroid']
        tbl = self.cat.to_table(columns=columns)
        assert len(tbl) == 7
        assert tbl.colnames == columns

    def test_invalid_inputs(self):
        segm = SegmentationImage(np.zeros(self.data.shape, dtype=int))
        with pytest.raises(ValueError):
            SourceCatalog(self.data, segm)

        # test 1D arrays
        img1d = np.arange(4)
        segm = SegmentationImage(img1d)
        with pytest.raises(ValueError):
            SourceCatalog(img1d, segm)

        wrong_shape = np.ones((3, 3), dtype=int)
        with pytest.raises(ValueError):
            SourceCatalog(wrong_shape, self.segm)

        with pytest.raises(ValueError):
            SourceCatalog(self.data, self.segm, error=wrong_shape)

        with pytest.raises(ValueError):
            SourceCatalog(self.data, self.segm, background=wrong_shape)

        with pytest.raises(ValueError):
            SourceCatalog(self.data, self.segm, mask=wrong_shape)

        with pytest.raises(ValueError):
            segm = SegmentationImage(wrong_shape)
            SourceCatalog(self.data, segm)

        with pytest.raises(TypeError):
            SourceCatalog(self.data, wrong_shape)

        with pytest.raises(TypeError):
            obj = SourceCatalog(self.data, self.segm)[0]
            len(obj)

        with pytest.raises(ValueError):
            SourceCatalog(self.data, self.segm, localbkg_width=-1)
        with pytest.raises(ValueError):
            SourceCatalog(self.data, self.segm, localbkg_width=3.4)
        with pytest.raises(ValueError):
            apermask_method = 'invalid'
            SourceCatalog(self.data, self.segm,
                          apermask_method=apermask_method)
        with pytest.raises(ValueError):
            kron_params = (0.0, 1.0)
            SourceCatalog(self.data, self.segm, kron_params=kron_params)
        with pytest.raises(ValueError):
            kron_params = (2.5, 0.0)
            SourceCatalog(self.data, self.segm, kron_params=kron_params)
        with pytest.raises(ValueError):
            kron_params = (-2.5, 0.0)
            SourceCatalog(self.data, self.segm, kron_params=kron_params)
        with pytest.raises(ValueError):
            kron_params = (2.5, -4.0)
            SourceCatalog(self.data, self.segm, kron_params=kron_params)
        with pytest.raises(ValueError):
            kron_params = (2.5, 1.4, -2.0)
            SourceCatalog(self.data, self.segm, kron_params=kron_params)

    def test_invalid_units(self):
        unit = u.uJy
        wrong_unit = u.km

        with pytest.raises(ValueError):
            SourceCatalog(self.data << unit, self.segm,
                          error=self.error << wrong_unit)

        with pytest.raises(ValueError):
            SourceCatalog(self.data << unit, self.segm,
                          background=self.background << wrong_unit)

        # all array inputs must have the same unit
        with pytest.raises(ValueError):
            SourceCatalog(self.data << unit, self.segm, error=self.error)

        with pytest.raises(ValueError):
            SourceCatalog(self.data, self.segm,
                          background=self.background << unit)

    def test_wcs(self):
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
        cat = SourceCatalog(self.data, self.segm)
        assert len(cat) == 7
        tbl = cat.to_table()
        assert isinstance(tbl, QTable)
        assert len(tbl) == 7
        obj = cat[0]
        assert obj.nlabels == 1
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

        props = ('xcentroid', 'ycentroid', 'area', 'orientation',
                 'segment_flux', 'segment_fluxerr', 'kron_flux',
                 'kron_fluxerr', 'background_mean')
        obj = cat[0]
        for prop in props:
            assert np.isnan(getattr(obj, prop))
        objs = cat[1:]
        for prop in props:
            assert np.all(np.isfinite(getattr(objs, prop)))

        # test that mask=None is the same as mask=np.ma.nomask
        cat1 = SourceCatalog(data, self.segm, mask=None)
        cat2 = SourceCatalog(data, self.segm, mask=np.ma.nomask)
        assert cat1[0].xcentroid == cat2[0].xcentroid

    def test_repr_str(self):
        cat = SourceCatalog(self.data, self.segm)
        assert repr(cat) == str(cat)

        lines = ('Length: 7', 'labels: [1 2 3 4 5 6 7]')
        for line in lines:
            assert line in repr(cat)

    def test_detection_cat(self):
        data2 = self.data - 5
        cat1 = SourceCatalog(data2, self.segm)
        cat2 = SourceCatalog(data2, self.segm, detection_cat=self.cat)
        assert len(cat2.kron_aperture) == len(cat2)
        assert not np.array_equal(cat1.kron_radius, cat2.kron_radius)
        assert not np.array_equal(cat1.kron_flux, cat2.kron_flux)
        assert_allclose(cat2.kron_radius, self.cat.kron_radius)
        assert not np.array_equal(cat2.kron_flux, self.cat.kron_flux)

        with pytest.raises(TypeError):
            SourceCatalog(data2, self.segm, detection_cat=np.arange(4))

        with pytest.raises(ValueError):
            segm = self.segm.copy()
            segm.remove_labels((6, 7))
            cat = SourceCatalog(self.data, segm)
            SourceCatalog(self.data, self.segm, detection_cat=cat)

    def test_kron_minradius(self):
        kron_params = (2.5, 2.5)
        cat = SourceCatalog(self.data, self.segm, mask=self.mask,
                            apermask_method='none', kron_params=kron_params)
        assert cat.kron_aperture[0] is None
        assert np.isnan(cat.kron_radius[0])
        kronrad = cat.kron_radius.value
        kronrad = kronrad[~np.isnan(kronrad)]
        assert np.min(kronrad) == kron_params[1]
        assert isinstance(cat.kron_aperture[2], EllipticalAperture)
        assert isinstance(cat.kron_aperture[4], EllipticalAperture)
        assert isinstance(cat.kron_params, tuple)

    def test_kron_masking(self):
        apermask_method = 'none'
        cat1 = SourceCatalog(self.data, self.segm,
                             apermask_method=apermask_method)
        apermask_method = 'mask'
        cat2 = SourceCatalog(self.data, self.segm,
                             apermask_method=apermask_method)
        apermask_method = 'correct'
        cat3 = SourceCatalog(self.data, self.segm,
                             apermask_method=apermask_method)
        idx = 2  # source with close neighbors
        assert cat1[idx].kron_flux > cat2[idx].kron_flux
        assert cat3[idx].kron_flux > cat2[idx].kron_flux
        assert cat1[idx].kron_flux > cat3[idx].kron_flux

    def test_kron_negative(self):
        cat = SourceCatalog(self.data - 10, self.segm)
        assert_allclose(cat.kron_radius.value, cat.kron_params[1])

    def test_kron_photometry(self):
        flux0, fluxerr0 = self.cat.kron_photometry((2.5, 1.4))
        assert_allclose(flux0, self.cat.kron_flux)
        assert_allclose(fluxerr0, self.cat.kron_fluxerr)

        flux1, fluxerr1 = self.cat.kron_photometry((1.0, 1.4), name='kron1')
        flux2, fluxerr2 = self.cat.kron_photometry((2.0, 1.4), name='kron2')
        assert_allclose(flux1, self.cat.kron1_flux)
        assert_allclose(fluxerr1, self.cat.kron1_fluxerr)
        assert_allclose(flux2, self.cat.kron2_flux)
        assert_allclose(fluxerr2, self.cat.kron2_fluxerr)

        assert np.all((flux2 > flux1) | (np.isnan(flux2) & np.isnan(flux1)))
        assert np.all((fluxerr2 > fluxerr1)
                      | (np.isnan(fluxerr2) & np.isnan(fluxerr1)))

        # test different min Kron radius
        flux3, fluxerr3 = self.cat.kron_photometry((2.5, 2.5))
        assert np.all((flux3 > flux0) | (np.isnan(flux3) & np.isnan(flux0)))
        assert np.all((fluxerr3 > fluxerr0)
                      | (np.isnan(fluxerr3) & np.isnan(fluxerr0)))

        obj = self.cat[1]
        flux1, fluxerr1 = obj.kron_photometry((1.0, 1.4), name='kron0')
        assert np.isscalar(flux1)
        assert np.isscalar(fluxerr1)
        assert_allclose(flux1, obj.kron0_flux)
        assert_allclose(fluxerr1, obj.kron0_fluxerr)

        cat = SourceCatalog(self.data, self.segm)
        _, fluxerr = cat.kron_photometry((2.0, 1.4))
        assert np.all(np.isnan(fluxerr))

        with pytest.raises(ValueError):
            self.cat.kron_photometry(2.5)
        with pytest.raises(ValueError):
            self.cat.kron_photometry((2.5, 0.0))
        with pytest.raises(ValueError):
            self.cat.kron_photometry((0.0, 1.4))
        with pytest.raises(ValueError):
            self.cat.kron_photometry((2.5, 0.0, 1.5))

    def test_circular_photometry(self):
        flux1, fluxerr1 = self.cat.circular_photometry(1.0, name='circ1')
        flux2, fluxerr2 = self.cat.circular_photometry(5.0, name='circ5')
        assert_allclose(flux1, self.cat.circ1_flux)
        assert_allclose(fluxerr1, self.cat.circ1_fluxerr)
        assert_allclose(flux2, self.cat.circ5_flux)
        assert_allclose(fluxerr2, self.cat.circ5_fluxerr)

        assert np.all((flux2 > flux1) | (np.isnan(flux2) & np.isnan(flux1)))
        assert np.all((fluxerr2 > fluxerr1)
                      | (np.isnan(fluxerr2) & np.isnan(fluxerr1)))

        obj = self.cat[1]
        assert obj.isscalar
        flux1, fluxerr1 = obj.circular_photometry(1.0, name='circ0')
        assert np.isscalar(flux1)
        assert np.isscalar(fluxerr1)
        assert_allclose(flux1, obj.circ0_flux)
        assert_allclose(fluxerr1, obj.circ0_fluxerr)

        cat = SourceCatalog(self.data, self.segm)
        _, fluxerr = cat.circular_photometry(1.0)
        assert np.all(np.isnan(fluxerr))

        # with "center" mode, tiny apertures that do not overlap any
        # center should return NaN
        cat2 = self.cat.copy()
        cat2._set_semode()  # sets "center" mode
        flux1, fluxerr1 = cat2.circular_photometry(0.1)
        assert np.all(np.isnan(flux1[2:4]))
        assert np.all(np.isnan(fluxerr1[2:4]))

        with pytest.raises(ValueError):
            self.cat.circular_photometry(0.0)
        with pytest.raises(ValueError):
            self.cat.circular_photometry(-1.0)
        with pytest.raises(ValueError):
            self.cat.make_circular_apertures(0.0)
        with pytest.raises(ValueError):
            self.cat.make_circular_apertures(-1.0)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plots(self):
        from matplotlib.patches import Patch

        patches = self.cat.plot_circular_apertures(5.0)
        assert isinstance(patches, list)
        for patch in patches:
            assert isinstance(patch, Patch)

        patches = self.cat.plot_kron_apertures()
        assert isinstance(patches, list)
        for patch in patches:
            assert isinstance(patch, Patch)

        patches2 = self.cat.plot_kron_apertures((2.0, 1.2))
        assert isinstance(patches2, list)
        for patch in patches2:
            assert isinstance(patch, Patch)

        # test scalar
        obj = self.cat[1]
        patch1 = obj.plot_kron_apertures()
        assert isinstance(patch1, Patch)
        patch2 = obj.plot_kron_apertures((2.0, 1.2))
        assert isinstance(patch2, Patch)

    def test_fluxfrac_radius(self):
        radius1 = self.cat.fluxfrac_radius(0.1, name='fluxfrac_r1')
        radius2 = self.cat.fluxfrac_radius(0.5, name='fluxfrac_r5')
        assert_allclose(radius1, self.cat.fluxfrac_r1)
        assert_allclose(radius2, self.cat.fluxfrac_r5)
        assert np.all((radius2 > radius1)
                      | (np.isnan(radius2) & np.isnan(radius1)))

        cat = SourceCatalog(self.data, self.segm)
        obj = cat[1]
        radius = obj.fluxfrac_radius(0.5)
        assert radius.isscalar  # Quantity radius - can't use np.isscalar
        assert_allclose(radius.value, 7.899648)

        with pytest.raises(ValueError):
            radius = self.cat.fluxfrac_radius(0)
        with pytest.raises(ValueError):
            radius = self.cat.fluxfrac_radius(-1)

        cat = SourceCatalog(self.data - 50.0, self.segm, error=self.error,
                            background=self.background, mask=self.mask,
                            wcs=self.wcs, localbkg_width=24)
        radius_hl = cat.fluxfrac_radius(0.5)
        assert np.isnan(radius_hl[0])

    def test_cutout_units(self):
        obj = self.cat_units[0]
        quantities = (obj.data, obj.error, obj.background)
        ndarray = (obj.segment, obj.segment_ma, obj.data_ma, obj.error_ma,
                   obj.background_ma)
        for arr in quantities:
            assert isinstance(arr, u.Quantity)
        for arr in ndarray:
            assert not isinstance(arr, u.Quantity)

    @pytest.mark.parametrize('scalar', (True, False))
    def test_extra_properties(self, scalar):
        cat = SourceCatalog(self.data, self.segm)
        if scalar:
            cat = cat[1]

        segment_snr = cat.segment_flux / cat.segment_fluxerr

        with pytest.raises(ValueError):
            # built-in attribute
            cat.add_extra_property('_data', segment_snr)
        with pytest.raises(ValueError):
            # built-in property
            cat.add_extra_property('label', segment_snr)
        with pytest.raises(ValueError):
            # built-in lazyproperty
            cat.add_extra_property('area', segment_snr)

        cat.add_extra_property('segment_snr', segment_snr)

        with pytest.raises(ValueError):
            # already exists
            cat.add_extra_property('segment_snr', segment_snr)

        cat.add_extra_property('segment_snr', 2.0 * segment_snr,
                               overwrite=True)
        assert len(cat.extra_properties) == 1
        assert_equal(cat.segment_snr, 2.0 * segment_snr)

        with pytest.raises(ValueError):
            cat.remove_extra_property('invalid')

        cat.remove_extra_property(cat.extra_properties)
        assert len(cat.extra_properties) == 0

        cat.add_extra_property('segment_snr', segment_snr)
        cat.add_extra_property('segment_snr2', segment_snr)
        cat.add_extra_property('segment_snr3', segment_snr)
        assert len(cat.extra_properties) == 3

        cat.remove_extra_properties(cat.extra_properties)
        assert len(cat.extra_properties) == 0

        cat.add_extra_property('segment_snr', segment_snr)
        new_name = 'segment_snr0'
        cat.rename_extra_property('segment_snr', new_name)
        assert new_name in cat.extra_properties

        # key in extra_properties, but not a defined attribute
        cat._extra_properties.append('invalid')
        with pytest.raises(ValueError):
            cat.add_extra_property('invalid', segment_snr)
        cat._extra_properties.remove('invalid')

        assert cat._has_len([1, 2, 3])
        assert not cat._has_len('test_string')

    def test_extra_properties_invalid(self):
        cat = SourceCatalog(self.data, self.segm)
        with pytest.raises(ValueError):
            cat.add_extra_property('invalid', 1.0)
        with pytest.raises(ValueError):
            cat.add_extra_property('invalid', (1.0, 2.0))

        obj = cat[1]
        with pytest.raises(ValueError):
            obj.add_extra_property('invalid', (1.0, 2.0))
        with pytest.raises(ValueError):
            val = np.arange(2) << u.km
            obj.add_extra_property('invalid', val)
        with pytest.raises(ValueError):
            coord = SkyCoord([42, 43], [44, 45], unit='deg')
            obj.add_extra_property('invalid', coord)

    def test_properties(self):
        attrs = ('label', 'labels', 'slices', 'xcentroid',
                 'segment_flux', 'kron_flux')
        for attr in attrs:
            assert attr in self.cat.properties

    def test_copy(self):
        cat = SourceCatalog(self.data, self.segm)
        cat2 = cat.copy()
        _ = cat.kron_flux
        assert 'kron_flux' not in cat2.__dict__
        tbl = cat2.to_table()
        assert len(tbl) == 7

    def test_data_dtype(self):
        """
        Regression test that input ``data`` with int dtype does not
        raise UFuncTypeError due to subtraction of float array from int
        array.
        """
        data = np.zeros((25, 25), dtype=np.uint16)
        data[8:16, 8:16] = 10
        segmdata = np.zeros((25, 25), dtype=int)
        segmdata[8:16, 8:16] = 1
        segm = SegmentationImage(segmdata)
        cat = SourceCatalog(data, segm, localbkg_width=3)
        assert cat.min_value == 10
        assert cat.max_value == 10

    def test_make_circular_apertures(self):
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
        aper = self.cat.make_kron_apertures()
        assert len(aper) == len(self.cat)
        assert isinstance(aper[1], EllipticalAperture)

        aper2 = self.cat.make_kron_apertures((2.0, 1.4))
        assert len(aper2) == len(self.cat)

        obj = self.cat[1]
        aper = obj.make_kron_apertures()
        assert isinstance(aper, EllipticalAperture)

    @pytest.mark.skipif(not HAS_SKIMAGE, reason='skimage is required')
    def test_make_cutouts(self):
        data = make_100gaussians_image()
        bkg_estimator = MedianBackground()
        bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                           bkg_estimator=bkg_estimator)
        data -= bkg.background  # subtract the background
        threshold = 1.5 * bkg.background_rms
        kernel = make_2dgaussian_kernel(3.0, size=5)
        convolved_data = convolve(data, kernel)
        npixels = 10
        finder = SourceFinder(npixels=npixels, progress_bar=False)
        segment_map = finder(convolved_data, threshold)
        cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)

        shape = (100, 100)

        with pytest.raises(ValueError):
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
        assert cutouts[1].data.shape == shape

        obj = cat[1]
        cut = obj.make_cutouts(shape)
        assert isinstance(cut, CutoutImage)
        assert cut.data.shape == shape

        cutouts = cat.make_cutouts(shape, mode='partial', fill_value=-100)
        assert cutouts[0].data[0, 0] == -100

        # cutout will be None if source is completely masked
        cutouts = self.cat.make_cutouts(shape)
        assert cutouts[0] is None

    def test_meta(self):
        meta = self.cat.meta
        attrs = ['localbkg_width', 'apermask_method', 'kron_params']
        for attr in attrs:
            assert attr in meta

        tbl = self.cat.to_table()
        assert tbl.meta == self.cat.meta

        out = StringIO()
        tbl.write(out, format='ascii.ecsv')
        tbl2 = QTable.read(out.getvalue(), format='ascii.ecsv')
        # check order of meta keys
        assert list(tbl2.meta.keys()) == list(tbl.meta.keys())

    def test_semode(self):
        self.cat._set_semode()
        tbl = self.cat.to_table()
        assert len(tbl) == 7


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
@pytest.mark.skipif(not HAS_SKIMAGE, reason='skimage is required')
def test_kron_params():
    data = make_100gaussians_image()
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                       bkg_estimator=bkg_estimator)
    data -= bkg.background  # subtract the background

    threshold = 1.5 * bkg.background_rms

    kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data = convolve(data, kernel)

    npixels = 10
    finder = SourceFinder(npixels=npixels, progress_bar=False)
    segm = finder(convolved_data, threshold)

    minrad = 1.4
    kron_params = (2.5, minrad, 0.0)
    cat = SourceCatalog(data, segm, convolved_data=convolved_data,
                        kron_params=kron_params)
    assert cat.kron_radius.value.min() == minrad
    assert_allclose(cat.kron_flux.min(), 246.91339096556538)
    rh = cat.fluxfrac_radius(0.5)
    assert_allclose(rh.value.min(), 1.3925520107605818)

    minrad = 1.2
    kron_params = (2.5, minrad, 0.0)
    cat = SourceCatalog(data, segm, convolved_data=convolved_data,
                        kron_params=kron_params)
    assert cat.kron_radius.value.min() == minrad
    assert_allclose(cat.kron_flux.min(), 246.91339096556538)
    rh = cat.fluxfrac_radius(0.5)
    assert_allclose(rh.value.min(), 1.3979610808075127)

    minrad = 0.2
    kron_params = (2.5, minrad, 0.0)
    cat = SourceCatalog(data, segm, convolved_data=convolved_data,
                        kron_params=kron_params)
    assert_allclose(cat.kron_radius.value.min(), 0.5712931578847312)
    assert_allclose(cat.kron_flux.min(), 246.91339096556538)
    rh = cat.fluxfrac_radius(0.5)
    assert_allclose(rh.value.min(), 1.34218684691197)

    kron_params = (2.5, 1.4, 7.0)
    cat = SourceCatalog(data, segm, convolved_data=convolved_data,
                        kron_params=kron_params)
    assert cat.kron_radius.value.min() == 0.0
    assert_allclose(cat.kron_flux.min(), 246.91339096556538)
    rh = cat.fluxfrac_radius(0.5)
    assert_allclose(rh.value.min(), 1.3649418211298536)
    assert isinstance(cat.kron_aperture[0], CircularAperture)


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
@pytest.mark.skipif(not HAS_SKIMAGE, reason='skimage is required')
def test_centroid_win():
    g1 = Gaussian2D(1621, 6.29, 10.95, 1.55, 1.29, 0.296706)
    g2 = Gaussian2D(3596, 13.81, 8.29, 1.44, 1.27, 0.628319)
    m = g1 + g2
    yy, xx = np.mgrid[0:21, 0:21]
    data = m(xx, yy)
    noise = make_noise_image(data.shape, mean=0, stddev=65.0, seed=123)
    data += noise

    kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data = convolve(data, kernel)
    npixels = 10
    finder = SourceFinder(npixels=npixels, progress_bar=False)
    threshold = 107.9
    segment_map = finder(convolved_data, threshold)
    cat = SourceCatalog(data, segment_map, convolved_data=convolved_data,
                        apermask_method='none')

    assert cat.xcentroid[0] != cat.xcentroid_win[0]
    assert cat.ycentroid[0] != cat.ycentroid_win[0]
    # centroid_win moved beyond 1-sigma ellipse and was reset to
    # isophotal centroid
    assert cat.xcentroid[1] == cat.xcentroid_win[1]
    assert cat.ycentroid[1] == cat.ycentroid_win[1]


@pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
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
    segm = detect_sources(data, 98.0, npixels=5)
    cat = SourceCatalog(data, segm)

    indices = (0, 3, 14, 30)
    for idx in indices:
        assert_equal(cat.centroid_win[idx], cat.centroid[idx])
