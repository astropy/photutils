# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the catalog module.
"""

from copy import deepcopy

import astropy.units as u
from astropy.modeling.models import Gaussian2D
from astropy.table import QTable
from numpy.testing import assert_allclose, assert_equal
import numpy as np
import pytest

from ..catalog import SourceCatalog
from ..core import SegmentationImage
from ..detect import detect_sources
from ...aperture import CircularAperture, EllipticalAperture
from ...datasets import make_gwcs, make_wcs, make_noise_image
from ...utils._optional_deps import HAS_GWCS, HAS_SCIPY  # noqa


@pytest.mark.skipif('not HAS_SCIPY')
class TestSourceCatalog:
    def setup_class(self):
        xcen = 51.
        ycen = 52.7
        major_sigma = 8.
        minor_sigma = 3.
        theta = np.pi / 6.
        g1 = Gaussian2D(111., xcen, ycen, major_sigma, minor_sigma,
                        theta=theta)
        g2 = Gaussian2D(50, 20, 80, 5.1, 4.5)
        g3 = Gaussian2D(70, 75, 18, 9.2, 4.5)
        g4 = Gaussian2D(111., 11.1, 12.2, major_sigma, minor_sigma,
                        theta=theta)
        g5 = Gaussian2D(81., 61, 42.7, major_sigma, minor_sigma, theta=theta)
        g6 = Gaussian2D(107., 75, 61, major_sigma, minor_sigma, theta=-theta)
        g7 = Gaussian2D(107., 90, 90, 4, 2, theta=-theta)

        yy, xx = np.mgrid[0:101, 0:101]
        self.data = (g1(xx, yy) + g2(xx, yy) + g3(xx, yy) + g4(xx, yy)
                     + g5(xx, yy) + g6(xx, yy) + g7(xx, yy))
        threshold = 27.
        self.segm = detect_sources(self.data, threshold, npixels=5)
        self.error = make_noise_image(self.data.shape, mean=0, stddev=2.,
                                      seed=123)
        self.background = np.ones(self.data.shape) * 5.1
        self.mask = np.zeros(self.data.shape, dtype=bool)
        self.mask[0:30, 0:30] = True

        self.wcs = make_wcs(self.data.shape)
        self.cat = SourceCatalog(self.data, self.segm, error=self.error,
                                 background=self.background, mask=self.mask,
                                 wcs=self.wcs, localbkg_width=24)
        unit = u.nJy
        self.cat_units = SourceCatalog(self.data << unit, self.segm,
                                       error=self.error << unit,
                                       background=self.background << unit,
                                       mask=self.mask, wcs=self.wcs,
                                       localbkg_width=24)

    @pytest.mark.parametrize('with_units', (True, False))
    def test_catalog(self, with_units):
        props1 = ('background_centroid', 'background_mean', 'background_sum',
                  'bbox', 'covar_sigx2', 'covar_sigxy', 'covar_sigy2', 'cxx',
                  'cxy', 'cyy', 'ellipticity', 'elongation', 'fwhm',
                  'equivalent_radius', 'gini', 'kron_radius', 'maxval_xindex',
                  'maxval_yindex', 'minval_xindex', 'minval_yindex',
                  'perimeter', 'sky_bbox_ll', 'sky_bbox_lr', 'sky_bbox_ul',
                  'sky_bbox_ur', 'sky_centroid_icrs')

        props2 = ('centroid', 'covariance', 'covariance_eigvals',
                  'cutout_centroid', 'cutout_maxval_index',
                  'cutout_minval_index', 'inertia_tensor', 'maxval_index',
                  'minval_index', 'moments', 'moments_central', 'background',
                  'background_ma', 'convdata', 'convdata_ma', 'data',
                  'data_ma', 'error', 'error_ma', 'segment', 'segment_ma')

        props = tuple(self.cat.default_columns) + props1 + props2

        if with_units:
            cat1 = deepcopy(self.cat_units)
            cat2 = deepcopy(self.cat_units)
        else:
            cat1 = deepcopy(self.cat)
            cat2 = deepcopy(self.cat)

        idx = 1

        # evaluate (cache) catalog properties before slice
        obj = cat1[idx]
        for prop in props:
            assert_equal(getattr(cat1, prop)[idx], getattr(obj, prop))

        # slice catalog before evaluating catalog properties
        obj = cat2[idx]
        for prop in props:
            assert_equal(getattr(obj, prop), getattr(cat2, prop)[idx])

    def test_minimal_catalog(self):
        cat = SourceCatalog(self.data, self.segm)
        obj = cat[4]
        props = ('background', 'background_ma', 'error', 'error_ma')
        for prop in props:
            assert getattr(obj, prop) is None

        props = ('background_mean', 'background_sum', 'background_centroid',
                 'segment_fluxerr', 'kron_fluxerr')
        for prop in props:
            assert np.isnan(getattr(obj, prop))

        assert obj.local_background_aperture is None
        assert obj.local_background == 0.

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

        obj6 = obj5[0]
        assert obj6.label == labels[0]

        mask = self.cat.label > 3
        obj7 = self.cat[mask]
        assert obj7.nlabels == 4
        assert len(obj7) == 4

        with pytest.raises(TypeError):
            obj1 = self.cat[0]
            obj2 = obj1[0]

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
        # test 1D arrays
        img1d = np.arange(4)
        segm = SegmentationImage(img1d)
        with pytest.raises(ValueError):
            SourceCatalog(img1d, segm)

        wrong_shape = np.ones((3, 3))
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
            kron_params = (2.5, 0.0, 3.0)
            SourceCatalog(self.data, self.segm, kron_params=kron_params)
        with pytest.raises(ValueError):
            kron_params = (-2.5, 0.0)
            SourceCatalog(self.data, self.segm, kron_params=kron_params)
        with pytest.raises(ValueError):
            kron_params = (2.5, -4.0)
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
        assert obj.sky_bbox_ll is not None
        assert obj.sky_bbox_ul is not None
        assert obj.sky_bbox_lr is not None
        assert obj.sky_bbox_ur is not None

    @pytest.mark.skipif('not HAS_GWCS')
    def test_gwcs(self):
        mywcs = make_gwcs(self.data.shape)
        cat = SourceCatalog(self.data, self.segm, wcs=mywcs)
        obj = cat[1]
        assert obj.sky_centroid is not None
        assert obj.sky_centroid_icrs is not None
        assert obj.sky_bbox_ll is not None
        assert obj.sky_bbox_ul is not None
        assert obj.sky_bbox_lr is not None
        assert obj.sky_bbox_ur is not None

    def test_nowcs(self):
        cat = SourceCatalog(self.data, self.segm, wcs=None)
        obj = cat[2]
        assert obj.sky_centroid is None
        assert obj.sky_centroid_icrs is None
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

    def test_kernel(self):
        kernel = np.array([[1., 2, 1], [2, 4, 2], [1, 2, 100]])
        kernel /= kernel.sum()
        cat1 = SourceCatalog(self.data, self.segm, kernel=None)
        cat2 = SourceCatalog(self.data, self.segm, kernel=kernel)
        assert not np.array_equal(cat1.xcentroid, cat2.xcentroid)
        assert not np.array_equal(cat1.ycentroid, cat2.ycentroid)

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
            segm = deepcopy(self.segm)
            segm.remove_labels((6, 7))
            cat = SourceCatalog(self.data, segm)
            SourceCatalog(self.data, self.segm, detection_cat=cat)

    def test_kron_minradius(self):
        kron_params = (2.5, 10.0)
        cat = SourceCatalog(self.data, self.segm, mask=self.mask,
                            apermask_method='none', kron_params=kron_params)
        assert cat.kron_aperture[0] is None
        assert isinstance(cat.kron_aperture[2], EllipticalAperture)
        assert isinstance(cat.kron_aperture[4], CircularAperture)

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
        assert np.all(np.isnan(cat.kron_radius.value))
        assert np.all(np.isnan(cat.kron_flux))

    def test_circular_photometry(self):
        flux1, fluxerr1 = self.cat.circular_photometry(1.0)
        flux2, fluxerr2 = self.cat.circular_photometry(5.0)
        assert np.all((flux2 > flux1) | (np.isnan(flux2) & np.isnan(flux1)))
        assert np.all((fluxerr2 > fluxerr1)
                      | (np.isnan(fluxerr2) & np.isnan(fluxerr1)))

        cat = SourceCatalog(self.data, self.segm)
        _, fluxerr = cat.circular_photometry(1.0)
        assert np.all(np.isnan(fluxerr))

    def test_fluxfrac_radius(self):
        radius1 = self.cat.fluxfrac_radius(0.1)
        radius2 = self.cat.fluxfrac_radius(0.5)
        assert np.all((radius2 > radius1)
                      | (np.isnan(radius2) & np.isnan(radius1)))

        cat = SourceCatalog(self.data, self.segm)
        obj = cat[1]
        radius = obj.fluxfrac_radius(0.5)
        assert_allclose(radius, 7.899648)

        with pytest.raises(ValueError):
            radius = self.cat.fluxfrac_radius(0)
        with pytest.raises(ValueError):
            radius = self.cat.fluxfrac_radius(-1)

        cat = SourceCatalog(self.data - 50., self.segm, error=self.error,
                            background=self.background, mask=self.mask,
                            wcs=self.wcs, localbkg_width=24)
        radius_hl = cat.fluxfrac_radius(0.5)
        assert np.all(np.isnan(radius_hl))

    def test_cutout_units(self):
        obj = self.cat_units[0]
        quantities = (obj.data, obj.error, obj.background)
        ndarray = (obj.segment, obj.segment_ma, obj.data_ma, obj.error_ma,
                   obj.background_ma)
        for arr in quantities:
            assert isinstance(arr, u.Quantity)
        for arr in ndarray:
            assert not isinstance(arr, u.Quantity)
