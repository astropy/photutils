# Licensed under a 3-clause BSD style license - see LICENSE.rst

import itertools

import numpy as np
from numpy.testing import assert_allclose
import pytest

from astropy.tests.helper import assert_quantity_allclose
from astropy.modeling import models
from astropy.table import QTable
import astropy.units as u
from astropy.utils.misc import isiterable
import astropy.wcs as WCS

from ..properties import (SegmentationImage, SourceProperties,
                          source_properties, SourceCatalog)

try:
    import scipy    # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import skimage    # noqa
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


XCEN = 51.
YCEN = 52.7
MAJOR_SIG = 8.
MINOR_SIG = 3.
THETA = np.pi / 6.
g = models.Gaussian2D(111., XCEN, YCEN, MAJOR_SIG, MINOR_SIG, theta=THETA)
y, x = np.mgrid[0:100, 0:100]
IMAGE = g(x, y)
THRESHOLD = 0.1
SEGM = (IMAGE >= THRESHOLD).astype(np.int)

ERR_VALS = [0., 2.5]
BACKGRD_VALS = [None, 0., 1., 3.5]


@pytest.mark.skipif('not HAS_SKIMAGE')
@pytest.mark.skipif('not HAS_SCIPY')
class TestSourceProperties:
    def test_segment_shape(self):
        with pytest.raises(ValueError):
            SourceProperties(IMAGE, np.zeros((2, 2)), label=1)

    @pytest.mark.parametrize('label', (0, -1))
    def test_label_invalid(self, label):
        with pytest.raises(ValueError):
            SourceProperties(IMAGE, SEGM, label=label)

    @pytest.mark.parametrize('label', (0, -1))
    def test_label_missing(self, label):
        segm = SEGM.copy()
        segm[0:2, 0:2] = 3   # skip label 2
        with pytest.raises(ValueError):
            SourceProperties(IMAGE, segm, label=2)

    def test_wcs(self):
        mywcs = WCS.WCS(naxis=2)
        rho = np.pi / 3.
        scale = 0.1 / 3600.
        mywcs.wcs.cd = [[scale*np.cos(rho), -scale*np.sin(rho)],
                        [scale*np.sin(rho), scale*np.cos(rho)]]
        mywcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        props = SourceProperties(IMAGE, SEGM, wcs=mywcs, label=1)
        assert props.sky_centroid_icrs is not None
        assert props.sky_bbox_ll is not None
        assert props.sky_bbox_ul is not None
        assert props.sky_bbox_lr is not None
        assert props.sky_bbox_ur is not None

    def test_nowcs(self):
        props = SourceProperties(IMAGE, SEGM, wcs=None, label=1)
        assert props.sky_centroid_icrs is None
        assert props.sky_bbox_ll is None
        assert props.sky_bbox_ul is None
        assert props.sky_bbox_lr is None
        assert props.sky_bbox_ur is None

    def test_to_table(self):
        props = SourceProperties(IMAGE, SEGM, label=1)
        t1 = props.to_table()
        assert isinstance(t1, QTable)
        assert len(t1) == 1
        assert_quantity_allclose(t1['area'], 1058 * u.pix**2)


@pytest.mark.skipif('not HAS_SKIMAGE')
@pytest.mark.skipif('not HAS_SCIPY')
class TestSourcePropertiesFunctionInputs:
    def test_segment_shape(self):
        wrong_shape = np.zeros((2, 2))
        with pytest.raises(ValueError):
            source_properties(IMAGE, wrong_shape)

    def test_error_shape(self):
        wrong_shape = np.zeros((2, 2))
        with pytest.raises(ValueError):
            source_properties(IMAGE, SEGM, error=wrong_shape)

    def test_background_shape(self):
        wrong_shape = np.zeros((2, 2))
        with pytest.raises(ValueError):
            source_properties(IMAGE, SEGM, background=wrong_shape)

    def test_mask_shape(self):
        wrong_shape = np.zeros((2, 2))
        with pytest.raises(ValueError):
            source_properties(IMAGE, SEGM, mask=wrong_shape)

    def test_labels(self):
        props = source_properties(IMAGE, SEGM, labels=1)
        assert props[0].id == 1

    def test_invalidlabels(self):
        props = source_properties(IMAGE, SEGM, labels=-1)
        assert len(props) == 0


@pytest.mark.skipif('not HAS_SKIMAGE')
@pytest.mark.skipif('not HAS_SCIPY')
class TestSourcePropertiesFunction:
    def test_properties(self):
        props = source_properties(IMAGE, SEGM)
        assert props[0].id == 1
        assert_quantity_allclose(props[0].xcentroid, XCEN*u.pix,
                                 rtol=1.e-2)
        assert_quantity_allclose(props[0].ycentroid, YCEN*u.pix,
                                 rtol=1.e-2)
        assert_allclose(props[0].source_sum, IMAGE[IMAGE >= THRESHOLD].sum())
        assert_quantity_allclose(props[0].semimajor_axis_sigma,
                                 MAJOR_SIG*u.pix, rtol=1.e-2)
        assert_quantity_allclose(props[0].semiminor_axis_sigma,
                                 MINOR_SIG*u.pix, rtol=1.e-2)
        assert_quantity_allclose(props[0].orientation, THETA*u.rad,
                                 rtol=1.e-3)
        assert_allclose(props[0].bbox.value, [35, 25, 70, 77])
        assert_quantity_allclose(props[0].area, 1058.0*u.pix**2)
        assert_allclose(len(props[0].values), props[0].area.value)
        assert_allclose(len(props[0].coords), 2)
        assert_allclose(len(props[0].coords[0]), props[0].area.value)

        properties = ['background_at_centroid', 'background_mean',
                      'eccentricity', 'ellipticity', 'elongation',
                      'equivalent_radius', 'max_value', 'maxval_xpos',
                      'maxval_ypos', 'min_value', 'minval_xpos',
                      'minval_ypos', 'perimeter', 'cxx', 'cxy', 'cyy',
                      'covar_sigx2', 'covar_sigxy', 'covar_sigy2', 'xmax',
                      'xmin', 'ymax', 'ymin']
        for propname in properties:
            assert not isiterable(getattr(props[0], propname))

        properties = ['centroid', 'covariance_eigvals', 'cutout_centroid',
                      'maxval_cutout_pos', 'minval_cutout_pos']
        shapes = [getattr(props[0], p).shape for p in properties]
        for shape in shapes:
            assert shape == (2,)

        properties = ['covariance', 'inertia_tensor']
        shapes = [getattr(props[0], p).shape for p in properties]
        for shape in shapes:
            assert shape == (2, 2)

        properties = ['moments', 'moments_central']
        shapes = [getattr(props[0], p).shape for p in properties]
        for shape in shapes:
            assert shape == (4, 4)

    def test_properties_background_notnone(self):
        value = 1.
        props = source_properties(IMAGE, SEGM, background=value)
        assert props[0].background_mean == value
        assert_allclose(props[0].background_at_centroid, value)

    def test_properties_background_units(self):
        value = 1. * u.uJy
        props = source_properties(IMAGE, SEGM, background=value)
        assert props[0].background_mean == value
        assert_allclose(props[0].background_at_centroid, value)

    def test_properties_error_background_none(self):
        props = source_properties(IMAGE, SEGM)
        assert props[0].background_cutout_ma is None
        assert props[0].error_cutout_ma is None

    def test_cutout_shapes(self):
        error = np.ones_like(IMAGE) * 1.
        props = source_properties(IMAGE, SEGM, error=error, background=1.)
        bbox = props[0].bbox.value
        true_shape = (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1)
        properties = ['background_cutout_ma', 'data_cutout',
                      'data_cutout_ma', 'error_cutout_ma']
        shapes = [getattr(props[0], p).shape for p in properties]
        for shape in shapes:
            assert shape == true_shape

    def test_make_cutout(self):
        props = source_properties(IMAGE, SEGM)
        data = np.ones((2, 2))
        with pytest.raises(ValueError):
            props[0].make_cutout(data)

    @pytest.mark.parametrize(('error_value', 'background'),
                             list(itertools.product(ERR_VALS, BACKGRD_VALS)))
    def test_segmentation_inputs(self, error_value, background):
        error = np.ones_like(IMAGE) * error_value
        props = source_properties(IMAGE, SEGM, error=error,
                                  background=background)
        assert_quantity_allclose(props[0].xcentroid, XCEN*u.pix, rtol=1.e-2)
        assert_quantity_allclose(props[0].ycentroid, YCEN*u.pix, rtol=1.e-2)
        assert_quantity_allclose(props[0].semimajor_axis_sigma,
                                 MAJOR_SIG*u.pix, rtol=1.e-2)
        assert_quantity_allclose(props[0].semiminor_axis_sigma,
                                 MINOR_SIG*u.pix, rtol=1.e-2)
        assert_quantity_allclose(props[0].orientation, THETA*u.rad,
                                 rtol=1.e-3)
        assert_allclose(props[0].bbox.value, [35, 25, 70, 77])
        area = props[0].area.value
        assert_allclose(area, 1058.0)

        if background is not None:
            assert_allclose(props[0].background_sum, area * background)
        true_sum = IMAGE[IMAGE >= THRESHOLD].sum()
        assert_allclose(props[0].source_sum, true_sum)

        true_error = np.sqrt(props[0].area.value) * error_value
        assert_allclose(props[0].source_sum_err, true_error)

    def test_data_allzero(self):
        props = source_properties(IMAGE*0., SEGM)
        proplist = ['xcentroid', 'ycentroid', 'semimajor_axis_sigma',
                    'semiminor_axis_sigma', 'eccentricity', 'orientation',
                    'ellipticity', 'elongation', 'cxx', 'cxy', 'cyy']
        for prop in proplist:
            assert np.isnan(getattr(props[0], prop))

    def test_scalar_error(self):
        error_value = 2.5
        error = np.ones_like(IMAGE) * error_value
        props = source_properties(IMAGE, SEGM, error=error_value)
        props2 = source_properties(IMAGE, SEGM, error)
        assert_quantity_allclose(props.source_sum, props2.source_sum)
        assert_quantity_allclose(props.source_sum_err, props2.source_sum_err)

    def test_mask(self):
        data = np.zeros((3, 3))
        data[0, 1] = 1.
        data[1, 1] = 1.
        mask = np.zeros_like(data, dtype=np.bool)
        mask[0, 1] = True
        segm = data.astype(np.int)
        props = source_properties(data, segm, mask=mask)
        assert_allclose(props[0].xcentroid.value, 1)
        assert_allclose(props[0].ycentroid.value, 1)
        assert_allclose(props[0].source_sum, 1)
        assert_allclose(props[0].area.value, 1)

    def test_mask_nomask(self):
        props = source_properties(IMAGE, SEGM, mask=np.ma.nomask)
        mask = np.zeros(IMAGE.shape).astype(bool)
        props2 = source_properties(IMAGE, SEGM, mask=mask)
        assert_allclose(props.xcentroid.value, props2.xcentroid.value)
        assert_allclose(props.ycentroid.value, props2.ycentroid.value)
        assert_quantity_allclose(props.source_sum, props2.source_sum)

    def test_single_pixel_segment(self):
        segm = np.zeros_like(SEGM)
        segm[50, 50] = 1
        props = source_properties(IMAGE, segm)
        assert props[0].eccentricity == 0

    def test_filtering(self):
        from astropy.convolution import Gaussian2DKernel
        FWHM2SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        filter_kernel = Gaussian2DKernel(2.*FWHM2SIGMA, x_size=3, y_size=3)
        error = np.sqrt(IMAGE)
        props1 = source_properties(IMAGE, SEGM, error=error)
        props2 = source_properties(IMAGE, SEGM, error=error,
                                   filter_kernel=filter_kernel.array)
        p1, p2 = props1[0], props2[0]
        keys = ['source_sum', 'source_sum_err']
        for key in keys:
            assert p1[key] == p2[key]
        keys = ['semimajor_axis_sigma', 'semiminor_axis_sigma']
        for key in keys:
            assert p1[key] != p2[key]

    def test_filtering_kernel(self):
        data = np.zeros((3, 3))
        data[1, 1] = 1.
        from astropy.convolution import Gaussian2DKernel
        FWHM2SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        filter_kernel = Gaussian2DKernel(2.*FWHM2SIGMA, x_size=3, y_size=3)
        error = np.sqrt(IMAGE)
        props1 = source_properties(IMAGE, SEGM, error=error)
        props2 = source_properties(IMAGE, SEGM, error=error,
                                   filter_kernel=filter_kernel)
        p1, p2 = props1[0], props2[0]
        keys = ['source_sum', 'source_sum_err']
        for key in keys:
            assert p1[key] == p2[key]
        keys = ['semimajor_axis_sigma', 'semiminor_axis_sigma']
        for key in keys:
            assert p1[key] != p2[key]

    def test_data_nan(self):
        """Test case when data contains NaNs within a segment."""

        data = np.ones((20, 20))
        data[2, 2] = np.nan
        segm = np.zeros((20, 20)).astype(int)
        segm[1:5, 1:5] = 1
        segm[7:15, 7:15] = 2
        segm = SegmentationImage(segm)
        props = source_properties(data, segm)
        assert_quantity_allclose(props.minval_xpos, [1, 7]*u.pix)


@pytest.mark.skipif('not HAS_SKIMAGE')
@pytest.mark.skipif('not HAS_SCIPY')
class TestSourceCatalog:
    def test_basic(self):
        segm = np.zeros(IMAGE.shape)
        x = y = np.arange(0, 100, 10)
        segm[y, x] = np.arange(10)
        cat = source_properties(IMAGE, segm)
        assert len(cat) == 9
        cat2 = cat[0:5]
        assert len(cat2) == 5
        cat3 = SourceCatalog(cat2)
        del cat3[4]
        assert len(cat3) == 4

    def test_inputs(self):
        cat = source_properties(IMAGE, SEGM)
        cat2 = SourceCatalog(cat[0])
        assert len(cat) == 1
        assert len(cat2) == 1

        with pytest.raises(ValueError):
            SourceCatalog('a')

    def test_table(self):
        cat = source_properties(IMAGE, SEGM)
        t = cat.to_table()
        assert isinstance(t, QTable)
        assert len(t) == 1

    def test_table_include(self):
        cat = source_properties(IMAGE, SEGM)
        columns = ['id', 'xcentroid']
        t = cat.to_table(columns=columns)
        assert isinstance(t, QTable)
        assert len(t) == 1
        assert t.colnames == columns

    def test_table_include_invalidname(self):
        cat = source_properties(IMAGE, SEGM)
        columns = ['idzz', 'xcentroidzz']
        with pytest.raises(AttributeError):
            cat.to_table(columns=columns)

    def test_table_exclude(self):
        cat = source_properties(IMAGE, SEGM)
        exclude = ['id', 'xcentroid']
        t = cat.to_table(exclude_columns=exclude)
        assert isinstance(t, QTable)
        assert len(t) == 1
        with pytest.raises(KeyError):
            t['id']

    def test_table_empty_props(self):
        cat = source_properties(IMAGE, SEGM, labels=-1)
        with pytest.raises(ValueError):
            cat.to_table()

    def test_table_wcs(self):
        mywcs = WCS.WCS(naxis=2)
        rho = np.pi / 3.
        scale = 0.1 / 3600.
        mywcs.wcs.cd = [[scale*np.cos(rho), -scale*np.sin(rho)],
                        [scale*np.sin(rho), scale*np.cos(rho)]]
        mywcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']

        cat = source_properties(IMAGE, SEGM, wcs=mywcs)
        columns = ['sky_centroid', 'sky_centroid_icrs', 'icrs_centroid',
                   'ra_icrs_centroid', 'dec_icrs_centroid', 'sky_bbox_ll',
                   'sky_bbox_ul', 'sky_bbox_lr', 'sky_bbox_ur']
        t = cat.to_table(columns=columns)
        assert t[0]['sky_centroid'] is not None
        assert t.colnames == columns

        obj = cat[0]
        row = t[0]
        assert_quantity_allclose(obj.sky_bbox_ll.ra, row['sky_bbox_ll'].ra)
        assert_quantity_allclose(obj.sky_bbox_ll.dec, row['sky_bbox_ll'].dec)

        assert_quantity_allclose(obj.sky_bbox_ul.ra, row['sky_bbox_ul'].ra)
        assert_quantity_allclose(obj.sky_bbox_ul.dec, row['sky_bbox_ul'].dec)

        assert_quantity_allclose(obj.sky_bbox_lr.ra, row['sky_bbox_lr'].ra)
        assert_quantity_allclose(obj.sky_bbox_lr.dec, row['sky_bbox_lr'].dec)

        assert_quantity_allclose(obj.sky_bbox_ur.ra, row['sky_bbox_ur'].ra)
        assert_quantity_allclose(obj.sky_bbox_ur.dec, row['sky_bbox_ur'].dec)

    def test_table_no_wcs(self):
        cat = source_properties(IMAGE, SEGM)
        columns = ['sky_centroid', 'sky_centroid_icrs', 'icrs_centroid',
                   'ra_icrs_centroid', 'dec_icrs_centroid', 'sky_bbox_ll',
                   'sky_bbox_ul', 'sky_bbox_lr', 'sky_bbox_ur']
        t = cat.to_table(columns=columns)
        assert t[0]['sky_centroid'] is None
        assert t.colnames == columns
