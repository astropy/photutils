# Licensed under a 3-clause BSD style license - see LICENSE.rst

import itertools

import numpy as np
from numpy.testing import assert_allclose
import pytest

from astropy.modeling import models
from astropy.table import QTable
from astropy.tests.helper import assert_quantity_allclose, catch_warnings
import astropy.units as u
from astropy.utils.exceptions import AstropyUserWarning
from astropy.utils.misc import isiterable
import astropy.wcs as WCS

from ..core import SegmentationImage
from ..detect import detect_sources
from ..properties import SourceProperties, source_properties, SourceCatalog

try:
    import scipy    # noqa
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

XCEN = 51.
YCEN = 52.7
MAJOR_SIG = 8.
MINOR_SIG = 3.
THETA = np.pi / 6.
g1 = models.Gaussian2D(111., XCEN, YCEN, MAJOR_SIG, MINOR_SIG, theta=THETA)
g2 = models.Gaussian2D(50, 20, 80, 5.1, 4.5)
g3 = models.Gaussian2D(70, 75, 18, 9.2, 4.5)
y, x = np.mgrid[0:100, 0:100]
IMAGE = g1(x, y) + g2(x, y) + g3(x, y)
THRESHOLD = 0.1
SEGM = detect_sources(IMAGE, THRESHOLD, npixels=5)

ERR_VALS = [0., 2.5]
BACKGRD_VALS = [None, 0., 1., 3.5]


@pytest.mark.skipif('not HAS_SCIPY')
class TestSourceProperties:
    def test_invalid_shapes(self):
        wrong_shape = np.ones((3, 3))
        with pytest.raises(ValueError):
            SourceProperties(IMAGE, np.eye(3, dtype=int), label=1)

        with pytest.raises(ValueError):
            SourceProperties(IMAGE, SEGM, label=1, filtered_data=wrong_shape)

        with pytest.raises(ValueError):
            SourceProperties(IMAGE, SEGM, label=1, error=wrong_shape)

        with pytest.raises(ValueError):
            SourceProperties(IMAGE, SEGM, label=1, background=wrong_shape)

    def test_invalid_units(self):
        unit = u.uJy
        wrong_unit = u.km

        with pytest.raises(ValueError):
            SourceProperties(IMAGE*unit, SEGM, label=1,
                             filtered_data=IMAGE*wrong_unit)

        with pytest.raises(ValueError):
            SourceProperties(IMAGE*unit, SEGM, label=1,
                             error=IMAGE*wrong_unit)

        with pytest.raises(ValueError):
            SourceProperties(IMAGE*unit, SEGM, label=1,
                             background=IMAGE*wrong_unit)

        # all array inputs must have the same unit
        with pytest.raises(ValueError):
            SourceProperties(IMAGE*unit, SEGM, label=1,
                             filtered_data=IMAGE)

    @pytest.mark.parametrize('label', (0, -1))
    def test_label_invalid(self, label):
        with pytest.raises(ValueError):
            SourceProperties(IMAGE, SEGM, label=label)

    @pytest.mark.parametrize('label', (0, -1))
    def test_label_missing(self, label):
        segm = SEGM.copy()
        segm.remove_label(2)
        with pytest.raises(ValueError):
            SourceProperties(IMAGE, segm, label=2)
            SourceProperties(IMAGE, segm, label=label)

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

        tbl = props.to_table()
        assert len(tbl) == 1

    def test_nowcs(self):
        props = SourceProperties(IMAGE, SEGM, wcs=None, label=1)
        assert props.sky_centroid_icrs is None
        assert props.sky_bbox_ll is None
        assert props.sky_bbox_ul is None
        assert props.sky_bbox_lr is None
        assert props.sky_bbox_ur is None

    def test_to_table(self):
        props = SourceProperties(IMAGE, SEGM, label=2)
        t1 = props.to_table()
        assert isinstance(t1, QTable)
        assert len(t1) == 1
        assert_quantity_allclose(t1['area'], 1058 * u.pix**2)

    def test_masks(self):
        """
        Test masks, including automatic masking of all non-finite (e.g.
        NaN, inf) values in the data array.
        """

        error = np.ones_like(IMAGE) * 5.1
        error[41, 35] = np.nan
        error[42, 36] = np.inf
        background = np.ones_like(IMAGE) * 1.2
        background[62, 55] = np.nan
        background[63, 56] = np.inf
        mask = np.zeros(IMAGE.shape).astype(bool)
        mask[45:55, :] = True
        data = np.copy(IMAGE)
        data[40, 40:45] = np.nan
        data[60, 60:65] = np.inf
        data[65, 65:70] = -np.inf

        props = SourceProperties(data, SEGM, label=2, error=error,
                                 background=background, mask=mask)

        # ensure mask is identical for data, error, and background
        assert props.data_cutout_ma.compressed().size == 677
        assert (props.data_cutout_ma.compressed().size ==
                props.filtered_data_cutout_ma.compressed().size)
        assert (props.data_cutout_ma.compressed().size ==
                props.error_cutout_ma.compressed().size)
        assert (props.data_cutout_ma.compressed().size ==
                props.background_cutout_ma.compressed().size)
        assert (len(props._filtered_data_values) ==
                props.filtered_data_cutout_ma.compressed().size)

        # test for non-finite values in error and/or background outside
        # of the data mask
        tbl = props.to_table()
        assert np.isnan(tbl['source_sum_err'])
        assert np.isnan(tbl['background_sum'])
        assert np.isnan(tbl['background_mean'])

        # test that the masks are independent objects
        assert (np.count_nonzero(props._segment_mask) !=
                np.count_nonzero(props._total_mask))
        assert (np.count_nonzero(props._data_mask) !=
                np.count_nonzero(props._total_mask))

        assert_allclose(props._data_zeroed.sum(), props.source_sum)

        assert_allclose(props.data_cutout,
                        props.make_cutout(props._data, masked_array=False))
        assert_allclose(props.data_cutout_ma,
                        props.make_cutout(props._data, masked_array=True))
        assert_allclose(props.error_cutout_ma,
                        props.make_cutout(props._error, masked_array=True))
        assert_allclose(props.background_cutout_ma,
                        props.make_cutout(props._background,
                                          masked_array=True))

    def test_completely_masked(self):
        """Test case where a source is completely masked."""

        error = np.ones_like(IMAGE) * 5.1
        background = np.ones_like(IMAGE) * 1.2
        mask = np.ones(IMAGE.shape).astype(bool)
        obj = source_properties(IMAGE, SEGM, error=error,
                                background=background, mask=mask)[0]
        assert np.isnan(obj.xcentroid.value)
        assert np.isnan(obj.ycentroid.value)
        assert np.isnan(obj.source_sum)
        assert np.isnan(obj.source_sum_err)
        assert np.isnan(obj.background_sum)
        assert np.isnan(obj.background_mean)
        assert np.isnan(obj.background_at_centroid)
        assert np.isnan(obj.area)
        assert np.isnan(obj.perimeter)
        assert np.isnan(obj.min_value)
        assert np.isnan(obj.max_value)
        assert np.isnan(obj.minval_xpos.value)
        assert np.isnan(obj.minval_ypos.value)
        assert np.isnan(obj.maxval_xpos.value)
        assert np.isnan(obj.maxval_ypos.value)
        assert np.all(np.isnan(obj.minval_cutout_pos.value))
        assert np.all(np.isnan(obj.maxval_cutout_pos.value))
        assert np.isnan(obj.gini)

    def test_repr_str(self):
        props = SourceProperties(IMAGE, SEGM, label=1)
        assert repr(props) == str(props)

        attrs = ['label', 'centroid', 'sky_centroid']
        for attr in attrs:
            assert '{}:'.format(attr) in repr(props)


@pytest.mark.skipif('not HAS_SCIPY')
class TestSourcePropertiesFunctionInputs:
    def test_segment_shape(self):
        wrong_shape = np.eye(3, dtype=int)
        with pytest.raises(ValueError):
            source_properties(IMAGE, wrong_shape)

    def test_error_shape(self):
        wrong_shape = np.ones((2, 2))
        with pytest.raises(ValueError):
            source_properties(IMAGE, SEGM, error=wrong_shape)

    def test_background_shape(self):
        wrong_shape = np.ones((2, 2))
        with pytest.raises(ValueError):
            source_properties(IMAGE, SEGM, background=wrong_shape)

    def test_mask_shape(self):
        wrong_shape = np.zeros((2, 2)).astype(bool)
        with pytest.raises(ValueError):
            source_properties(IMAGE, SEGM, mask=wrong_shape)

    def test_labels(self):
        props = source_properties(IMAGE, SEGM, labels=1)
        assert props[0].id == 1

    def test_invalidlabels(self):
        with catch_warnings(AstropyUserWarning) as warning_lines:
            source_properties(IMAGE, SEGM, labels=[-1, 1])

            assert warning_lines[0].category == AstropyUserWarning
            assert ('label -1 is not in the segmentation image.' in
                    str(warning_lines[0].message))

    def test_nosources(self):
        with pytest.raises(ValueError):
            source_properties(IMAGE, SEGM, labels=-1)


@pytest.mark.skipif('not HAS_SCIPY')
class TestSourcePropertiesFunction:
    def test_properties(self):
        obj = source_properties(IMAGE, SEGM)[1]
        assert obj.id == 2
        assert_quantity_allclose(obj.xcentroid, XCEN*u.pix,
                                 rtol=1.e-2)
        assert_quantity_allclose(obj.ycentroid, YCEN*u.pix,
                                 rtol=1.e-2)
        assert_allclose(obj.source_sum, 16723.388)
        assert_quantity_allclose(obj.semimajor_axis_sigma,
                                 MAJOR_SIG*u.pix, rtol=1.e-2)
        assert_quantity_allclose(obj.semiminor_axis_sigma,
                                 MINOR_SIG*u.pix, rtol=1.e-2)
        assert_quantity_allclose(obj.orientation, THETA*u.rad,
                                 rtol=1.e-3)
        assert obj.bbox_xmin.value == 25
        assert obj.bbox_xmax.value == 77
        assert obj.bbox_ymin.value == 35
        assert obj.bbox_ymax.value == 70

        assert_quantity_allclose(obj.area, 1058.0*u.pix**2)
        assert_allclose(len(obj.indices), 2)
        assert_allclose(len(obj.indices[0]), obj.area.value)

        properties = ['background_at_centroid', 'background_mean',
                      'eccentricity', 'ellipticity', 'elongation',
                      'equivalent_radius', 'max_value', 'maxval_xpos',
                      'maxval_ypos', 'min_value', 'minval_xpos',
                      'minval_ypos', 'perimeter', 'cxx', 'cxy', 'cyy',
                      'covar_sigx2', 'covar_sigxy', 'covar_sigy2',
                      'bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax']
        for propname in properties:
            assert not isiterable(getattr(obj, propname))

        properties = ['centroid', 'covariance_eigvals', 'cutout_centroid',
                      'maxval_cutout_pos', 'minval_cutout_pos']
        shapes = [getattr(obj, p).shape for p in properties]
        for shape in shapes:
            assert shape == (2,)

        properties = ['covariance', 'inertia_tensor']
        shapes = [getattr(obj, p).shape for p in properties]
        for shape in shapes:
            assert shape == (2, 2)

        properties = ['moments', 'moments_central']
        shapes = [getattr(obj, p).shape for p in properties]
        for shape in shapes:
            assert shape == (4, 4)

    def test_properties_background_notnone(self):
        value = 1.
        props = source_properties(IMAGE, SEGM, background=value)
        assert props[0].background_mean == value
        assert_allclose(props[0].background_at_centroid, value)

    def test_properties_background_units(self):
        unit = u.uJy
        value = 1. * unit
        props = source_properties(IMAGE * unit, SEGM, background=value)
        assert props[0].background_mean == value
        assert_allclose(props[0].background_at_centroid, value)

    def test_properties_error_background_none(self):
        props = source_properties(IMAGE, SEGM)
        assert props[0].background_cutout_ma is None
        assert props[0].error_cutout_ma is None

    def test_cutout_shapes(self):
        error = np.ones_like(IMAGE) * 1.
        props = source_properties(IMAGE, SEGM, error=error, background=1.)
        bbox = props[0].bbox
        properties = ['background_cutout_ma', 'data_cutout',
                      'data_cutout_ma', 'error_cutout_ma']
        shapes = [getattr(props[0], p).shape for p in properties]
        for shape in shapes:
            assert shape == bbox.shape

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
        obj = props[1]
        assert_quantity_allclose(obj.xcentroid, XCEN*u.pix, rtol=1.e-2)
        assert_quantity_allclose(obj.ycentroid, YCEN*u.pix, rtol=1.e-2)
        assert_quantity_allclose(obj.semimajor_axis_sigma,
                                 MAJOR_SIG*u.pix, rtol=1.e-2)
        assert_quantity_allclose(obj.semiminor_axis_sigma,
                                 MINOR_SIG*u.pix, rtol=1.e-2)
        assert_quantity_allclose(obj.orientation, THETA*u.rad,
                                 rtol=1.e-3)
        assert obj.bbox_xmin.value == 25
        assert obj.bbox_xmax.value == 77
        assert obj.bbox_ymin.value == 35
        assert obj.bbox_ymax.value == 70

        area = obj.area.value
        assert_allclose(area, 1058.0)

        if background is not None:
            assert_allclose(obj.background_sum, area * background)
        assert_allclose(obj.source_sum, 16723.388)

        true_error = np.sqrt(obj.area.value) * error_value
        assert_allclose(obj.source_sum_err, true_error)

    def test_data_allzero(self):
        props = source_properties(IMAGE*0., SEGM)
        proplist = ['xcentroid', 'ycentroid', 'semimajor_axis_sigma',
                    'semiminor_axis_sigma', 'eccentricity', 'orientation',
                    'ellipticity', 'elongation', 'cxx', 'cxy', 'cyy']
        for prop in proplist:
            assert np.isnan(getattr(props[0], prop))

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
            assert getattr(p1, key) == getattr(p2, key)
        keys = ['semimajor_axis_sigma', 'semiminor_axis_sigma']
        for key in keys:
            assert getattr(p1, key) != getattr(p2, key)

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
            assert getattr(p1, key) == getattr(p2, key)
        keys = ['semimajor_axis_sigma', 'semiminor_axis_sigma']
        for key in keys:
            assert getattr(p1, key) != getattr(p2, key)

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

    def test_nonconsecutive_labels(self):
        segm = SEGM.copy()
        segm.reassign_label(1, 1000)
        cat = source_properties(IMAGE, segm)
        assert len(cat) == 3
        assert cat.id == [2, 3, 1000]
        assert cat[0].id == 2
        assert cat[1].id == 3
        assert cat[segm.get_index(label=1000)].id == 1000


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

        # test iteration
        for obj in cat:
            assert obj.area.value == 1

    def test_inputs(self):
        cat = source_properties(IMAGE, SEGM)
        cat2 = SourceCatalog(cat[0])
        assert len(cat) == 3
        assert len(cat2) == 1

        with pytest.raises(ValueError):
            SourceCatalog([])
        with pytest.raises(ValueError):
            SourceCatalog('a')

    def test_table(self):
        cat = source_properties(IMAGE, SEGM)
        t = cat.to_table()
        assert isinstance(t, QTable)
        assert len(t) == 3

    def test_table_include(self):
        cat = source_properties(IMAGE, SEGM)
        columns = ['id', 'xcentroid']
        t = cat.to_table(columns=columns)
        assert isinstance(t, QTable)
        assert len(t) == 3
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
        assert len(t) == 3
        with pytest.raises(KeyError):
            t['id']

    def test_table_wcs(self):
        mywcs = WCS.WCS(naxis=2)
        rho = np.pi / 3.
        scale = 0.1 / 3600.
        mywcs.wcs.cd = [[scale*np.cos(rho), -scale*np.sin(rho)],
                        [scale*np.sin(rho), scale*np.cos(rho)]]
        mywcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']

        cat = source_properties(IMAGE, SEGM, wcs=mywcs)
        columns = ['sky_centroid', 'sky_centroid_icrs', 'sky_bbox_ll',
                   'sky_bbox_ul', 'sky_bbox_lr', 'sky_bbox_ur']
        t = cat.to_table(columns=columns)
        for column in columns:
            assert t[0][column] is not None
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
        columns = ['sky_centroid', 'sky_centroid_icrs', 'sky_bbox_ll',
                   'sky_bbox_ul', 'sky_bbox_lr', 'sky_bbox_ur']
        t = cat.to_table(columns=columns)
        for column in columns:
            assert t[0][column] is None
        assert t.colnames == columns

    def test_repr_str(self):
        segm = np.zeros(IMAGE.shape)
        x = y = np.arange(0, 100, 10)
        segm[y, x] = np.arange(10)
        cat = source_properties(IMAGE, segm)

        assert repr(cat) == str(cat)
        assert 'Catalog length:' in repr(cat)
