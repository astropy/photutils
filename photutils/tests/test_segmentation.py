# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from distutils.version import LooseVersion
import itertools
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest, assert_quantity_allclose
from astropy.modeling import models
from astropy.table import Table
import astropy.units as u
from astropy.utils.misc import isiterable
import astropy.wcs as WCS
from ..segmentation import (SegmentationImage, SourceProperties,
                            source_properties, properties_table)

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import skimage
    HAS_SKIMAGE = True
    if LooseVersion(skimage.__version__) < LooseVersion('0.11'):
        SKIMAGE_LT_0P11 = True
    else:
        SKIMAGE_LT_0P11 = False
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
EFFGAIN_VALS = [None, 2., 1.e10]
BACKGRD_VALS = [None, 0., 1., 3.5]


@pytest.mark.skipif('not HAS_SKIMAGE')
@pytest.mark.skipif('not HAS_SCIPY')
class TestSegmentationImage(object):
    def setup_class(self):
        self.data = [[1, 1, 0, 0, 4, 4],
                     [0, 0, 0, 0, 0, 4],
                     [0, 0, 3, 3, 0, 0],
                     [7, 0, 0, 0, 0, 5],
                     [7, 7, 0, 5, 5, 5],
                     [7, 7, 0, 0, 5, 5]]
        self.segm = SegmentationImage(self.data)

    def test_array(self):
        assert_allclose(self.segm.data, self.segm.array)
        assert_allclose(self.segm.data, self.segm.__array__())

    def test_negative_data(self):
        data = np.arange(-1, 8).reshape(3, 3)
        with pytest.raises(ValueError):
            SegmentationImage(data)

    def test_zero_label(self):
        with pytest.raises(ValueError):
            self.segm.check_label(0)

    def test_negative_label(self):
        with pytest.raises(ValueError):
            self.segm.check_label(-1)

    def test_invalid_label(self):
        with pytest.raises(ValueError):
            self.segm.check_label(2)

    def test_data_masked(self):
        assert isinstance(self.segm.data_masked, np.ma.MaskedArray)
        assert np.ma.count(self.segm.data_masked) == 18
        assert np.ma.count_masked(self.segm.data_masked) == 18

    def test_labels(self):
        assert_allclose(self.segm.labels, [1, 3, 4, 5, 7])

    def test_nlabels(self):
        assert self.segm.nlabels == 5

    def test_max(self):
        assert self.segm.max == 7

    def test_outline_segments(self):
        if SKIMAGE_LT_0P11:
            return    # skip this test
        segm_array = np.zeros((5, 5)).astype(int)
        segm_array[1:4, 1:4] = 2
        segm = SegmentationImage(segm_array)
        segm_array_ref = np.copy(segm_array)
        segm_array_ref[2, 2] = 0
        assert_allclose(segm.outline_segments(), segm_array_ref)

    def test_outline_segments_masked_background(self):
        if SKIMAGE_LT_0P11:
            return    # skip this test
        segm_array = np.zeros((5, 5)).astype(int)
        segm_array[1:4, 1:4] = 2
        segm = SegmentationImage(segm_array)
        segm_array_ref = np.copy(segm_array)
        segm_array_ref[2, 2] = 0
        segm_outlines = segm.outline_segments(mask_background=True)
        assert isinstance(segm_outlines, np.ma.MaskedArray)
        assert np.ma.count(segm_outlines) == 8
        assert np.ma.count_masked(segm_outlines) == 17

    def test_relabel(self):
        segm = SegmentationImage(self.data)
        segm.relabel(labels=[1, 7], new_label=2)
        ref_data = np.array([[2, 2, 0, 0, 4, 4],
                             [0, 0, 0, 0, 0, 4],
                             [0, 0, 3, 3, 0, 0],
                             [2, 0, 0, 0, 0, 5],
                             [2, 2, 0, 5, 5, 5],
                             [2, 2, 0, 0, 5, 5]])
        assert_allclose(segm.data, ref_data)
        assert segm.nlabels == len(segm.slices) - segm.slices.count(None)

    @pytest.mark.parametrize('start_label', [1, 5])
    def test_relabel_sequential(self, start_label):
        segm = SegmentationImage(self.data)
        ref_data = np.array([[1, 1, 0, 0, 3, 3],
                             [0, 0, 0, 0, 0, 3],
                             [0, 0, 2, 2, 0, 0],
                             [5, 0, 0, 0, 0, 4],
                             [5, 5, 0, 4, 4, 4],
                             [5, 5, 0, 0, 4, 4]])
        ref_data[ref_data != 0] += (start_label - 1)
        segm.relabel_sequential(start_label=start_label)
        assert_allclose(segm.data, ref_data)

        # relabel_sequential should do nothing if already sequential
        segm.relabel_sequential(start_label=start_label)
        assert_allclose(segm.data, ref_data)
        assert segm.nlabels == len(segm.slices) - segm.slices.count(None)

    @pytest.mark.parametrize('start_label', [0, -1])
    def test_relabel_sequential_start_invalid(self, start_label):
        with pytest.raises(ValueError):
            segm = SegmentationImage(self.data)
            segm.relabel_sequential(start_label=start_label)

    def test_keep_labels(self):
        ref_data = np.array([[0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 3, 3, 0, 0],
                             [0, 0, 0, 0, 0, 5],
                             [0, 0, 0, 5, 5, 5],
                             [0, 0, 0, 0, 5, 5]])
        segm = SegmentationImage(self.data)
        segm.keep_labels([5, 3])
        assert_allclose(segm.data, ref_data)

    def test_keep_labels_relabel(self):
        ref_data = np.array([[0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0, 2],
                             [0, 0, 0, 2, 2, 2],
                             [0, 0, 0, 0, 2, 2]])
        segm = SegmentationImage(self.data)
        segm.keep_labels([5, 3], relabel=True)
        assert_allclose(segm.data, ref_data)

    def test_remove_labels(self):
        ref_data = np.array([[1, 1, 0, 0, 4, 4],
                             [0, 0, 0, 0, 0, 4],
                             [0, 0, 0, 0, 0, 0],
                             [7, 0, 0, 0, 0, 0],
                             [7, 7, 0, 0, 0, 0],
                             [7, 7, 0, 0, 0, 0]])
        segm = SegmentationImage(self.data)
        segm.remove_labels(labels=[5, 3])
        assert_allclose(segm.data, ref_data)

    def test_remove_labels_relabel(self):
        ref_data = np.array([[1, 1, 0, 0, 2, 2],
                             [0, 0, 0, 0, 0, 2],
                             [0, 0, 0, 0, 0, 0],
                             [3, 0, 0, 0, 0, 0],
                             [3, 3, 0, 0, 0, 0],
                             [3, 3, 0, 0, 0, 0]])
        segm = SegmentationImage(self.data)
        segm.remove_labels(labels=[5, 3], relabel=True)
        assert_allclose(segm.data, ref_data)

    def test_remove_border_labels(self):
        ref_data = np.array([[0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 3, 3, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]])
        segm = SegmentationImage(self.data)
        segm.remove_border_labels(border_width=1)
        assert_allclose(segm.data, ref_data)

    def test_remove_border_labels_border_width(self):
        with pytest.raises(ValueError):
            segm = SegmentationImage(self.data)
            segm.remove_border_labels(border_width=3)

    def test_remove_masked_labels(self):
        ref_data = np.array([[0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 3, 3, 0, 0],
                             [7, 0, 0, 0, 0, 5],
                             [7, 7, 0, 5, 5, 5],
                             [7, 7, 0, 0, 5, 5]])
        segm = SegmentationImage(self.data)
        mask = np.zeros_like(segm.data, dtype=np.bool)
        mask[0, :] = True
        segm.remove_masked_labels(mask)
        assert_allclose(segm.data, ref_data)

    def test_remove_masked_labels_without_partial_overlap(self):
        ref_data = np.array([[0, 0, 0, 0, 4, 4],
                             [0, 0, 0, 0, 0, 4],
                             [0, 0, 3, 3, 0, 0],
                             [7, 0, 0, 0, 0, 5],
                             [7, 7, 0, 5, 5, 5],
                             [7, 7, 0, 0, 5, 5]])
        segm = SegmentationImage(self.data)
        mask = np.zeros_like(segm.data, dtype=np.bool)
        mask[0, :] = True
        segm.remove_masked_labels(mask, partial_overlap=False)
        assert_allclose(segm.data, ref_data)

    def test_remove_masked_segments_mask_shape(self):
        segm = SegmentationImage(np.ones((5, 5)))
        mask = np.zeros((3, 3), dtype=np.bool)
        with pytest.raises(ValueError):
            segm.remove_masked_labels(mask)


@pytest.mark.skipif('not HAS_SKIMAGE')
@pytest.mark.skipif('not HAS_SCIPY')
class TestSourceProperties(object):
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
        assert props.icrs_centroid is not None
        assert props.ra_icrs_centroid is not None
        assert props.dec_icrs_centroid is not None

    def test_nowcs(self):
        props = SourceProperties(IMAGE, SEGM, wcs=None, label=1)
        assert props.icrs_centroid is None
        assert props.ra_icrs_centroid is None
        assert props.dec_icrs_centroid is None

    def test_to_table(self):
        props = SourceProperties(IMAGE, SEGM, label=1)
        t1 = props.to_table()
        t2 = properties_table(props)
        assert isinstance(t1, Table)
        assert isinstance(t2, Table)
        assert len(t1) == 1
        assert t1 == t2


@pytest.mark.skipif('not HAS_SKIMAGE')
@pytest.mark.skipif('not HAS_SCIPY')
class TestSourcePropertiesFunctionInputs(object):
    def test_segment_shape(self):
        wrong_shape = np.zeros((2, 2))
        with pytest.raises(ValueError):
            source_properties(IMAGE, wrong_shape)

    def test_error_shape(self):
        wrong_shape = np.zeros((2, 2))
        with pytest.raises(ValueError):
            source_properties(IMAGE, SEGM, error=wrong_shape)

    def test_effective_gain_shape(self):
        wrong_shape = np.zeros((2, 2))
        with pytest.raises(ValueError):
            source_properties(IMAGE, SEGM, error=IMAGE,
                              effective_gain=wrong_shape)

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
class TestSourcePropertiesFunction(object):
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

    def test_properties_background_notNone(self):
        value = 1.
        props = source_properties(IMAGE, SEGM, background=value)
        assert props[0].background_mean == value
        assert_allclose(props[0].background_at_centroid, value)

    def test_properties_error_background_None(self):
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

    @pytest.mark.parametrize(('error_value', 'effective_gain', 'background'),
                             list(itertools.product(
                                 ERR_VALS, EFFGAIN_VALS, BACKGRD_VALS)))
    def test_segmentation_inputs(self, error_value, effective_gain,
                                 background):
        error = np.ones_like(IMAGE) * error_value
        props = source_properties(IMAGE, SEGM, error=error,
                                  effective_gain=effective_gain,
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
        if effective_gain is not None:
            true_error = np.sqrt(
                (props[0].source_sum / effective_gain) + true_error**2)
        assert_allclose(props[0].source_sum_err, true_error)

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

    def test_effective_gain_negative(self, effective_gain=-1):
        error = np.ones_like(IMAGE) * 2.
        with pytest.raises(ValueError):
            source_properties(IMAGE, SEGM, error=error,
                              effective_gain=effective_gain)

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


@pytest.mark.skipif('not HAS_SKIMAGE')
@pytest.mark.skipif('not HAS_SCIPY')
class TestPropertiesTable(object):
    def test_properties_table(self):
        props = source_properties(IMAGE, SEGM)
        t = properties_table(props)
        assert isinstance(t, Table)
        assert len(t) == 1

    def test_properties_table_include(self):
        props = source_properties(IMAGE, SEGM)
        columns = ['id', 'xcentroid']
        t = properties_table(props, columns=columns)
        assert isinstance(t, Table)
        assert len(t) == 1
        assert t.colnames == columns

    def test_properties_table_include_invalidname(self):
        props = source_properties(IMAGE, SEGM)
        columns = ['idzz', 'xcentroidzz']
        with pytest.raises(AttributeError):
            properties_table(props, columns=columns)

    def test_properties_table_exclude(self):
        props = source_properties(IMAGE, SEGM)
        exclude = ['id', 'xcentroid']
        t = properties_table(props, exclude_columns=exclude)
        assert isinstance(t, Table)
        assert len(t) == 1
        with pytest.raises(KeyError):
            t['id']

    def test_properties_table_empty_props(self):
        props = source_properties(IMAGE, SEGM, labels=-1)
        with pytest.raises(ValueError):
            properties_table(props)

    def test_properties_table_empty_list(self):
        with pytest.raises(ValueError):
            properties_table([])

    def test_properties_table_wcs(self):
        mywcs = WCS.WCS(naxis=2)
        rho = np.pi / 3.
        scale = 0.1 / 3600.
        mywcs.wcs.cd = [[scale*np.cos(rho), -scale*np.sin(rho)],
                        [scale*np.sin(rho), scale*np.cos(rho)]]
        mywcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']

        props = source_properties(IMAGE, SEGM, wcs=mywcs)
        columns = ['icrs_centroid', 'ra_icrs_centroid', 'dec_icrs_centroid']
        t = properties_table(props, columns=columns)
        assert t[0]['icrs_centroid'] is not None
        assert t[0]['ra_icrs_centroid'] is not None
        assert t[0]['dec_icrs_centroid'] is not None
