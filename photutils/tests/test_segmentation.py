# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.modeling import models
from astropy.table import Table
from astropy.utils.misc import isiterable
from ..segmentation import (SegmentProperties, segment_properties,
                            properties_table)
try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import skimage
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

ERR_VALS = [2., 2., 2., 2.]
EFFGAIN_VALS = [None, 2., 1.e10, 2.]
BACKGRD_VALS = [None, None, None, 5.]


@pytest.mark.skipif('not HAS_SKIMAGE')
@pytest.mark.skipif('not HAS_SCIPY')
class TestSegmentProperties(object):
    def test_segment_shape(self):
        with pytest.raises(ValueError):
            SegmentProperties(IMAGE, np.zeros((2, 2)), label=1)

    @pytest.mark.parametrize('label', (0, -1))
    def test_label_invalid(self, label):
        with pytest.raises(ValueError):
            SegmentProperties(IMAGE, SEGM, label=label)

    @pytest.mark.parametrize('label', (0, -1))
    def test_label_missing(self, label):
        segm = SEGM.copy()
        segm[0:2, 0:2] = 3   # skip label 2
        with pytest.raises(ValueError):
            SegmentProperties(IMAGE, segm, label=2, label_slice=None)


@pytest.mark.skipif('not HAS_SKIMAGE')
@pytest.mark.skipif('not HAS_SCIPY')
class TestSegmentPropertiesFunctionInputs(object):
    def test_segment_shape(self):
        wrong_shape = np.zeros((2, 2))
        with pytest.raises(ValueError):
            segment_properties(IMAGE, wrong_shape)

    def test_error_shape(self):
        wrong_shape = np.zeros((2, 2))
        with pytest.raises(ValueError):
            segment_properties(IMAGE, SEGM, error=wrong_shape)

    def test_gain_shape(self):
        wrong_shape = np.zeros((2, 2))
        with pytest.raises(ValueError):
            segment_properties(IMAGE, SEGM, error=IMAGE,
                               effective_gain=wrong_shape)

    def test_background_shape(self):
        wrong_shape = np.zeros((2, 2))
        with pytest.raises(ValueError):
            segment_properties(IMAGE, SEGM, background=wrong_shape)

    def test_mask_shape(self):
        wrong_shape = np.zeros((2, 2))
        with pytest.raises(ValueError):
            segment_properties(IMAGE, SEGM, mask=wrong_shape)

    def test_labels(self):
        props = segment_properties(IMAGE, SEGM, labels=1)
        assert props[0].id == 1

    def test_invalidlabels(self):
        props = segment_properties(IMAGE, SEGM, labels=-1)
        assert len(props) == 0


@pytest.mark.skipif('not HAS_SKIMAGE')
@pytest.mark.skipif('not HAS_SCIPY')
class TestSegmentPropertiesFunction(object):
    def test_properties(self):
        props = segment_properties(IMAGE, SEGM)
        assert props[0].id == 1
        assert_allclose(props[0].xcentroid, XCEN, rtol=1.e-2)
        assert_allclose(props[0].ycentroid, YCEN, rtol=1.e-2)
        assert_allclose(props[0].segment_sum, IMAGE[IMAGE >= THRESHOLD].sum())
        assert_allclose(props[0].semimajor_axis_sigma, MAJOR_SIG, rtol=1.e-2)
        assert_allclose(props[0].semiminor_axis_sigma, MINOR_SIG, rtol=1.e-2)
        assert_allclose(props[0].orientation, THETA, rtol=1.e-3)
        assert_allclose(props[0].bbox, [35, 25, 70, 77])
        assert_allclose(props[0].area, 1058.0)
        assert_allclose(len(props[0].values), props[0].area)
        assert_allclose(len(props[0].coords), 2)
        assert_allclose(len(props[0].coords[0]), props[0].area)

        properties = ['background_atcentroid', 'background_mean',
                      'eccentricity', 'ellipticity', 'elongation',
                      'equivalent_radius', 'max_value', 'maxval_xpos',
                      'maxval_ypos', 'min_value', 'minval_xpos',
                      'minval_ypos', 'perimeter', 'se_cxx', 'se_cxy',
                      'se_cyy', 'se_x2', 'se_xy', 'se_y2', 'xmax',
                      'xmin', 'ymax', 'ymin']
        for propname in properties:
            assert not isiterable(getattr(props[0], propname))

        properties = ['centroid', 'covariance_eigvals', 'local_centroid',
                      'maxval_local_pos', 'minval_local_pos']
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
        props = segment_properties(IMAGE, SEGM, background=value)
        assert props[0].background_mean == value
        assert props[0].background_atcentroid == value

    def test_properties_error_background_None(self):
        props = segment_properties(IMAGE, SEGM)
        assert props[0].background_cutout_ma is None
        assert props[0].error_cutout_ma is None

    def test_cutout_shapes(self):
        error = np.ones_like(IMAGE) * 1.
        props = segment_properties(IMAGE, SEGM, error=error, background=1.)
        bbox = props[0].bbox.value
        true_shape = (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1)
        properties = ['background_cutout_ma', 'data_cutout',
                      'data_cutout_ma', 'error_cutout_ma']
        shapes = [getattr(props[0], p).shape for p in properties]
        for shape in shapes:
            assert shape == true_shape

    @pytest.mark.parametrize(('error_value', 'effective_gain', 'background'),
                             zip(ERR_VALS, EFFGAIN_VALS, BACKGRD_VALS))
    def test_segmentation_inputs(self, error_value, effective_gain,
                                 background):
        error = np.ones_like(IMAGE) * error_value
        props = segment_properties(IMAGE, SEGM, error=error,
                                   effective_gain=effective_gain,
                                   background=background)
        assert_allclose(props[0].xcentroid, XCEN, rtol=1.e-2)
        assert_allclose(props[0].ycentroid, YCEN, rtol=1.e-2)
        assert_allclose(props[0].semimajor_axis_sigma, MAJOR_SIG, rtol=1.e-2)
        assert_allclose(props[0].semiminor_axis_sigma, MINOR_SIG, rtol=1.e-2)
        assert_allclose(props[0].orientation, THETA, rtol=1.e-3)
        assert_allclose(props[0].bbox, [35, 25, 70, 77])
        assert_allclose(props[0].area, 1058.0)
        if background is None:
            background_sum = 0.
        else:
            background_sum = props[0].background_sum
        true_sum = IMAGE[IMAGE >= THRESHOLD].sum() - background_sum
        assert_allclose(props[0].segment_sum, true_sum)
        if effective_gain is None:
            true_error = np.sqrt(props[0].area) * error_value
        else:
            true_error = np.sqrt(((props[0].segment_sum + background_sum) /
                                  effective_gain) +
                                 (np.sqrt(props[0].area.value) *
                                  error_value)**2)
        assert_allclose(props[0].segment_sum_err, true_error)

    def test_mask(self):
        data = np.zeros((3, 3))
        data[0, 1] = 1.
        data[1, 1] = 1.
        mask = np.zeros_like(data, dtype=np.bool)
        mask[0, 1] = True
        segm = data.astype(np.int)
        props = segment_properties(data, segm, mask=mask)
        assert_allclose(props[0].xcentroid, 1)
        assert_allclose(props[0].ycentroid, 1)
        assert_allclose(props[0].segment_sum, 1)
        assert_allclose(props[0].area, 1)

    def test_effective_gain_negative(self, effective_gain=-1):
        error = np.ones_like(IMAGE) * 2.
        with pytest.raises(ValueError):
            segment_properties(IMAGE, SEGM, error=error,
                               effective_gain=effective_gain)

    def test_single_pixel_segment(self):
        segm = np.zeros_like(SEGM)
        segm[50, 50] = 1
        props = segment_properties(IMAGE, segm)
        assert props[0].eccentricity == 0


@pytest.mark.skipif('not HAS_SKIMAGE')
@pytest.mark.skipif('not HAS_SCIPY')
class TestPropertiesTable(object):
    def test_properties_table(self):
        props = segment_properties(IMAGE, SEGM)
        t = properties_table(props)
        assert isinstance(t, Table)
        assert len(t) == 1

    def test_properties_table_include(self):
        props = segment_properties(IMAGE, SEGM)
        columns = ['id', 'xcentroid']
        t = properties_table(props, columns=columns)
        assert isinstance(t, Table)
        assert len(t) == 1
        assert t.colnames == columns

    def test_properties_table_include_invalidname(self):
        props = segment_properties(IMAGE, SEGM)
        columns = ['idzz', 'xcentroidzz']
        with pytest.raises(AttributeError):
            properties_table(props, columns=columns)

    def test_properties_table_exclude(self):
        props = segment_properties(IMAGE, SEGM)
        exclude = ['id', 'xcentroid']
        t = properties_table(props, exclude_columns=exclude)
        assert isinstance(t, Table)
        assert len(t) == 1
        with pytest.raises(KeyError):
            t['id']

    def test_properties_table_empty_props(self):
        props = segment_properties(IMAGE, SEGM, labels=-1)
        with pytest.raises(ValueError):
            properties_table(props)

    def test_properties_table_empty_list(self):
        with pytest.raises(ValueError):
            properties_table([])
