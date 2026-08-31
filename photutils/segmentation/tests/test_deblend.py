# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the deblend module.
"""

import warnings
from unittest.mock import patch

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.utils.exceptions import (AstropyDeprecationWarning,
                                      AstropyUserWarning)
from numpy.testing import assert_allclose, assert_equal
from scipy import ndimage as ndi

from photutils.segmentation import (SegmentationImage, deblend_sources,
                                    detect_sources)
from photutils.segmentation._deblend_watershed import deblend_watershed
from photutils.segmentation.deblend import (_DeblendParams,
                                            _SingleSourceDeblender)
from photutils.segmentation.flags import SEGMENTATION_FLAGS
from photutils.segmentation.utils import _make_binary_structure
from photutils.utils._optional_deps import HAS_SKIMAGE
from photutils.utils.exceptions import DeblendWarning


class TestDeblendSources:
    @pytest.fixture(autouse=True)
    def setup(self):
        g1 = Gaussian2D(100, 50, 50, 5, 5)
        g2 = Gaussian2D(100, 35, 50, 5, 5)
        g3 = Gaussian2D(30, 70, 50, 5, 5)
        y, x = np.mgrid[0:100, 0:100]
        self.x = x
        self.y = y
        self.data = g1(x, y) + g2(x, y)
        self.data3 = self.data + g3(x, y)
        self.threshold = 10
        self.n_pixels = 5
        self.segm = detect_sources(self.data, self.threshold, self.n_pixels)
        self.segm3 = detect_sources(self.data3, self.threshold, self.n_pixels)

    @pytest.mark.parametrize('mode', ['exponential', 'linear', 'sinh'])
    def test_deblend_sources(self, mode):
        """
        Test deblend sources.
        """
        result = deblend_sources(self.data, self.segm, self.n_pixels,
                                 mode=mode)
        assert result.data.dtype == self.segm.data.dtype
        assert result.n_labels == 2
        assert result.n_labels == len(result.slices)
        mask1 = (result.data == 1)
        mask2 = (result.data == 2)
        assert_allclose(len(result.data[mask1]), len(result.data[mask2]))
        assert_allclose(np.sum(self.data[mask1]), np.sum(self.data[mask2]))
        assert_allclose(np.nonzero(self.segm), np.nonzero(result))
        assert_equal(result.parent_to_deblended_labels, {1: [1, 2]})

    def test_deblend_multiple_sources(self):
        """
        Test deblend multiple sources.
        """
        g4 = Gaussian2D(100, 50, 15, 5, 5)
        g5 = Gaussian2D(100, 35, 15, 5, 5)
        g6 = Gaussian2D(100, 50, 85, 5, 5)
        g7 = Gaussian2D(100, 35, 85, 5, 5)
        x = self.x
        y = self.y
        data = self.data + g4(x, y) + g5(x, y) + g6(x, y) + g7(x, y)
        segm = detect_sources(data, self.threshold, self.n_pixels)
        result = deblend_sources(data, segm, self.n_pixels)
        assert result.n_labels == 6
        assert result.n_labels == len(result.slices)
        assert result.areas[0] == result.areas[1]
        assert result.areas[0] == result.areas[2]
        assert result.areas[0] == result.areas[3]
        assert result.areas[0] == result.areas[4]
        assert result.areas[0] == result.areas[5]

    def test_deblend_multiple_sources_with_neighbor(self):
        """
        Test deblend multiple sources with neighbor.
        """
        g1 = Gaussian2D(100, 50, 50, 20, 5, theta=45)
        g2 = Gaussian2D(100, 35, 50, 5, 5)
        g3 = Gaussian2D(100, 60, 20, 5, 5)

        x = self.x
        y = self.y
        data = (g1 + g2 + g3)(x, y)
        segm = detect_sources(data, self.threshold, self.n_pixels)
        result = deblend_sources(data, segm, self.n_pixels)
        assert result.n_labels == 3

    def test_deblend_labels(self):
        """
        Test deblend labels.
        """
        g1 = Gaussian2D(100, 50, 50, 20, 5, theta=45)
        g2 = Gaussian2D(100, 35, 50, 5, 5)
        g3 = Gaussian2D(100, 60, 20, 5, 5)
        x = self.x
        y = self.y
        data = (g1 + g2 + g3)(x, y)
        segm = detect_sources(data, self.threshold, self.n_pixels)
        result = deblend_sources(data, segm, self.n_pixels, labels=1)
        assert result.n_labels == 2

    @pytest.mark.parametrize(('contrast', 'n_labels'),
                             [(0.001, 6), (0.017, 5), (0.06, 4), (0.1, 3),
                              (0.15, 2), (0.45, 1)])
    def test_deblend_contrast(self, contrast, n_labels):
        """
        Test deblend contrast.
        """
        y, x = np.mgrid[0:51, 0:151]
        y0 = 25
        data = (Gaussian2D(9.5, 16, y0, 5, 5)(x, y)
                + Gaussian2D(51, 30, y0, 3, 3)(x, y)
                + Gaussian2D(30, 42, y0, 5, 5)(x, y)
                + Gaussian2D(80, 66, y0, 8, 8)(x, y)
                + Gaussian2D(71, 88, y0, 8, 8)(x, y)
                + Gaussian2D(18, 119, y0, 7, 7)(x, y))

        n_pixels = 5
        segm = detect_sources(data, 1.0, n_pixels)
        segm2 = deblend_sources(data, segm, n_pixels, mode='linear',
                                n_levels=32, contrast=contrast)
        assert segm2.n_labels == n_labels

    def test_deblend_contrast_levels(self):
        """
        Test deblend contrast levels.

        Regression test for case where contrast=1.0.
        """
        y, x = np.mgrid[0:51, 0:151]
        y0 = 25
        data = (Gaussian2D(9.5, 16, y0, 5, 5)(x, y)
                + Gaussian2D(51, 30, y0, 3, 3)(x, y)
                + Gaussian2D(30, 42, y0, 5, 5)(x, y)
                + Gaussian2D(80, 66, y0, 8, 8)(x, y)
                + Gaussian2D(71, 88, y0, 8, 8)(x, y)
                + Gaussian2D(18, 119, y0, 7, 7)(x, y))

        n_pixels = 5
        segm = detect_sources(data, 1.0, n_pixels)
        for contrast in np.arange(1, 11) / 10.0:
            segm3 = deblend_sources(data, segm, n_pixels, mode='linear',
                                    n_levels=32, contrast=contrast)
            assert segm3.n_labels >= 1

    def test_deblend_connectivity(self):
        """
        Test deblend connectivity.
        """
        data = np.zeros((51, 51))
        data[15:36, 15:36] = 10.0
        data[14, 36] = 1.0
        data[13, 37] = 10
        data[14, 14] = 5.0
        data[13, 13] = 10.0
        data[36, 14] = 10.0
        data[37, 13] = 10.0
        data[36, 36] = 10.0
        data[37, 37] = 10.0

        segm = detect_sources(data, 0.1, 1, connectivity=4)
        assert segm.n_labels == 9
        segm2 = deblend_sources(data, segm, 1, mode='linear', connectivity=4)
        assert segm2.n_labels == 9

        segm = detect_sources(data, 0.1, 1, connectivity=8)
        assert segm.n_labels == 1
        segm2 = deblend_sources(data, segm, 1, mode='linear', connectivity=8)
        assert segm2.n_labels == 3

        match = 'Deblending failed for source'
        with pytest.raises(ValueError, match=match):
            deblend_sources(data, segm, 1, mode='linear', connectivity=4)

    def test_deblend_label_assignment(self):
        """
        Test to ensure newly-deblended labels are unique.
        """
        y, x = np.mgrid[0:201, 0:101]
        y0a = 35
        y1a = 60
        yshift = 100
        y0b = y0a + yshift
        y1b = y1a + yshift
        data = (Gaussian2D(80, 36, y0a, 8, 8)(x, y)
                + Gaussian2D(71, 58, y1a, 8, 8)(x, y)
                + Gaussian2D(30, 36, y1a, 7, 7)(x, y)
                + Gaussian2D(30, 58, y0a, 7, 7)(x, y)
                + Gaussian2D(80, 36, y0b, 8, 8)(x, y)
                + Gaussian2D(71, 58, y1b, 8, 8)(x, y)
                + Gaussian2D(30, 36, y1b, 7, 7)(x, y)
                + Gaussian2D(30, 58, y0b, 7, 7)(x, y))

        n_pixels = 5
        segm1 = detect_sources(data, 5.0, n_pixels)
        segm2 = deblend_sources(data, segm1, n_pixels, mode='linear',
                                n_levels=32, contrast=0.3)
        assert segm2.n_labels == 4

    @pytest.mark.parametrize('mode', ['exponential', 'linear'])
    def test_deblend_sources_norelabel(self, mode):
        """
        Test deblend sources norelabel.
        """
        result = deblend_sources(self.data, self.segm, self.n_pixels,
                                 mode=mode, relabel=False)
        assert result.n_labels == 2
        assert_equal(result.labels, [2, 3])
        assert_equal(result.parent_to_deblended_labels, {1: [2, 3]})
        assert len(result.slices) <= result.max_label
        assert len(result.slices) == result.n_labels
        assert_allclose(np.nonzero(self.segm), np.nonzero(result))

    @pytest.mark.parametrize('mode', ['exponential', 'linear'])
    def test_deblend_three_sources(self, mode):
        """
        Test deblend three sources.
        """
        result = deblend_sources(self.data3, self.segm3, self.n_pixels,
                                 mode=mode)
        assert result.n_labels == 3
        assert_allclose(np.nonzero(self.segm3), np.nonzero(result))

    def test_segmentation_image(self):
        """
        Test segmentation image.
        """
        segm_wrong = np.ones((2, 2), dtype=int)  # ndarray
        match = 'segmentation_image must be a SegmentationImage'
        with pytest.raises(TypeError, match=match):
            deblend_sources(self.data, segm_wrong, self.n_pixels)

        segm_wrong = SegmentationImage(segm_wrong)  # wrong shape
        match = 'segmentation_image must have the same shape as data'
        with pytest.raises(ValueError, match=match):
            deblend_sources(self.data, segm_wrong, self.n_pixels)

    @pytest.mark.parametrize('relabel', [False, True])
    def test_contrast_one_relabel(self, relabel):
        """
        Test that contrast=1 (no deblending) honors the relabel keyword
        for non-consecutive input labels.
        """
        segm = self.segm.copy()
        segm.reassign_label(1, 1000)
        result = deblend_sources(self.data, segm, self.n_pixels,
                                 contrast=1, relabel=relabel)
        expected = [1] if relabel else [1000]
        assert_equal(result.labels, expected)

    def test_empty_segmentation_image(self):
        """
        Test that a segmentation image with no non-zero labels raises a
        ValueError.
        """
        segm = SegmentationImage(np.zeros(self.data.shape, dtype=int))
        match = 'segmentation_image must have at least one non-zero label'
        with pytest.raises(ValueError, match=match):
            deblend_sources(self.data, segm, self.n_pixels)

    @pytest.mark.parametrize('n_pixels', [0, -5, 2.5])
    def test_invalid_n_pixels(self, n_pixels):
        """
        Test that invalid n_pixels values raise a ValueError.
        """
        match = 'n_pixels must be a positive integer'
        with pytest.raises(ValueError, match=match):
            deblend_sources(self.data, self.segm, n_pixels)

    def test_invalid_n_levels(self):
        """
        Test invalid n_levels.
        """
        match = 'n_levels must be >= 1'
        with pytest.raises(ValueError, match=match):
            deblend_sources(self.data, self.segm, self.n_pixels, n_levels=0)

    def test_invalid_contrast(self):
        """
        Test invalid contrast.
        """
        match = 'contrast must be >= 0 and <= 1'
        with pytest.raises(ValueError, match=match):
            deblend_sources(self.data, self.segm, self.n_pixels, contrast=-1)

    def test_invalid_mode(self):
        """
        Test invalid mode.
        """
        match = "mode must be 'exponential', 'linear', or 'sinh'"
        with pytest.raises(ValueError, match=match):
            deblend_sources(self.data, self.segm, self.n_pixels,
                            mode='invalid')

    def test_invalid_connectivity(self):
        """
        Test invalid connectivity.
        """
        match = 'Invalid connectivity'
        with pytest.raises(ValueError, match=match):
            deblend_sources(self.data, self.segm, self.n_pixels,
                            connectivity='invalid')

    def test_constant_source(self):
        """
        Test constant source.
        """
        data = self.data.copy()
        data[data.nonzero()] = 1.0
        result = deblend_sources(data, self.segm, self.n_pixels)
        assert_allclose(result, self.segm)

    def test_source_with_negval(self):
        """
        Test source with negval.
        """
        data = self.data.copy()
        data -= 20
        match = 'The deblending mode of one or more source labels from the'
        with pytest.warns(DeblendWarning, match=match):
            segm = deblend_sources(data, self.segm, self.n_pixels)
        assert list(segm.info) == ['nonposmin_labels']
        assert_equal(segm.info['nonposmin_labels'], [1])

    def test_flags_deblended(self):
        """
        Test that deblended children carry the deblended flag and
        nothing else when no mode fallback occurred.
        """
        result = deblend_sources(self.data, self.segm, self.n_pixels)
        assert_equal(result.flags,
                     np.full(result.n_labels,
                             SEGMENTATION_FLAGS.DEBLENDED))

    def test_flags_nonposmin_children(self):
        """
        Test that children of a parent whose mode fell back due to
        non-positive minimum data values carry both the deblended and
        fallback flags.
        """
        data = self.data.copy()
        data -= 20
        match = 'The deblending mode of one or more source labels'
        with pytest.warns(DeblendWarning, match=match):
            segm = deblend_sources(data, self.segm, self.n_pixels)
        expected = (SEGMENTATION_FLAGS.DEBLENDED
                    | SEGMENTATION_FLAGS.DEBLEND_NONPOSMIN)
        for label in segm.parent_to_deblended_labels[1]:
            idx = segm.get_index(label)
            assert segm.flags[idx] == expected

    def test_flags_detect_sources_zero(self):
        """
        Test that detect_sources output has all-zero flags.
        """
        assert_equal(self.segm.flags,
                     np.zeros(self.segm.n_labels, dtype=int))

    def test_source_zero_min(self):
        """
        Test source zero min.
        """
        data = self.data.copy()
        data -= data[self.segm.data > 0].min()
        match = 'The deblending mode of one or more source labels from the'
        with pytest.warns(DeblendWarning, match=match):
            segm = deblend_sources(data, self.segm, self.n_pixels)
        assert_equal(segm.info['nonposmin_labels'], [1])

    def test_connectivity(self):
        """
        Test connectivity.

        Regression test for #341.
        """
        data = np.zeros((3, 3))
        data[0, 0] = 2
        data[1, 1] = 2
        data[2, 2] = 1
        segm = np.zeros(data.shape, dtype=int)
        segm[data.nonzero()] = 1
        segm = SegmentationImage(segm)
        data = data * 100.0
        segm_deblend = deblend_sources(data, segm, n_pixels=1, connectivity=8)
        assert segm_deblend.n_labels == 1
        match = 'Deblending failed for source'
        with pytest.raises(ValueError, match=match):
            deblend_sources(data, segm, n_pixels=1, connectivity=4)

    def test_data_nan(self):
        """
        Test that deblending occurs even if the data within a segment
        contains one or more NaNs.

        Regression test for #658.
        """
        data = self.data.copy()
        data[50, 50] = np.nan
        segm2 = deblend_sources(data, self.segm, 5)
        assert segm2.n_labels == 2

    def test_watershed(self):
        """
        Test that the watershed input mask is a bool array.

        With scikit-image >= 0.13, the mask must be a bool array. In
        particular, if the mask array contains label 512, the watershed
        algorithm fails.
        """
        segm = self.segm.copy()
        segm.reassign_label(1, 512)
        result = deblend_sources(self.data, segm, self.n_pixels)
        assert result.n_labels == 2

    def test_nondetection(self):
        """
        Test for case where no sources are detected at one of the
        threshold levels.

        For this case, a `NoDetectionsWarning` should not be raised when
        deblending sources.
        """
        data = np.copy(self.data3)
        data[50, 50] = 1000.0
        data[50, 70] = 500.0
        self.segm = detect_sources(data, self.threshold, self.n_pixels)
        deblend_sources(data, self.segm, self.n_pixels)

    def test_nonconsecutive_labels(self):
        """
        Test nonconsecutive labels.
        """
        segm = self.segm.copy()
        segm.reassign_label(1, 1000)
        result = deblend_sources(self.data, segm, self.n_pixels)
        assert result.n_labels == 2

    def test_single_source_methods(self):
        """
        Test the multithreshold and make_markers methods of the
        _SingleSourceDeblender class.

        These methods are useful for debugging but are not currently
        used by the deblend_sources function.
        """
        data = self.data3
        segm = self.segm3
        n_pixels = 5
        footprint = np.ones((3, 3))
        deblend_params = _DeblendParams(n_pixels, footprint, 32, 0.001,
                                        'linear')
        single_debl = _SingleSourceDeblender(data, segm.data, 1,
                                             deblend_params)
        segms = single_debl.multithreshold()
        assert len(segms) == 32

        markers = single_debl.make_markers(return_all=True)
        assert len(markers) == 19

    def test_info_empty_without_warnings(self):
        """
        Test that the returned segmentation image always has an info
        attribute, which is an empty dict when no deblending warnings
        occurred.
        """
        result = deblend_sources(self.data, self.segm, self.n_pixels)
        assert result.info == {}

        # detect_sources output must also have an info attribute
        assert self.segm.info == {}

    @pytest.mark.parametrize('kwargs', [{'progress_bar': False},
                                        {'n_processes': 2}])
    def test_deprecated_keywords(self, kwargs):
        """
        Test that the progress_bar and n_processes keywords are
        deprecated and have no effect on the results.
        """
        name = next(iter(kwargs))
        with pytest.warns(AstropyDeprecationWarning, match=name):
            result = deblend_sources(self.data, self.segm,
                                     self.n_pixels, mode='linear',
                                     **kwargs)
        assert result.n_labels == 2


def make_marker_test_image(kind):
    """
    Return an image containing a single connected source for the
    marker-path equivalence tests.

    Parameters
    ----------
    kind : {'blend', 'quantized', 'plateau'}
        The image type. 'blend' is a Gaussian envelope with compact
        peaks spanning a wide amplitude range, 'quantized' is the
        same image coarsely quantized (duplicate data values produce
        empty multithreshold levels), and 'plateau' contains flat
        stepped square annuli with two embedded peaks.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The image.
    """
    if kind == 'plateau':
        data = np.zeros((101, 101))
        data[10:90, 10:90] = 1.0
        data[30:70, 30:70] = 2.0
        data[35:45, 35:45] = 5.0
        data[55:65, 55:65] = 4.0
        return data

    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:151, 0:151]
    data = Gaussian2D(20, 75, 75, 40, 40)(x, y)
    amplitudes = np.geomspace(3.0, 100.0, 12)
    radii = 40 * np.sqrt(rng.uniform(0.0, 1.0, 12))
    angles = rng.uniform(0.0, 2.0 * np.pi, 12)
    for amplitude, radius, angle in zip(amplitudes, radii, angles,
                                        strict=True):
        xc = 75 + radius * np.cos(angle)
        yc = 75 + radius * np.sin(angle)
        data += Gaussian2D(amplitude, xc, yc, 2.5, 2.5)(x, y)
    if kind == 'quantized':
        data = np.round(data * 2.0) / 2.0
    return data


def normalize_markers(markers):
    """
    Relabel a marker image with consecutive raster-ordered labels.

    Parameters
    ----------
    markers : 2D int `~numpy.ndarray`
        The marker image.

    Returns
    -------
    result : 2D int `~numpy.ndarray`
        The relabeled marker image.
    """
    from photutils.segmentation.deblend import _create_relabel_map
    relabel_map = _create_relabel_map(markers)
    if relabel_map is not None:
        markers = relabel_map[markers]
    return markers


@pytest.mark.parametrize('kind', ['blend', 'quantized', 'plateau'])
@pytest.mark.parametrize('mode', ['exponential', 'linear', 'sinh'])
@pytest.mark.parametrize('connectivity', [8, 4])
def test_make_markers_matches_legacy(kind, mode, connectivity):
    """
    Test that make_markers produces the same markers as the legacy
    per-level path (the last image of the return_all=True chain).

    The markers must contain the same regions with the same
    raster-scan label ordering, since the ordering determines the
    final deblended label assignment.
    """
    from photutils.segmentation.utils import _make_binary_structure

    data = make_marker_test_image(kind)
    segm = detect_sources(data, 0.5, 5, connectivity=connectivity)
    footprint = _make_binary_structure(2, connectivity)
    n_seen = 0
    for label, slc in zip(segm.labels, segm.slices, strict=True):
        params = _DeblendParams(5, footprint, 32, 0.001, mode)
        deblender = _SingleSourceDeblender(data[slc], segm.data[slc],
                                           label, params)
        markers = deblender.make_markers()

        params = _DeblendParams(5, footprint, 32, 0.001, mode)
        deblender = _SingleSourceDeblender(data[slc], segm.data[slc],
                                           label, params)
        legacy = deblender.make_markers(return_all=True)
        legacy = None if legacy is None else legacy[-1]

        if markers is None or legacy is None:
            assert markers is None
            assert legacy is None
        else:
            n_seen += 1
            assert_equal(normalize_markers(markers),
                         normalize_markers(legacy))
    assert n_seen >= 1


@pytest.mark.skipif(not HAS_SKIMAGE, reason='skimage is required')
@pytest.mark.parametrize('connectivity', [8, 4])
def test_watershed_matches_skimage(connectivity):
    """
    Test that the deblending watershed kernel produces results identical
    to skimage.segmentation.watershed over randomized images, including
    integer-valued and constant images whose plateaus exercise the
    queue-age tie-breaking.
    """
    from skimage.segmentation import watershed

    footprint = _make_binary_structure(2, connectivity)
    rng = np.random.default_rng(987)
    n_run = 0
    for trial in range(150):
        ny, nx = rng.integers(5, 35, 2)
        kind = trial % 4
        if kind == 0:
            image = rng.normal(0.0, 1.0, (ny, nx))
        elif kind == 1:
            image = rng.integers(0, 4, (ny, nx)).astype(float)
        elif kind == 2:
            image = np.zeros((ny, nx))
        else:
            image = np.round(rng.normal(0.0, 1.0, (ny, nx)), 1)
        mask = rng.random((ny, nx)) < 0.8
        indices = np.flatnonzero(mask)
        if indices.size < 4:
            continue
        seeds = np.zeros((ny, nx), dtype=bool)
        pick = rng.choice(indices, size=min(6, indices.size),
                          replace=False)
        seeds.ravel()[pick] = True
        markers = ndi.label(seeds,
                            structure=footprint)[0].astype(np.int32)
        expected = watershed(image, markers, mask=mask,
                             connectivity=footprint)
        result = deblend_watershed(image, markers, mask, connectivity)
        assert_equal(result, expected)
        n_run += 1
    assert n_run > 100


def python_deblend_chunk(data, segm_data, driver_data,  # noqa: ARG001
                         driver_segm,  # noqa: ARG001
                         labels, slices, deblend_params):
    """
    Deblend a chunk of sources with the pure-Python reference path.

    This mirrors the compiled chunk driver used by deblend_sources,
    computing the markers and fallbacks per source in Python.

    Parameters
    ----------
    data, segm_data, driver_data, driver_segm : 2D `~numpy.ndarray`
        The (driver-compatible) data and segmentation arrays.

    labels : 1D `~numpy.ndarray`
        The labels of the sources in the chunk.

    slices : list of tuple of slice
        The bounding-box slices of the sources in the chunk.

    deblend_params : `_DeblendParams`
        The parameters for deblending the sources.

    Returns
    -------
    results : list of (2D `~numpy.ndarray` or `None`, dict)
        The deblended cutout and warnings for each source.
    """
    results = []
    for label, slc in zip(labels, slices, strict=True):
        deblender = _SingleSourceDeblender(data[slc], segm_data[slc],
                                           label, deblend_params)
        results.append((deblender.deblend_source(),
                        deblender.warnings))
    return results


@pytest.mark.parametrize('dtype', ['float64', 'float32', 'int32'])
@pytest.mark.parametrize('scene', ['blend', 'negmin', 'checkerboard',
                                   'gaussian', 'flat',
                                   'contrast-batch', 'contrast-single',
                                   'contrast-all', 'contrast-negmin'])
def test_chunk_driver_matches_python_path(dtype, scene):
    """
    Test that the compiled chunk driver and contrast loop produce
    results identical to the pure-Python per-source path, including
    the threshold computation, the mode fallbacks, the below-contrast
    marker removal, and the recorded warnings, for float64, float32, and
    integer data.

    The scenes cover deblending sources, the non-positive-minimum and
    too-many-markers mode fallbacks, a source that does not split,
    a constant source, and contrast values that trigger the batched
    removal, the one-at-a-time removal, the removal of all but one
    basin, and the removal path for sources with a negative minimum
    (which always removes one marker at a time).
    """
    from photutils.segmentation import deblend as deblend_module

    contrast = 0.001
    if scene == 'blend':
        data = make_marker_test_image('blend')
        threshold, n_pixels = 0.5, 5
    elif scene == 'negmin':
        data = make_marker_test_image('blend') - 15.0
        threshold, n_pixels = -14.5, 5
    elif scene.startswith('contrast'):
        data, _ = make_multipeak_source()
        threshold, n_pixels = 0.5, 5
        contrast = {'contrast-batch': 0.15, 'contrast-single': 0.07,
                    'contrast-all': 0.35,
                    'contrast-negmin': 0.15}[scene]
        if scene == 'contrast-negmin':
            data = data - 5.0
            threshold = -4.5
    elif scene == 'gaussian':
        y, x = np.mgrid[0:51, 0:51]
        data = Gaussian2D(10, 25, 25, 5, 5)(x, y)
        threshold, n_pixels = 0.5, 5
    elif scene == 'flat':
        data = np.zeros((51, 51))
        data[20:40, 20:40] = 5.0
        threshold, n_pixels = 0.5, 5
    else:
        # The n_markers fallback checkerboard scene
        size = 51
        data1 = np.resize([0, 0, 1, 1], size)
        data1 = np.abs(data1 - np.atleast_2d(data1).T) + 2.0
        for i in range(size):
            if i % 2 == 0:
                data1[i, :] = 1
                data1[:, i] = 1
        data = np.zeros((101, 101))
        data[25:25 + size, 25:25 + size] = data1
        data[50:60, 50:60] = 10.0
        threshold, n_pixels = 0.01, 1

    data = data.astype(dtype)
    segm = detect_sources(data, threshold, n_pixels)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeblendWarning)
        result = deblend_sources(data, segm, n_pixels,
                                 contrast=contrast)
        with patch.object(deblend_module, '_deblend_sources_chunk',
                          python_deblend_chunk):
            expected = deblend_sources(data, segm, n_pixels,
                                       contrast=contrast)

    assert_equal(result.data, expected.data)
    assert result.info.keys() == expected.info.keys()
    for key in expected.info:
        assert_equal(result.info[key], expected.info[key])
    assert result._flags_map == expected._flags_map


def test_deblend_segm_dtype():
    """
    Test that deblending a segmentation image with a non-native
    integer dtype gives the same result as the int32 one.
    """
    data, segm = make_multipeak_source()
    expected = deblend_sources(data, segm, 5)
    segm16 = SegmentationImage(segm.data.astype(np.int16))
    result = deblend_sources(data, segm16, 5)
    assert_equal(result.data, expected.data)


def test_python_path_connectivity_mismatch():
    """
    Test that the pure-Python reference path raises the same error
    as the compiled contrast loop when the detection and deblending
    connectivities differ.
    """
    from photutils.segmentation import deblend as deblend_module

    data = np.zeros((51, 51))
    data[15:36, 15:36] = 10.0
    data[14, 36] = 1.0
    data[13, 37] = 10
    data[14, 14] = 5.0
    data[13, 13] = 10.0
    data[36, 14] = 10.0
    data[37, 13] = 10.0
    data[36, 36] = 10.0
    data[37, 37] = 10.0
    segm = detect_sources(data, 0.1, 1, connectivity=8)
    match = 'Deblending failed for source'
    with (patch.object(deblend_module, '_deblend_sources_chunk',
                       python_deblend_chunk),
          pytest.raises(ValueError, match=match)):
        deblend_sources(data, segm, 1, mode='linear', connectivity=4)


def make_multipeak_source():
    """
    Return the image and segmentation image for a single connected
    source with peaks spanning a wide range of basin fluxes.

    The watershed basin flux fractions are approximately 0.061, 0.062,
    0.225, and 0.652, so the contrast keyword controls how many of the
    faintest basins fail the contrast criterion.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The image.

    segm : `~photutils.segmentation.SegmentationImage`
        The segmentation image containing a single label.
    """
    y, x = np.mgrid[0:101, 0:101]
    envelope = Gaussian2D(1.0, 50, 50, 30, 30)
    g_bright = Gaussian2D(100, 40, 50, 3, 3)
    g_medium = Gaussian2D(30, 68, 50, 3, 3)
    g_faint1 = Gaussian2D(3.0, 50, 30, 3, 3)
    g_faint2 = Gaussian2D(4.0, 50, 72, 3, 3)
    data = (envelope(x, y) + g_bright(x, y) + g_medium(x, y)
            + g_faint1(x, y) + g_faint2(x, y))
    segm = detect_sources(data, 0.5, 5)
    return data, segm


@pytest.mark.parametrize(('contrast', 'n_labels'),
                         [(0.0, 4), (0.07, 2), (0.15, 2), (0.35, 1)])
def test_contrast_removal(contrast, n_labels):
    """
    Test the below-contrast marker removal over a range of contrasts.

    The contrast=0.15 case removes the two faintest basins together
    (their total flux fraction is below both the contrast and the
    next-faintest basin flux), the contrast=0.07 case removes the same
    two basins one at a time (their total is above the contrast),
    and the contrast=0.35 case removes all but one basin so that no
    deblending occurs.
    """
    data, segm = make_multipeak_source()
    result = deblend_sources(data, segm, 5, contrast=contrast)
    assert result.n_labels == n_labels
    assert_equal(np.nonzero(segm.data), np.nonzero(result.data))
    if n_labels > 1:
        assert_equal(result.parent_to_deblended_labels,
                     {1: list(range(1, n_labels + 1))})
    else:
        assert_equal(result.parent_to_deblended_labels, {})


def test_n_markers_fallback():
    """
    Test that if there are too many markers, a warning is raised.
    """
    size = 51
    data1 = np.resize([0, 0, 1, 1], size)
    data1 = np.abs(data1 - np.atleast_2d(data1).T) + 2

    for i in range(size):
        if i % 2 == 0:
            data1[i, :] = 1
            data1[:, i] = 1

    data = np.zeros((101, 101))
    data[25:25 + size, 25:25 + size] = data1
    data[50:60, 50:60] = 10.0

    segm = detect_sources(data, 0.01, 10)
    match = 'The deblending mode of one or more source labels from the'
    with pytest.warns(DeblendWarning, match=match):
        segm2 = deblend_sources(data, segm, 1, mode='exponential')
    assert segm2.info['n_markers_labels'][0] == 1


def test_flags_n_markers_fallback():
    """
    Test that the n_markers fallback flag is set on the output
    sources produced from the affected input label.
    """
    size = 51
    data1 = np.resize([0, 0, 1, 1], size)
    data1 = np.abs(data1 - np.atleast_2d(data1).T) + 2

    for i in range(size):
        if i % 2 == 0:
            data1[i, :] = 1
            data1[:, i] = 1

    data = np.zeros((101, 101))
    data[25:25 + size, 25:25 + size] = data1
    data[50:60, 50:60] = 10.0

    segm = detect_sources(data, 0.01, 10)
    match = 'The deblending mode of one or more source labels'
    with pytest.warns(DeblendWarning, match=match):
        segm2 = deblend_sources(data, segm, 1, mode='exponential')

    bit = SEGMENTATION_FLAGS.DEBLEND_N_MARKERS
    flagged = segm2.labels[(segm2.flags & bit) > 0]
    assert len(flagged) > 0
    # Every flagged label traces back to input label 1
    if 1 in segm2.parent_to_deblended_labels:
        assert_equal(np.sort(flagged),
                     np.sort(segm2.parent_to_deblended_labels[1]))
    else:
        assert_equal(flagged, [1])


@pytest.mark.parametrize('relabel', [True, False])
def test_flags_fallback_without_deblending(relabel):
    """
    Test that a fallback parent that did not split still gets the
    fallback flag on its output label, with and without relabeling.
    """
    # Three well-separated, non-deblendable sources with a negative
    # minimum (to trigger the nonposmin fallback). The middle label is
    # removed after detection so the remaining input labels (1 and 3)
    # are non-consecutive, exercising the relabel-map translation path
    # when relabel=True.
    g1 = Gaussian2D(100, 25, 25, 3, 3)
    g2 = Gaussian2D(100, 50, 50, 3, 3)
    g3 = Gaussian2D(100, 75, 75, 3, 3)
    yy, xx = np.mgrid[0:101, 0:101]
    data = g1(xx, yy) + g2(xx, yy) + g3(xx, yy) - 20.0

    segm = detect_sources(data + 20.0, 10, 5)
    assert_equal(segm.labels, [1, 2, 3])
    segm.remove_label(2, relabel=False)
    assert_equal(segm.labels, [1, 3])

    match = 'The deblending mode of one or more source labels'
    with pytest.warns(DeblendWarning, match=match):
        segm2 = deblend_sources(data, segm, 5,
                                relabel=relabel)

    bit = SEGMENTATION_FLAGS.DEBLEND_NONPOSMIN
    # Both remaining sources fell back (both have negative minima);
    # neither splits, so each output label carries the fallback bit
    # but not the deblended bit
    assert_equal(segm2.flags & bit, [bit] * segm2.n_labels)
    assert_equal(segm2.flags & SEGMENTATION_FLAGS.DEBLENDED,
                 [0] * segm2.n_labels)


def test_nonposmin_astropy_user_warning():
    """
    Test that the nonposmin warning is caught as an
    AstropyUserWarning, checking that DeblendWarning is a subclass of
    it so that existing warning filters continue to work.
    """
    g1 = Gaussian2D(100, 50, 50, 8, 8)
    g2 = Gaussian2D(100, 35, 50, 8, 8)
    yy, xx = np.mgrid[0:101, 0:101]
    data = g1(xx, yy) + g2(xx, yy) - 20  # negative values

    segm = detect_sources(data + 20, 10, 5)  # detect sources on positive data
    match = 'The deblending mode of one or more source labels from the'
    with pytest.warns(AstropyUserWarning, match=match):
        segm2 = deblend_sources(data, segm, 5)
    assert 'nonposmin_labels' in segm2.info
    assert np.all(segm2.flags & SEGMENTATION_FLAGS.DEBLEND_NONPOSMIN)


def test_n_markers_fallback_returns_none():
    """
    Test that deblend_source returns None when make_markers returns
    None on the linear-mode fallback (second attempt after >200
    markers).
    """
    # Create a source with varying data values so source_min != source_max
    data = np.ones((20, 20)) * 10.0
    data[5:15, 5:15] = 50.0
    data[8:12, 8:12] = 100.0  # peak in center
    segment = np.zeros((20, 20), dtype=int)
    segment[5:15, 5:15] = 1

    deblend_params = _DeblendParams(5, np.ones((3, 3)), 32, 0.001,
                                    'exponential')

    deblender = _SingleSourceDeblender(data, segment, 1, deblend_params)

    call_count = [0]

    def mock_make_markers(*, _return_all=False):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: return markers with > 200 labels
            markers = np.zeros((20, 20), dtype=int)
            for i in range(201):
                r, c = divmod(i, 20)
                if r < 20 and c < 20:
                    markers[r, c] = i + 1
            return markers
        # Second call (linear fallback): return None
        return None

    with patch.object(deblender, 'make_markers', mock_make_markers):
        result = deblender.deblend_source()

    assert result is None
