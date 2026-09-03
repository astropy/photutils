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

from photutils.segmentation import SegmentationImage
from photutils.segmentation import deblend as deblend_module
from photutils.segmentation import deblend_sources, detect_sources
from photutils.segmentation._deblend_markers import (deblend_markers_chunk,
                                                     deblend_source_stats)
from photutils.segmentation._deblend_reference import _SingleSourceDeblender
from photutils.segmentation._deblend_watershed import deblend_watershed
from photutils.segmentation.deblend import (_compute_thresholds,
                                            _create_relabel_map,
                                            _DeblendParams)
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

    @pytest.mark.parametrize('n_levels', [0, -3, 2.7])
    def test_invalid_n_levels(self, n_levels):
        """
        Test that invalid n_levels values raise a ValueError.
        """
        match = 'n_levels must be a positive integer'
        with pytest.raises(ValueError, match=match):
            deblend_sources(self.data, self.segm, self.n_pixels,
                            n_levels=n_levels)

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
        # The NaN pixel is assigned to the source surrounding it
        assert segm2.data[50, 50] == segm2.data[50, 51] != 0

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

        markers = single_debl.make_markers_per_level()
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
                                        {'n_processes': 2}, {'nproc': 2}])
    def test_deprecated_keywords(self, kwargs):
        """
        Test that the progress_bar, n_processes, and nproc keywords
        each emit a single deprecation warning and have no effect on
        the results.
        """
        name = next(iter(kwargs))
        with pytest.warns(AstropyDeprecationWarning, match=name) as record:
            result = deblend_sources(self.data, self.segm,
                                     self.n_pixels, mode='linear',
                                     **kwargs)
        assert len(record) == 1
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
    per-level path (the last image of the make_markers_per_level
    chain).

    The markers must contain the same regions with the same
    raster-scan label ordering, since the ordering determines the
    final deblended label assignment.
    """
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
        legacy = deblender.make_markers_per_level()
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


@pytest.mark.parametrize('contrast_method', ['basin', 'saddle'])
@pytest.mark.parametrize('dtype', ['float64', 'float32', '>f4', 'int32'])
@pytest.mark.parametrize('scene', ['blend', 'negmin', 'checkerboard',
                                   'gaussian', 'flat',
                                   'contrast-batch', 'contrast-single',
                                   'contrast-all', 'contrast-negmin',
                                   'neighbors', 'nan',
                                   'checkerboard-linear',
                                   'checkerboard-negmin'])
def test_chunk_driver_matches_python_path(dtype, scene, contrast_method):
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
    (which always removes one marker at a time), a scene of several
    segments with overlapping bounding boxes, NaN pixels within a
    segment, the checkerboard deblended in linear mode, which keeps
    all of its markers instead of falling back, and the checkerboard
    with a non-positive minimum, which falls back to sinh and is then
    retried with linear levels, each with both contrast criteria.
    """
    contrast = 0.001
    mode = 'linear' if scene == 'checkerboard-linear' else 'exponential'
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
    elif scene == 'neighbors':
        # Blended pairs whose bounding boxes contain other segments
        y, x = np.mgrid[0:121, 0:161]
        data = (Gaussian2D(100, 50, 60, 6, 6)(x, y)
                + Gaussian2D(90, 68, 60, 6, 6)(x, y)
                + Gaussian2D(60, 82, 40, 3, 3)(x, y)
                + Gaussian2D(80, 120, 70, 5, 5)(x, y)
                + Gaussian2D(70, 135, 70, 5, 5)(x, y)
                + Gaussian2D(50, 118, 95, 3, 3)(x, y))
        threshold, n_pixels = 5.0, 5
        contrast = 0.01
    elif scene == 'nan':
        if not np.issubdtype(np.dtype(dtype), np.floating):
            pytest.skip('NaN requires a floating-point dtype')
        data = make_marker_test_image('blend')
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
    if scene == 'checkerboard-negmin':
        # A non-positive minimum (sinh fallback) combined with too many
        # markers (linear retry), applied after the detection
        data = data - 2
    if scene == 'nan':
        # NaN pixels within the segment, set after the detection
        rng = np.random.default_rng(11)
        ys, xs = np.nonzero(segm.data > 0)
        pick = rng.choice(ys.size, size=40, replace=False)
        data[ys[pick], xs[pick]] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeblendWarning)
        result = deblend_sources(data, segm, n_pixels,
                                 contrast=contrast, mode=mode,
                                 contrast_method=contrast_method)
        with patch.object(deblend_module, '_deblend_sources_chunk',
                          python_deblend_chunk):
            expected = deblend_sources(data, segm, n_pixels,
                                       contrast=contrast, mode=mode,
                                       contrast_method=contrast_method)

    assert_equal(result.data, expected.data)
    assert result.info.keys() == expected.info.keys()
    for key in expected.info:
        assert_equal(result.info[key], expected.info[key])
    assert result._flags_map == expected._flags_map


def test_n_threads_identical():
    """
    Test that multithreaded deblending produces results identical to
    the single-threaded computation, including the mode-fallback
    warning, info, and flags, and when there are more threads than
    sources.
    """
    y, x = np.mgrid[0:101, 0:301]
    data = (Gaussian2D(100, 50, 50, 5, 5)(x, y)
            + Gaussian2D(100, 35, 50, 5, 5)(x, y)
            + Gaussian2D(80, 150, 50, 5, 5)(x, y)
            + Gaussian2D(60, 165, 50, 5, 5)(x, y)
            + Gaussian2D(50, 250, 50, 5, 5)(x, y))
    segm = detect_sources(data, 10, 5)
    assert segm.n_labels == 3
    data -= 20  # non-positive minima trigger the mode fallback

    match = 'The deblending mode of one or more source labels'
    with pytest.warns(DeblendWarning, match=match):
        expected = deblend_sources(data, segm, 5)
    assert expected.n_labels == 5
    assert_equal(expected.info['nonposmin_labels'], [1, 2, 3])
    for n_threads in (2, 3, 64):
        with pytest.warns(DeblendWarning, match=match) as record:
            result = deblend_sources(data, segm, 5, n_threads=n_threads)
        assert len(record) == 1
        assert_equal(result.data, expected.data)
        assert result._flags_map == expected._flags_map
        assert_equal(result.parent_to_deblended_labels,
                     expected.parent_to_deblended_labels)
        assert result.info.keys() == expected.info.keys()
        for key in expected.info:
            assert_equal(result.info[key], expected.info[key])


def test_n_threads_worker_error():
    """
    Test that an error raised while deblending a chunk in a worker
    thread propagates to the caller.
    """
    tile = np.zeros((51, 51))
    tile[15:36, 15:36] = 10.0
    tile[13, 13] = 10.0
    tile[14, 14] = 5.0
    tile[36, 36] = 10.0
    tile[37, 37] = 10.0
    data = np.zeros((51, 120))
    data[:, :51] = tile
    data[:, 60:111] = tile
    segm = detect_sources(data, 0.1, 1, connectivity=8)
    assert segm.n_labels == 2
    match = 'Deblending failed for source'
    with pytest.raises(ValueError, match=match):
        deblend_sources(data, segm, 1, mode='linear', connectivity=4,
                        n_threads=2)


@pytest.mark.parametrize('n_threads', [0, -1, 1.5, True])
def test_n_threads_invalid(n_threads):
    """
    Test that invalid n_threads values raise a ValueError.
    """
    data, segm = make_multipeak_source()
    match = 'n_threads must be a positive integer'
    with pytest.raises(ValueError, match=match):
        deblend_sources(data, segm, 5, n_threads=n_threads)


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


def test_deblend_byte_order():
    """
    Test that non-native byte order data and segmentation images give
    results identical to the native ones.
    """
    data, segm = make_multipeak_source()
    for dtype in ('f4', 'f8'):
        expected = deblend_sources(data.astype(f'<{dtype}'), segm, 5,
                                   contrast=0.01)
        result = deblend_sources(data.astype(f'>{dtype}'), segm, 5,
                                 contrast=0.01)
        assert_equal(result.data, expected.data)

    expected = deblend_sources(data, segm, 5, contrast=0.01)
    segm_be = SegmentationImage(segm.data.astype('>i4'))
    result = deblend_sources(data, segm_be, 5, contrast=0.01)
    assert_equal(result.data, expected.data)


@pytest.mark.parametrize('dtype', ['float64', 'float32', 'int32'])
@pytest.mark.parametrize('mode', ['exponential', 'linear', 'sinh'])
@pytest.mark.parametrize('n_levels', [1, 7, 32])
def test_compute_thresholds_matches_reference(dtype, mode, n_levels):
    """
    Test that the vectorized multithreshold levels are bitwise identical
    to the per-source levels of the reference implementation, including
    the exponential-mode fallback for non-positive minima and the
    zero-step special case of np.linspace.
    """
    rng = np.random.default_rng(42)
    n_src = 60
    if dtype == 'int32':
        smin = rng.integers(-50, 50, n_src)
        smax = smin + rng.integers(1, 1000, n_src)
    else:
        smin = rng.uniform(-5.0, 5.0, n_src)
        smax = smin + 10.0 ** rng.uniform(-3.0, 3.0, n_src)
        smin[1] = 0.0
        smax[1] = np.finfo(dtype).smallest_subnormal
    smin[0] = 0
    smin = smin.astype(dtype)
    smax = smax.astype(dtype)
    assert np.all(smax > smin)

    thresholds, nonposmin = _compute_thresholds(smin, smax, n_levels,
                                                mode)
    assert thresholds.shape == (n_src, n_levels)
    assert thresholds.dtype == np.float64
    assert thresholds.flags.c_contiguous

    params = _DeblendParams(1, np.ones((3, 3)), n_levels, 0.001, mode)
    segment = np.ones((1, 2), dtype=int)
    for i in range(n_src):
        data = np.array([[smin[i], smax[i]]], dtype=dtype)
        deblender = _SingleSourceDeblender(data, segment, 1, params)
        expected = np.asarray(deblender.compute_thresholds(),
                              dtype=np.float64)
        # Compare the bit patterns, not just the values
        assert_equal(thresholds[i].view(np.int64),
                     expected.view(np.int64))
        assert nonposmin[i] == ('nonposmin' in deblender.warnings)


@pytest.mark.parametrize('dtype', ['float64', 'float32', 'int32'])
def test_source_stats_matches_reference(dtype):
    """
    Test that the compiled per-source minimum, maximum, and flux are
    identical to the reference implementation, with NaN pixels
    excluded, and that the flux agrees with np.nansum.
    """
    data, segm = make_multipeak_source()
    data = data.astype(dtype)
    if dtype != 'int32':
        data[45:48, 40:43] = np.nan
    driver_data = np.ascontiguousarray(data, dtype=np.float64)
    labels = np.asarray(segm.labels, dtype=np.int64)
    slc = segm.slices[0]
    y0 = np.array([slc[0].start])
    y1 = np.array([slc[0].stop])
    x0 = np.array([slc[1].start])
    x1 = np.array([slc[1].stop])
    smin, smax, ssum = deblend_source_stats(driver_data, segm.data,
                                            labels, y0, y1, x0, x1)

    params = _DeblendParams(5, np.ones((3, 3)), 32, 0.001, 'linear')
    deblender = _SingleSourceDeblender(data[slc], segm.data[slc], 1,
                                       params)
    assert smin[0] == deblender.source_min
    assert smax[0] == deblender.source_max
    assert ssum[0] == deblender.source_sum
    values = data[segm.data == 1]
    assert_allclose(ssum[0], np.nansum(values, dtype=np.float64),
                    rtol=1e-12)


def test_markers_chunk_packed_buffer():
    """
    Test that the marker kernel writes each source's markers into its
    packed region, leaves the regions of sources that do not split or
    that exceed max_markers at zero, and reports the marker counts.
    """
    y, x = np.mgrid[0:61, 0:141]
    data = (Gaussian2D(100, 30, 30, 5, 5)(x, y)
            + Gaussian2D(100, 45, 30, 5, 5)(x, y)
            + Gaussian2D(50, 110, 30, 5, 5)(x, y))
    segm = detect_sources(data, 10, 5)
    assert segm.n_labels == 2
    driver_data = np.ascontiguousarray(data)
    labels = np.asarray(segm.labels, dtype=np.int64)
    y0 = np.array([slc[0].start for slc in segm.slices])
    y1 = np.array([slc[0].stop for slc in segm.slices])
    x0 = np.array([slc[1].start for slc in segm.slices])
    x1 = np.array([slc[1].stop for slc in segm.slices])
    smin, smax, _ = deblend_source_stats(driver_data, segm.data, labels,
                                         y0, y1, x0, x1)
    thresholds, _ = _compute_thresholds(smin, smax, 32, 'exponential')
    sizes = (y1 - y0) * (x1 - x0)
    offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.intp)
    packed = np.zeros(offsets[-1], dtype=np.int32)

    n_markers = deblend_markers_chunk(driver_data, segm.data, labels,
                                      y0, y1, x0, x1, thresholds,
                                      packed, offsets[:-1], n_pixels=5,
                                      connectivity=8, max_markers=-1)
    assert_equal(n_markers, [2, 0])
    region0 = packed[offsets[0]:offsets[1]]
    region1 = packed[offsets[1]:offsets[2]]
    assert_equal(np.unique(region0), [0, 1, 2])
    assert not region1.any()

    # A limit below the marker count leaves the region zero but still
    # reports the count
    packed[:] = 0
    n_markers = deblend_markers_chunk(driver_data, segm.data, labels,
                                      y0, y1, x0, x1, thresholds,
                                      packed, offsets[:-1], n_pixels=5,
                                      connectivity=8, max_markers=1)
    assert_equal(n_markers, [2, 0])
    assert not packed.any()


@pytest.mark.parametrize('dtype', ['float64', 'float32'])
@pytest.mark.parametrize('connectivity', [8, 4])
@pytest.mark.parametrize(
    ('scene', 'contrast'),
    [('multipeak', 0.0), ('multipeak', 0.001), ('multipeak', 0.01),
     ('multipeak', 0.1), ('hierarchical', 0.001),
     ('hierarchical', 0.05), ('negmin', 0.01), ('discrete', 0.01),
     ('discrete', 0.03), ('discrete', 0.05), ('quantized', 0.001),
     ('quantized', 0.01)])
def test_saddle_markers_match_reference(dtype, connectivity, scene,
                                        contrast):
    """
    Test that the compiled saddle-criterion marker selection matches
    the per-level reference implementation, including nested splits,
    dissolving below-contrast siblings, the sinh fallback for negative
    minima, both connectivities, float32 data, and data with empty
    threshold levels (discrete values and coarse quantization), where
    a branch must be evaluated at the level above its junction rather
    than at the top of its unchanged range of levels.
    """
    y, x = np.mgrid[0:101, 0:101]
    mode = 'exponential'
    if scene == 'hierarchical':
        data = (Gaussian2D(1.0, 50, 50, 30, 30)(x, y)
                + Gaussian2D(100, 35, 50, 3, 3)(x, y)
                + Gaussian2D(80, 45, 50, 3, 3)(x, y)
                + Gaussian2D(60, 70, 50, 3, 3)(x, y))
    elif scene == 'discrete':
        # A pedestal with two stepped blocks. Most of the linear levels
        # between the steps are empty
        data = np.zeros((41, 41))
        data[2:39, 2:39] = 1.0
        for cy, cx in ((12, 12), (28, 28)):
            data[cy - 2:cy + 3, cx - 2:cx + 3] = 6.0
            data[cy - 1:cy + 2, cx - 1:cx + 2] = 10.0
        mode = 'linear'
    elif scene == 'quantized':
        data = make_marker_test_image('quantized')
    else:
        data, _ = make_multipeak_source()
    threshold = 0.5
    if scene == 'negmin':
        data = data - 5.0
        threshold = -4.5
    data = data.astype(dtype)

    footprint = _make_binary_structure(2, connectivity)
    segm = detect_sources(data, threshold, 5,
                          connectivity=connectivity)
    slc = segm.slices[0]
    cutout = data[slc]

    params = _DeblendParams(5, footprint, 32, contrast, mode, 'saddle')
    deblender = _SingleSourceDeblender(cutout, segm.data[slc], 1,
                                       params)
    expected = deblender.make_markers()
    thresholds = deblender.compute_thresholds()

    y0 = np.array([slc[0].start], dtype=np.int64)
    y1 = np.array([slc[0].stop], dtype=np.int64)
    x0 = np.array([slc[1].start], dtype=np.int64)
    x1 = np.array([slc[1].stop], dtype=np.int64)
    thresholds_2d = np.ascontiguousarray(thresholds[None, :],
                                         dtype=np.float64)
    limit = deblender.contrast * float(deblender.source_sum)
    packed = np.zeros(cutout.size, dtype=np.int32)
    starts = np.zeros(1, dtype=np.intp)
    n_markers = deblend_markers_chunk(
        np.ascontiguousarray(data), segm.data,
        np.array([1], dtype=np.int64), y0, y1, x0, x1, thresholds_2d,
        packed, starts, n_pixels=5, connectivity=connectivity,
        max_markers=-1, saddle_limits=np.array([limit], dtype=np.float64))
    result = None
    if n_markers[0] >= 2:
        result = packed.reshape(cutout.shape)

    if expected is None or result is None:
        assert expected is None
        assert result is None
    else:
        assert_equal(result, expected)


def test_contrast_method_invalid():
    """
    Test that an invalid contrast_method raises a ValueError.
    """
    data, segm = make_multipeak_source()
    match = 'contrast_method must be None, .basin., or .saddle.'
    with pytest.raises(ValueError, match=match):
        deblend_sources(data, segm, 5, contrast_method='invalid')


def test_contrast_method_default():
    """
    Test that the default contrast_method of None currently resolves
    to the 'basin' method.
    """
    data, segm = make_multipeak_source()
    result_default = deblend_sources(data, segm, 5, contrast=0.1)
    result_basin = deblend_sources(data, segm, 5, contrast=0.1,
                                   contrast_method='basin')
    assert_equal(result_default.data, result_basin.data)


def test_saddle_deblend():
    """
    Test deblending with the saddle contrast criterion through the
    public API, including a source that does not split, the parent
    label map, the deblended flags, and thread-count invariance.

    The label counts are regression pins for the multipeak scene (its
    watershed basin flux fractions are about 0.061, 0.062, 0.225, and
    0.652). The results are also compared with the pure-Python
    reference path.
    """
    y, x = np.mgrid[0:101, 0:181]
    data, _ = make_multipeak_source()
    image = np.zeros((101, 181))
    image[:, :101] = data
    image += Gaussian2D(10, 150, 50, 4, 4)(x, y)  # lone source
    segm = detect_sources(image, 0.5, 5)
    assert segm.n_labels == 2

    result = deblend_sources(image, segm, 5, contrast=0.001,
                             contrast_method='saddle')
    assert result.n_labels == 5
    assert_equal(result.parent_to_deblended_labels, {1: [2, 3, 4, 5]})
    assert np.count_nonzero(result.flags
                            & SEGMENTATION_FLAGS.DEBLENDED) == 4

    # Higher contrast drops the faint basins entirely (they cannot
    # combine and survive, as there is no removal iteration).
    result2 = deblend_sources(image, segm, 5, contrast=0.1,
                              contrast_method='saddle')
    assert result2.n_labels == 3

    result3 = deblend_sources(image, segm, 5, contrast=0.001,
                              contrast_method='saddle', n_threads=4)
    assert_equal(result3.data, result.data)

    with patch.object(deblend_module, '_deblend_sources_chunk',
                      python_deblend_chunk):
        expected = deblend_sources(image, segm, 5, contrast=0.001,
                                   contrast_method='saddle')
        expected2 = deblend_sources(image, segm, 5, contrast=0.1,
                                    contrast_method='saddle')
    assert_equal(result.data, expected.data)
    assert_equal(result2.data, expected2.data)


def test_nonposmin_fallback_sinh():
    """
    Test that the non-positive-minimum fallback uses sinh mode.

    The sinh spacing keeps the threshold levels concentrated near the
    source minimum, recovering a faint companion of a bright source that
    the linear spacing misses.
    """
    y, x = np.mgrid[0:61, 0:81]
    data = (Gaussian2D(1000, 36, 30, 1.7, 1.7)(x, y)
            + Gaussian2D(20, 44, 30, 1.7, 1.7)(x, y) - 1.2)
    segm = detect_sources(data, -0.2, 5)
    assert segm.n_labels == 1

    match = 'The deblending mode of one or more source labels'
    with pytest.warns(DeblendWarning, match=match):
        result = deblend_sources(data, segm, 5, mode='exponential',
                                 contrast=1e-6)
    assert result.n_labels == 2
    assert_equal(result.info['nonposmin_labels'], [1])
    bit = SEGMENTATION_FLAGS.DEBLEND_NONPOSMIN
    assert_equal(result.flags & bit, [bit, bit])

    # The linear mode does not recover this companion, so the sinh
    # fallback is a real sensitivity improvement over the previous
    # linear fallback.
    result_linear = deblend_sources(data, segm, 5, mode='linear',
                                    contrast=1e-6)
    assert result_linear.n_labels == 1


def test_python_path_connectivity_mismatch():
    """
    Test that the pure-Python reference path raises the same error
    as the compiled contrast loop when the detection and deblending
    connectivities differ.
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
    # Both remaining sources fell back (both have negative minima).
    # Neither splits, so each output label carries the fallback bit
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

    def mock_make_markers():
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
