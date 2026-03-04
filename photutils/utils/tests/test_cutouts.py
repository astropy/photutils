# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the cutouts module.
"""

import numpy as np
import pytest
from astropy.nddata import PartialOverlapError
from numpy.testing import assert_equal

from photutils.aperture import BoundingBox
from photutils.datasets import make_100gaussians_image
from photutils.utils.cutouts import CutoutImage, _make_cutouts


class TestCutoutImage:
    """
    Tests for the CutoutImage class.
    """

    def test_cutout(self):
        """
        Test CutoutImage with basic parameters.
        """
        data = make_100gaussians_image()
        shape = (24, 57)
        yxpos = (100, 51)
        cutout = CutoutImage(data, yxpos, shape)
        assert cutout.position == yxpos
        assert cutout.input_shape == shape
        assert cutout.mode == 'trim'
        assert np.isnan(cutout.fill_value)
        assert not cutout.copy

        assert cutout.data.shape == shape
        assert_equal(cutout.__array__(), cutout.data)

        assert isinstance(cutout.bbox_original, BoundingBox)
        assert isinstance(cutout.bbox_cutout, BoundingBox)
        assert cutout.slices_original == (slice(88, 112, None),
                                          slice(23, 80, None))
        assert cutout.slices_cutout == (slice(0, 24, None),
                                        slice(0, 57, None))

        assert_equal(cutout.xyorigin, np.array((23, 88)))

        cutouts2 = CutoutImage(data, yxpos, np.array(shape))
        assert cutouts2.input_shape == shape

        assert 'CutoutImage(' in repr(cutout)
        assert f'shape={shape}' in repr(cutout)
        assert f'Shape: {shape}' in str(cutout)

    def test_cutout_partial_overlap(self):
        """
        Test CutoutImage with partial overlap modes.
        """
        data = make_100gaussians_image()
        shape = (24, 57)

        # 'trim' mode
        cutout = CutoutImage(data, (11, 10), shape)
        assert cutout.input_shape == shape
        assert cutout.shape == (23, 39)

        # 'strict' mode
        match = 'Arrays overlap only partially'
        with pytest.raises(PartialOverlapError, match=match):
            CutoutImage(data, (11, 10), shape, mode='strict')

        # 'partial' mode
        cutout = CutoutImage(data, (11, 10), shape, mode='partial')
        assert cutout.input_shape == shape
        assert cutout.shape == shape

        assert (cutout.bbox_original
                == BoundingBox(ixmin=0, ixmax=39, iymin=0, iymax=23))
        assert (cutout.bbox_cutout
                == BoundingBox(ixmin=18, ixmax=57, iymin=1, iymax=24))

        assert cutout.slices_original == (slice(0, 23, None),
                                          slice(0, 39, None))
        assert cutout.slices_cutout == (slice(1, 24, None),
                                        slice(18, 57, None))

        assert_equal(cutout.xyorigin, np.array((-18, -1)))

        # Regression test for xyorgin in partial mode when cutout extends
        # beyond right or top edge
        data = make_100gaussians_image()
        shape = (54, 57)
        cutout = CutoutImage(data, (281, 485), shape, mode='partial')

        assert_equal(cutout.xyorigin, np.array((457, 254)))
        assert (cutout.bbox_original
                == BoundingBox(ixmin=457, ixmax=500, iymin=254, iymax=300))
        assert (cutout.bbox_cutout
                == BoundingBox(ixmin=0, ixmax=43, iymin=0, iymax=46))
        assert cutout.slices_original == (slice(254, 300, None),
                                          slice(457, 500, None))
        assert cutout.slices_cutout == (slice(0, 46, None),
                                        slice(0, 43, None))

    def test_cutout_copy(self):
        """
        Test CutoutImage with copy=True and copy=False.
        """
        data = make_100gaussians_image()

        cutout1 = CutoutImage(data, (1, 1), (3, 3), copy=True)
        cutout1.data[0, 0] = np.nan
        assert not np.isnan(data[0, 0])

        cutout2 = CutoutImage(data, (1, 1), (3, 3), copy=False)
        cutout2.data[0, 0] = np.nan
        assert np.isnan(data[0, 0])


class TestMakeCutouts:
    """
    Tests for the _make_cutouts utility function.
    """

    def setup_method(self):
        self.data = np.arange(100, dtype=float).reshape(10, 10)

    def test_fully_inside(self):
        """
        Test a source fully inside the image.
        """
        xpos = np.array([5.0])
        ypos = np.array([5.0])
        cutouts, mask = _make_cutouts(self.data, xpos, ypos, (3, 3))
        assert cutouts.shape == (1, 3, 3)
        np.testing.assert_array_equal(cutouts[0], self.data[4:7, 4:7])
        assert mask[0].all()

    def test_partial_overlap_corners(self):
        """
        Test sources at image corners that partially overlap.
        """
        xpos = np.array([0.0, 9.0])
        ypos = np.array([0.0, 9.0])
        _, mask = _make_cutouts(self.data, xpos, ypos, (5, 5))

        # Corner (0, 0): top-left 2 rows and 2 cols are outside
        assert not mask[0].all()  # not fully inside
        assert mask[0].any()  # not fully outside
        assert not mask[0, 0, 0]  # outside pixel
        assert mask[0, 2, 2]  # center pixel (the position itself)

        # Corner (9, 9): bottom-right 2 rows and 2 cols are outside
        assert not mask[1].all()
        assert mask[1].any()
        assert mask[1, 2, 2]  # center pixel
        assert not mask[1, 4, 4]  # outside pixel

    def test_no_overlap(self):
        """
        Test a source completely outside the image.
        """
        xpos = np.array([-10.0])
        ypos = np.array([-10.0])
        cutouts, mask = _make_cutouts(self.data, xpos, ypos, (3, 3))
        assert not mask[0].any()
        assert np.all(cutouts[0] == 0.0)

    def test_fill_value_nan(self):
        """
        Test that fill_value=NaN fills out-of-bounds pixels with NaN.
        """
        xpos = np.array([0.0])
        ypos = np.array([0.0])
        cutouts, mask = _make_cutouts(self.data, xpos, ypos, (5, 5),
                                      fill_value=np.nan)
        # Outside pixels should be NaN
        assert np.all(np.isnan(cutouts[0][~mask[0]]))
        # Inside pixels should not be NaN
        assert np.all(np.isfinite(cutouts[0][mask[0]]))

    def test_fill_value_custom(self):
        """
        Test that a custom fill_value is used for out-of-bounds pixels.
        """
        xpos = np.array([0.0])
        ypos = np.array([0.0])
        cutouts, mask = _make_cutouts(self.data, xpos, ypos, (3, 3),
                                      fill_value=-99.0)
        assert np.all(cutouts[0][~mask[0]] == -99.0)

    def test_overlap_mask_dtype(self):
        """
        Test that overlap_mask is a boolean array.
        """
        xpos = np.array([5.0])
        ypos = np.array([5.0])
        _, mask = _make_cutouts(self.data, xpos, ypos, (3, 3))
        assert mask.dtype == bool

    def test_mixed_sources(self):
        """
        Test a mix of fully-inside, partial, and outside sources.
        """
        xpos = np.array([5.0, 0.0, -10.0])
        ypos = np.array([5.0, 0.0, -10.0])
        _, mask = _make_cutouts(self.data, xpos, ypos, (3, 3))

        # Fully inside
        assert mask[0].all()
        # Partial overlap
        assert mask[1].any()
        assert not mask[1].all()
        # No overlap
        assert not mask[2].any()

    def test_even_shaped_cutout(self):
        """
        Test _make_cutouts with an even-shaped cutout.
        """
        xpos = np.array([5.0])
        ypos = np.array([5.0])
        cutouts, mask = _make_cutouts(self.data, xpos, ypos, (4, 4))
        assert cutouts.shape == (1, 4, 4)
        assert mask[0].all()  # fully inside
        # Half-widths: hy=2, hx=2; cutout rows [3..6], cols [3..6]
        expected = self.data[3:7, 3:7]
        np.testing.assert_array_equal(cutouts[0], expected)

    def test_even_shaped_cutout_at_edge(self):
        """
        Test _make_cutouts with an even-shaped cutout at the image edge.
        """
        xpos = np.array([0.0])
        ypos = np.array([0.0])
        cutouts, mask = _make_cutouts(self.data, xpos, ypos, (4, 4))
        assert cutouts.shape == (1, 4, 4)
        # Some pixels should be outside
        assert not mask[0].all()
        assert mask[0].any()
        # Outside pixels should be zero (default fill_value)
        assert np.all(cutouts[0][~mask[0]] == 0.0)

    def test_data_not_2d(self):
        """
        Test that a non-2D data array raises ValueError.
        """
        match = 'data must be a 2D array'
        with pytest.raises(ValueError, match=match):
            _make_cutouts(np.ones(10), np.array([5.0]),
                          np.array([5.0]), (3, 3))

    def test_xpos_not_1d(self):
        """
        Test that non-1D xpos/ypos arrays raise ValueError.
        """
        match = 'xpos and ypos must be 1D arrays'
        with pytest.raises(ValueError, match=match):
            _make_cutouts(self.data, np.ones((2, 2)),
                          np.array([5.0]), (3, 3))

    def test_cutout_shape_wrong_length(self):
        """
        Test that cutout_shape with != 2 elements raises ValueError.
        """
        match = 'cutout_shape must have exactly 2 elements'
        with pytest.raises(ValueError, match=match):
            _make_cutouts(self.data, np.array([5.0]),
                          np.array([5.0]), (3, 3, 3))

    def test_xpos_ypos_length_mismatch(self):
        """
        Test that mismatched xpos/ypos lengths raise ValueError.
        """
        match = 'xpos and ypos must have the same length'
        with pytest.raises(ValueError, match=match):
            _make_cutouts(self.data, np.array([5.0, 6.0]),
                          np.array([5.0]), (3, 3))
