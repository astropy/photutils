# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for deprecation warnings when optional arguments are passed
positionally.
"""

import numpy as np
import pytest
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.segmentation.core import SegmentationImage
from photutils.segmentation.utils import make_2dgaussian_kernel


class TestSegmentationImagePositionalKwargs:
    """
    Test SegmentationImage methods warn for positional optional args.
    """

    def setup_method(self):
        self.data = np.array([[1, 1, 0, 0, 2, 2],
                              [1, 1, 0, 0, 2, 2],
                              [0, 0, 3, 3, 0, 0],
                              [0, 0, 3, 3, 0, 0]])

    def test_reassign_label_positional_warns(self):
        segm = SegmentationImage(self.data.copy())
        match = 'reassign_label'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            segm.reassign_label(1, 4, True)  # noqa: FBT003

    def test_reassign_label_keyword_no_warning(self):
        segm = SegmentationImage(self.data.copy())
        segm.reassign_label(1, 4, relabel=True)

    def test_keep_label_positional_warns(self):
        segm = SegmentationImage(self.data.copy())
        match = 'keep_label'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            segm.keep_label(1, True)  # noqa: FBT003

    def test_keep_label_keyword_no_warning(self):
        segm = SegmentationImage(self.data.copy())
        segm.keep_label(1, relabel=True)

    def test_remove_label_positional_warns(self):
        segm = SegmentationImage(self.data.copy())
        match = 'remove_label'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            segm.remove_label(1, True)  # noqa: FBT003

    def test_remove_label_keyword_no_warning(self):
        segm = SegmentationImage(self.data.copy())
        segm.remove_label(1, relabel=True)

    def test_remove_border_labels_positional_warns(self):
        segm = SegmentationImage(self.data.copy())
        match = 'remove_border_labels'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            segm.remove_border_labels(1, True)  # noqa: FBT003

    def test_remove_border_labels_keyword_no_warning(self):
        segm = SegmentationImage(self.data.copy())
        segm.remove_border_labels(1, partial_overlap=True)

    def test_remove_masked_labels_positional_warns(self):
        segm = SegmentationImage(self.data.copy())
        mask = np.zeros(self.data.shape, dtype=bool)
        mask[0, 0] = True
        match = 'remove_masked_labels'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            segm.remove_masked_labels(mask, True)  # noqa: FBT003

    def test_remove_masked_labels_keyword_no_warning(self):
        segm = SegmentationImage(self.data.copy())
        mask = np.zeros(self.data.shape, dtype=bool)
        mask[0, 0] = True
        segm.remove_masked_labels(mask, partial_overlap=True)

    def test_make_cutout_positional_warns(self):
        segm = SegmentationImage(self.data.copy())
        segment = segm.segments[0]
        data = np.random.default_rng(0).random(self.data.shape)
        match = 'make_cutout'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            segment.make_cutout(data, True)  # noqa: FBT003

    def test_make_cutout_keyword_no_warning(self):
        segm = SegmentationImage(self.data.copy())
        segment = segm.segments[0]
        data = np.random.default_rng(0).random(self.data.shape)
        segment.make_cutout(data, masked_array=True)


class TestMake2DGaussianKernelPositionalKwargs:
    """
    Test make_2dgaussian_kernel warns for positional optional args.
    """

    def test_positional_warns(self):
        match = 'make_2dgaussian_kernel'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            make_2dgaussian_kernel(3.0, 5, 'oversample')

    def test_keyword_no_warning(self):
        make_2dgaussian_kernel(3.0, 5, mode='oversample')
