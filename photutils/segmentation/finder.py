# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for detecting sources in an image.
"""

from .detect import detect_sources
from .deblend import deblend_sources

__all__ = ['SourceFinder']


class SourceFinder:
    """
    Class to detect sources, including deblending, in an image using
    segmentation.

    Sources are deblended using a combination of
    multi-thresholding and `watershed segmentation
    <https://en.wikipedia.org/wiki/Watershed_(image_processing)>`_. In
    order to deblend sources, there must be a saddle between them.

    Parameters
    ----------
    npixels : int
        The number of connected pixels, each greater than a specified
        threshold, that an object must have to be detected. ``npixels``
        must be a positive integer.

    connectivity : {4, 8}, optional
        The type of pixel connectivity used in determining how pixels
        are grouped into a detected source. The options are 4 or
        8 (default). 4-connected pixels touch along their edges.
        8-connected pixels touch along their edges or corners. For
        reference, SourceExtractor uses 8-connected pixels.

    deblend : bool, optional
        Whether to deblend overlapping sources.

    nlevels : int, optional
        The number of multi-thresholding levels to use for deblending.
        Each source will be re-thresholded at ``nlevels`` levels spaced
        exponentially or linearly (see the ``mode`` keyword) between
        its minimum and maximum values. This keyword is ignored unless
        ``deblend=True``.

    contrast : float, optional
        The fraction of the total source flux that a local peak must
        have (at any one of the multi-thresholds) to be deblended
        as a separate object. ``contrast`` must be between 0 and 1,
        inclusive. If ``contrast=0`` then every local peak will be made
        a separate object (maximum deblending). If ``contrast=1`` then
        no deblending will occur. The default is 0.001, which will
        deblend sources with a 7.5 magnitude difference. This keyword is
        ignored unless ``deblend=True``.

    mode : {'exponential', 'linear'}, optional
        The mode used in defining the spacing between the
        multi-thresholding levels (see the ``nlevels`` keyword) during
        deblending. This keyword is ignored unless ``deblend=True``.
    """

    def __init__(self, npixels, *, connectivity=8, deblend=True, nlevels=32,
                 contrast=0.001, mode='exponential'):
        self.npixels = npixels
        self.deblend = deblend
        self.connectivity = connectivity
        self.nlevels = nlevels
        self.contrast = contrast
        self.mode = mode

    def __call__(self, data, threshold, mask=None):
        """
        Detect sources, including deblending, in an image using using
        segmentation.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The 2D array from which to detect sources. Typically, this
            array should be an image that has been convolved with a
            smoothing kernel.

        threshold : 2D `~numpy.ndarray` or float
            The data value or pixel-wise data values (as an array)
            to be used as the per-pixel detection threshold. A 2D
            ``threshold`` array must have the same shape as ``data``.

        mask : 2D bool `~numpy.ndarray`, optional
            A boolean mask with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked pixels will not be included in any source.

        Returns
        -------
        segment_image : `~photutils.segmentation.SegmentationImage` or `None`
            A 2D segmentation image, with the same shape as the input data,
            where sources are marked by different positive integer values. A
            value of zero is reserved for the background. If no sources are
            found then `None` is returned.
        """
        segment_img = detect_sources(data, threshold, self.npixels, mask=mask,
                                     connectivity=self.connectivity)
        if segment_img is None:
            return None

        # source deblending requires scikit-image
        if self.deblend:
            segment_img = deblend_sources(data, segment_img, self.npixels,
                                          nlevels=self.nlevels,
                                          contrast=self.contrast,
                                          mode=self.mode,
                                          connectivity=self.connectivity,
                                          relabel=True)

        return segment_img
