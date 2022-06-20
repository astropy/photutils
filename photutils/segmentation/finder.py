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

    This is a convenience class that combines the functionality
    of `~photutils.segmentation.detect_sources` and
    `~photutils.segmentation.deblend_sources`.

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

    progress_bar : bool, optional
       Whether to display a progress bar during source deblending. The
       progress bar requires that the `tqdm <https://tqdm.github.io/>`_
       optional dependency be installed. Note that the progress bar does
       not currently work in the Jupyter console due to limitations in
       ``tqdm``. This keyword is ignored unless ``deblend=True``.

    See Also
    --------
    :func:`photutils.segmentation.detect_sources`
    :func:`photutils.segmentation.deblend_sources`

    Examples
    --------
    .. plot::
        :include-source:

        from astropy.convolution import convolve
        from astropy.stats import gaussian_fwhm_to_sigma
        from astropy.visualization import simple_norm
        import matplotlib.pyplot as plt
        from photutils.background import Background2D, MedianBackground
        from photutils.datasets import make_100gaussians_image
        from photutils.segmentation import (SourceFinder,
                                            make_2dgaussian_kernel)

        # make a simulated image
        data = make_100gaussians_image()

        # subtract the background
        bkg_estimator = MedianBackground()
        bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                           bkg_estimator=bkg_estimator)
        data -= bkg.background

        # convolve the data
        kernel = make_2dgaussian_kernel(3., size=5)  # FWHM = 3.
        convolved_data = convolve(data, kernel)

        # detect the sources
        threshold = 1.5 * bkg.background_rms  # per-pixel threshold
        finder = SourceFinder(npixels=10)
        segm = finder(convolved_data, threshold)

        # plot the image and the segmentation image
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        norm = simple_norm(data, 'sqrt', percent=99.)
        ax1.imshow(data, origin='lower', interpolation='nearest',
                   norm=norm)
        ax2.imshow(segm.data, origin='lower', interpolation='nearest',
                   cmap=segm.cmap)
        plt.tight_layout()
    """

    def __init__(self, npixels, *, connectivity=8, deblend=True, nlevels=32,
                 contrast=0.001, mode='exponential', progress_bar=True):
        self.npixels = npixels
        self.deblend = deblend
        self.connectivity = connectivity
        self.nlevels = nlevels
        self.contrast = contrast
        self.mode = mode
        self.progress_bar = progress_bar

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
                                          relabel=True,
                                          progress_bar=self.progress_bar)

        return segment_img
