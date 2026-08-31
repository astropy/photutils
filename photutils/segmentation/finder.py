# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for detecting sources in an image.
"""

import numpy as np

from photutils.segmentation.deblend import (_validate_deblend_kwargs,
                                            deblend_sources)
from photutils.segmentation.detect import detect_sources
from photutils.utils._deprecation import (deprecated_getattr,
                                          deprecated_positional_kwargs,
                                          deprecated_renamed_argument)
from photutils.utils._parameters import as_pair
from photutils.utils._repr import make_repr

__all__ = ['SourceFinder']

# Remove in 4.0
_FINDER_DEPRECATED_ATTRIBUTES = {
    'npixels': 'n_pixels',
    'nlevels': 'n_levels',
    'nproc': 'n_processes',
}


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
    order to deblend sources, they must be separated enough that there
    is a saddle point between them.

    Parameters
    ----------
    n_pixels : int or array_like of 2 int
        The minimum number of connected pixels, each greater than a
        specified threshold, that an object must have to be detected. If
        ``n_pixels`` is an integer, then the value will be used for both
        source detection and deblending (which internally uses source
        detection at multiple thresholds). If ``n_pixels`` contains two
        values, then the first value will be used for source detection
        and the second value used for source deblending. ``n_pixels``
        values must be positive integers.

    connectivity : {4, 8}, optional
        The type of pixel connectivity used in determining how pixels
        are grouped into a detected source. The options are 4 or
        8 (default). 4-connected pixels touch along their edges.
        8-connected pixels touch along their edges or corners.

    deblend : bool, optional
        Whether to deblend overlapping sources.

    n_levels : int, optional
        The number of multi-thresholding levels to use for deblending.
        Each source will be re-thresholded at ``n_levels`` levels spaced
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

    contrast_method : {`None`, 'basin', 'saddle'}, optional
        The flux used by the contrast criterion. For ``'basin'``, the
        fraction is the total flux in the source's watershed basin.
        For ``'saddle'``, the fraction is the flux the source holds
        above the saddle level where it separates from its neighbors.
        If `None` (default), the ``'basin'`` method is currently used.
        The default may change to ``'saddle'`` in version 4.0. See
        :func:`~photutils.segmentation.deblend_sources` for details.
        This keyword is ignored unless ``deblend=True``.

    mode : {'exponential', 'linear', 'sinh'}, optional
        The mode used in defining the spacing between the
        multi-thresholding levels (see the ``n_levels`` keyword)
        during deblending. The ``'exponential'`` and ``'sinh'`` modes
        have more threshold levels near the source minimum and less
        near the source maximum. The ``'linear'`` mode evenly spaces
        the threshold levels between the source minimum and maximum.
        The ``'exponential'`` and ``'sinh'`` modes differ in that
        the ``'exponential'`` levels are dependent on the source
        maximum/minimum ratio (smaller ratios are more linear and
        larger ratios are more exponential), while the ``'sinh'`` levels
        are not. Unlike the ``'exponential'`` mode, the ``'sinh'``
        and ``'linear'`` modes are well defined for sources with
        non-positive minimum data values. For such sources, the
        ``'exponential'`` mode will be changed to ``'sinh'``. This
        keyword is ignored unless ``deblend=True``.

    relabel : bool, optional
        If `True` (default), then the segmentation image will be
        relabeled after deblending such that the labels are in
        consecutive order starting from 1. This keyword is ignored
        unless ``deblend=True``.

    n_threads : int, optional
        The number of threads to use to deblend the sources. The
        default is 1 (no multithreading). When ``n_threads`` > 1,
        the sources are divided into chunks and processed
        concurrently, producing results identical to the
        single-threaded computation. This keyword is ignored unless
        ``deblend=True``.

    nproc : int, optional
        This keyword is deprecated and has no effect. It was the name of
        the ``n_processes`` keyword before version 3.0.

        .. deprecated:: 3.0
            The ``nproc`` keyword is deprecated and will be removed in
            version 4.0.

    n_processes : int, optional
        This keyword is deprecated and has no effect. Multiprocessing
        no longer provides any benefit for source deblending. Use the
        ``n_threads`` keyword instead.

        .. deprecated:: 3.1
            The ``n_processes`` keyword is deprecated and will be
            removed in version 4.0.

    progress_bar : bool, optional
        This keyword is deprecated and has no effect. Deblending no
        longer displays a progress bar.

        .. deprecated:: 3.1
            The ``progress_bar`` keyword is deprecated and will be
            removed in version 4.0.

    See Also
    --------
    :func:`photutils.segmentation.detect_sources`
    :func:`photutils.segmentation.deblend_sources`

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from astropy.convolution import convolve
        from astropy.visualization import simple_norm
        from photutils.background import Background2D, MedianBackground
        from photutils.datasets import make_100gaussians_image
        from photutils.segmentation import (SourceFinder,
                                            make_2dgaussian_kernel)

        # Make a simulated image
        data = make_100gaussians_image()

        # Estimate the background using Background2D and subtract it
        bkg_estimator = MedianBackground()
        bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                           bkg_estimator=bkg_estimator)
        data -= bkg.background  # subtract the background

        # Convolve the data
        kernel = make_2dgaussian_kernel(3.0, size=5)
        convolved_data = convolve(data, kernel)

        # Detect the sources
        threshold = 1.5 * bkg.background_rms  # per-pixel detection threshold
        finder = SourceFinder(n_pixels=10)
        segment_map = finder(convolved_data, threshold)

        # Plot the image and the segmentation image
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 10))
        norm = simple_norm(data, 'sqrt', percent=99.5)
        ax1.imshow(data, norm=norm, origin='lower')
        segment_map.imshow(ax=ax2)
        fig.tight_layout()
    """

    @deprecated_renamed_argument('npixels', 'n_pixels', '3.0', until='4.0')
    @deprecated_renamed_argument('nlevels', 'n_levels', '3.0', until='4.0')
    @deprecated_renamed_argument('nproc', None, '3.0', until='4.0')
    @deprecated_renamed_argument('n_processes', None, '3.1', until='4.0')
    @deprecated_renamed_argument('progress_bar', None, '3.1', until='4.0')
    def __init__(self, n_pixels, *, connectivity=8, deblend=True, n_levels=32,
                 contrast=0.001, contrast_method=None,
                 mode='exponential', relabel=True,
                 n_threads=1,
                 nproc=1,  # noqa: ARG002
                 n_processes=1, progress_bar=True):
        self.n_pixels = as_pair('n_pixels', n_pixels, check_odd=False)
        for name, value in (('deblend', deblend), ('relabel', relabel)):
            if not isinstance(value, (bool, np.bool_)):
                msg = f'{name} must be a boolean, got {value!r}'
                raise TypeError(msg)
        _validate_deblend_kwargs(n_levels=n_levels, contrast=contrast,
                                 contrast_method=contrast_method,
                                 mode=mode, connectivity=connectivity,
                                 n_threads=n_threads)
        self.deblend = deblend
        self.connectivity = connectivity
        self.n_levels = n_levels
        self.contrast = contrast
        self.contrast_method = contrast_method
        self.mode = mode
        self.relabel = relabel
        self.n_threads = n_threads
        self.n_processes = n_processes
        self.progress_bar = progress_bar

    def __repr__(self):
        params = ('n_pixels', 'deblend', 'connectivity', 'n_levels',
                  'contrast', 'contrast_method', 'mode', 'relabel',
                  'n_threads', 'n_processes', 'progress_bar')
        return make_repr(self, params)

    # Remove in 4.0
    def __getattr__(self, name):
        return deprecated_getattr(self, name,
                                  _FINDER_DEPRECATED_ATTRIBUTES,
                                  since='3.0', until='4.0')

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __call__(self, data, threshold, mask=None):
        """
        Detect sources, including deblending, in an image using
        segmentation.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The 2D array from which to detect sources. Typically, this
            array should be an image that has been convolved with a
            smoothing kernel.

        threshold : 2D `~numpy.ndarray` or float
            The data value or pixel-wise data values (as an array) to be
            used as the per-pixel detection threshold. If ``data`` is
            a `~astropy.units.Quantity` array, then ``threshold`` must
            have the same units as ``data``. A 2D ``threshold`` array must
            have the same shape as ``data``.

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
        segment_img = detect_sources(data, threshold, self.n_pixels[0],
                                     mask=mask, connectivity=self.connectivity)
        if segment_img is None:
            return None

        if self.deblend:
            segment_img = deblend_sources(data, segment_img, self.n_pixels[1],
                                          n_levels=self.n_levels,
                                          contrast=self.contrast,
                                          contrast_method=(
                                              self.contrast_method),
                                          mode=self.mode,
                                          connectivity=self.connectivity,
                                          relabel=self.relabel,
                                          n_threads=self.n_threads)

        return segment_img
