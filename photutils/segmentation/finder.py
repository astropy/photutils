# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for detecting sources in an image.
"""

from photutils.segmentation.deblend import deblend_sources
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
    order to deblend sources, there must be a saddle between them.

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

    mode : {'exponential', 'linear', 'sinh'}, optional
        The mode used in defining the spacing between the
        multi-thresholding levels (see the ``n_levels`` keyword)
        during deblending. The ``'exponential'`` and ``'sinh'`` modes
        have more threshold levels near the source minimum and less
        near the source maximum. The ``'linear'`` mode evenly spaces
        the threshold levels between the source minimum and maximum.
        The ``'exponential'`` and ``'sinh'`` modes differ in that
        the ``'exponential'`` levels are dependent on the source
        maximum/minimum ratio (smaller ratios are more linear; larger
        ratios are more exponential), while the ``'sinh'`` levels
        are not. Also, the ``'exponential'`` mode will be changed to
        ``'linear'`` for sources with non-positive minimum data values.
        This keyword is ignored unless ``deblend=True``.

    relabel : bool, optional
        If `True` (default), then the segmentation image will be
        relabeled after deblending such that the labels are in
        consecutive order starting from 1. This keyword is ignored
        unless ``deblend=True``.

    n_processes : int, optional
        The number of processes to use for source deblending. If set to
        1, then a serial implementation is used instead of a parallel
        one. If `None`, then the number of processes will be set to the
        number of CPUs detected on the machine. Please note that due to
        overheads, multiprocessing may be slower than serial processing
        if only a small number of sources are to be deblended. The
        benefits of multiprocessing require ~1000 or more sources to
        deblend, with larger gains as the number of sources increase.
        This keyword is ignored unless ``deblend=True``.

    progress_bar : bool, optional
        Whether to display a progress bar. If ``n_processes = 1``, then the
        ID shown after the progress bar is the source label being
        deblended. If multiprocessing is used (``n_processes > 1``), the
        ID shown is the last source label that was deblended. The
        progress bar requires that the `tqdm <https://tqdm.github.io/>`_
        optional dependency be installed. This keyword is ignored unless
        ``deblend=True``.

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
        finder = SourceFinder(n_pixels=10, progress_bar=False)
        segment_map = finder(convolved_data, threshold)

        # Plot the image and the segmentation image
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        norm = simple_norm(data, 'sqrt', percent=99.5)
        ax1.imshow(data, norm=norm, origin='lower')
        segment_map.imshow(ax=ax2)
        fig.tight_layout()
    """

    @deprecated_renamed_argument('npixels', 'n_pixels', '3.0', until='4.0')
    @deprecated_renamed_argument('nlevels', 'n_levels', '3.0', until='4.0')
    @deprecated_renamed_argument('nproc', 'n_processes', '3.0', until='4.0')
    def __init__(self, n_pixels, *, connectivity=8, deblend=True, n_levels=32,
                 contrast=0.001, mode='exponential', relabel=True,
                 n_processes=1, progress_bar=True):
        self.n_pixels = as_pair('n_pixels', n_pixels, check_odd=False)
        self.deblend = deblend
        self.connectivity = connectivity
        self.n_levels = n_levels
        self.contrast = contrast
        self.mode = mode
        self.relabel = relabel
        self.n_processes = n_processes
        self.progress_bar = progress_bar

    def __repr__(self):
        params = ('n_pixels', 'deblend', 'connectivity', 'n_levels',
                  'contrast', 'mode', 'relabel', 'n_processes',
                  'progress_bar')
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

        # Source deblending requires scikit-image
        if self.deblend:
            segment_img = deblend_sources(data, segment_img, self.n_pixels[1],
                                          n_levels=self.n_levels,
                                          contrast=self.contrast,
                                          mode=self.mode,
                                          connectivity=self.connectivity,
                                          relabel=self.relabel,
                                          n_processes=self.n_processes,
                                          progress_bar=self.progress_bar)

        return segment_img
