# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import copy
import numpy as np
from scipy import ndimage
from astropy.table import Table

__all__ = ['segment_photometry']


def segment_photometry(image, segment_image, error_image=None,
                       mask_image=None, background=None, labels=None):
    """
    Perform photometry of sources whose extents are defined by a labeled
    segmentation image.

    When the segmentation image is defined using a thresholded flux
    level (e.g., see `detect_sources`), this is equivalent to performing
    isophotal photometry.

    Parameters
    ----------
    image : array_like
        The 2D array of the image.

    segment_image : array_like
        A 2D segmentation image of positive integers indicating labels
        for detected sources.  A value of zero is reserved for the
        background.

    error_image : array_like, optional
        The 2D array of the 1-sigma errors of the input ``image``.  If
        ``gain`` is input, then ``error_image`` should include all
        sources of "background" error but *exclude* the Poission error
        of the sources.  If ``gain`` is `None`, then the ``error_image``
        is assumed to include *all* sources of error, including the
        Poission error of the sources.  ``error_image`` must have the
        same shape as ``image``.

    gain : float, optional
        The gain factor that when multiplied by the input ``image``
        results in an image in units of electrons.  If ``gain`` is input,
        then ``error_image`` should include all sources of "background"
        error but *exclude* the Poission error of the sources.  If
        ``gain`` is `None`, then the ``error_image`` is assumed to include
        *all* sources of error, including the Poission error of the
        sources.

        For example, if your input ``image`` is in units of ADU, then
        ``gain`` should represent electrons/ADU.  If your input
        ``image`` is in units of electrons/s then ``gain`` should be the
        exposure time.

    mask_image : array_like, bool, optional
        A boolean mask with the same shape as ``image``, where a `True`
        value indicates the corresponding element of ``image`` is
        ignored when computing the photometry.

    background : float or array_like, optional
        The background level of the input ``image``.  ``background`` may
        either be a scalar value or a 2D image with the same shape as
        the input ``image``.  If the input ``image`` has been
        background-subtracted, then set ``background=None`` (default).

    labels : int, sequence of ints or None
        Subset of ``segment_image`` labels for which to perform the
        photometry.  If `None`, then photometry will be performed for
        all source segments.

    Returns
    -------
    table : `astropy.table.Table`
        A table of the segmented photometry containing the following
        parameters:

        * ``id``: the source identification number corresponding to the
          object label in the ``segment_image``.
        * ``xcen, ycen``: object centroid (zero-based origin).
        * ``area``: the number pixels in the source segment.
        * ``radius_equiv``: the equivalent circular radius derived from the
          source ``area``.
        * ``flux``: the total flux within the source segment.
        * ``flux_error``: the 1-sigma flux error within the source segment.
          ``flux_error`` is returned only if `error_image` is input.

    See Also
    --------
    detect_sources
    """

    assert image.shape == segment_image.shape, \
        ('image and segment_image must have the same shape')
    if labels is None:
        objids = np.unique(segment_image[segment_image > 0])
    else:
        objids = np.atleast_1d(labels)

    if mask_image is not None:
        assert image.shape == mask_image.shape, \
            ('image and mask_image must have the same shape')
        image = copy.deepcopy(image)
        image[mask_image.nonzero()] = 0.0
        if error_image is not None:
            assert image.shape == mask_image.shape, \
                ('image and error_image must have the same shape')
            error_image = copy.deepcopy(error_image)
            error_image[mask_image.nonzero()] = 0.0

    # TODO:  allow alternate centroid methods via input centroid_func:
    # npix = ndimage.labeled_comprehension(image, segment_image, objids,
    #                                      centroid_func, np.float32, np.nan)
    centroids = ndimage.center_of_mass(image, segment_image, objids)
    ycen, xcen = np.transpose(centroids)
    npix = ndimage.labeled_comprehension(image, segment_image, objids, len,
                                         np.float32, np.nan)
    radii = np.sqrt(npix / np.pi)
    fluxes = ndimage.measurements.sum(image, labels=segment_image,
                                      index=objids)
    data = [objids, xcen, ycen, npix, radii, fluxes]
    names = ('id', 'xcen', 'ycen', 'area', 'radius_equiv', 'flux')
    phot_table = Table(data, names=names)

    if error_image is not None:
        variance_image = error_image**2
        if gain is not None:
            variance_image += gain / image
        flux_variance = ndimage.measurements.sum(variance_image,
                                               labels=segment_image,
                                               index=objids)
        flux_errors = np.sqrt(flux_variance)
        phot_table['flux_error'] = flux_errors

    bad_labels = (~np.isfinite(phot_table['xcen'])).nonzero()
    phot_table['flux'][bad_labels] = np.nan
    return phot_table
