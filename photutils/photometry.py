# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from scipy import ndimage
from astropy.table import Table

__all__ = ['segment_photometry']


def segment_photometry(image, segment_image, labels=None):
    """
    Perform photometry using a labeled segmentation image.  This can be
    used to perform isophotal photometry when ``segment_image`` is
    defined using a thresholded flux level (e.g., see `detect_sources`).

    Parameters
    ----------
    image : array_like
        The 2D array of the image.

    segment_image : array_like
        A 2D segmentation image of positive integers indicating labels
        for detected sources.  A value of zero is reserved for the
        background.

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
          object label in the ``segment_image``
        * ``xcen, ycen``: object centroid (zero-based origin)
        * ``area``: the number pixels in the source segment
        * ``radius_equiv``: the equivalent circular radius derived from the
          source ``area``
        * ``flux``: the total flux within the source segment

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
    bad_labels = (~np.isfinite(phot_table['xcen'])).nonzero()
    phot_table['flux'][bad_labels] = np.nan   # change flux from 0 to np.nan
    return phot_table
