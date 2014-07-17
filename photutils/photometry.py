# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from scipy import ndimage
from astropy.table import Table

__all__ = ['segment_photometry']


def segment_photometry(image, segment_image):
    """
    Perform photometry using a labeled segmentation image.  This can be
    used to perform isophotal photometry when ``segment_image`` is
    defined using a thresholded flux level (e.g., see `detect_sources`).

    Parameters
    ----------
    image : array_like
        The 2D array of the image.

    segment_image : array_like
        A 2D segmentation image of positive integers indicating segment
        labels.  A value of zero is reserved for the background.

    Returns
    -------
    table : `astropy.table.Table`
        A table of the segmented photometry containing the following
        parameters:

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
    idx = np.arange(np.max(segment_image)) + 1
    # TODO:  allow alternate centroid methods
    centroids = ndimage.center_of_mass(image, segment_image, idx)
    ycen, xcen = np.transpose(centroids)
    npix = ndimage.labeled_comprehension(image, segment_image, idx, len,
                                         np.float, np.nan)
    radii = np.sqrt(npix / np.pi)
    fluxes = ndimage.measurements.sum(image, labels=segment_image, index=idx)
    data = [xcen, ycen, npix, radii, fluxes]
    names = ('xcen', 'ycen', 'area', 'radius_equiv', 'flux')
    phot_table = Table(data, names=names)
    return phot_table
