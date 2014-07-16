# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from scipy import ndimage
from astropy.table import Table

__all__ = ['segment_photometry']


def segment_photometry(image, segment_image):
    """
    Perform photometry on using a labeled segmentation image.  For
    example, this can be used to perform isophotal photometry.

    Parameters
    ----------
    image : array_like
        The 2D array of the image.

    segment_image : array_like
        A 2D segmentation image of integers indicating segment labels.

    Returns
    -------
    table : `astropy.table.Table`
        A table of the segmented photometry.
    """

    assert image.shape == segment_image.shape, \
        ('image and segment_image must have the same shape')
    idx = np.arange(np.max(segment_image)) + 1
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
