# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import copy
import numpy as np
from astropy.table import Table, Column
from skimage.measure._regionprops import _RegionProperties

__all__ = ['segment_props', 'segment_photometry']


class _SegmentProperties(object):
    def __init__(self, image, segment_image, label, slice):
        self.label = label
        self._slice = slice
        self._label_image = segment_image
        self._intensity_image = image
        self._cache_active = True
    area = _RegionProperties.area
    bbox = _RegionProperties.bbox
    _centroid = _RegionProperties.centroid
    xcen, ycen = _centroid
    local_centroid = _RegionProperties.local_centroid
    moments = _RegionProperties.moments
    _image_double = _RegionProperties._image_double
    image = _RegionProperties.image



def segment_props(image, segment_image, mask_image=None,
                  mask_method='exclude', labels=None):
    """
    Parameters
    ----------

    Returns
    -------

    xcen, ycen: centroids
    area
    radius_equivalent
    major/minor axis
    eccen
    theta
    2nd order central moments

    """

    from scipy import ndimage
    objects = ndimage.find_objects(segment_image)
    objpropslist = []
    for i, sl in enumerate(objects):
        if sl is None:
            continue
        label = i + 1
        cache = True
        #objprops = _RegionProperties(sl, label, segment_image, image, cache)
        objprops = _SegmentProperties(image, segment_image, label, sl)
        objpropslist.append(objprops)

    #props = ['centroid', 'area', 'radius']
    props = ['_centroid', 'area', 'bbox', 'xcen', 'ycen']
    props_table = Table()
    for prop in props:
        data = [getattr(objprops, prop) for objprops in objpropslist]
        props_table[prop] = Column(data)
    return props_table


def segment_photometry(image, segment_image, error_image=None, gain=None,
                       mask_image=None, mask_method='exclude', labels=None):
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

    mask_method : {'exclude', 'interpolate'}, optional
        Method used to treat masked pixels.  The currently supported
        methods are:

        'exclude'
            Exclude masked pixels from all calculations.  This is the
            default.

        'interpolate'
            The value of masked pixels are replaced by the mean value of
            the neighboring non-masked pixels.

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

    from scipy import ndimage
    assert image.shape == segment_image.shape, \
        ('image and segment_image must have the same shape')
    if labels is None:
        objids = np.unique(segment_image[segment_image > 0])
    else:
        objids = np.atleast_1d(labels)

    image_iscopy = False
    if background is not None:
        if not np.isscalar(background):
            assert image.shape == background.shape, \
                ('image and background image must have the same shape')
        image = copy.deepcopy(image)
        image_iscopy = True
        image -= background

    if error_image is not None:
        assert image.shape == mask_image.shape, \
            ('image and error_image must have the same shape')
        variance_image = error_image**2

    if mask_image is not None:
        assert image.shape == mask_image.shape, \
            ('image and mask_image must have the same shape')
        if not image_iscopy:
            image = copy.deepcopy(image)

        if mask_method == 'exclude':
            image[mask_image.nonzero()] = 0.0
            if error_image is not None:
                error_image[mask_image.nonzero()] = 0.0
        elif mask_method == 'interpolate':
            for j, i in zip(*mask_image.nonzero()):
                y0, y1 = max(j - 1, 0), min(j + 2, image.shape[0])
                x0, x1 = max(i - 1, 0), min(i + 2, image.shape[1])
                goodpix = ~mask_image[y0:y1, x0:x1]
                image[j, i] = np.mean(image[y0:y1, x0:x1][goodpix])
                if error_image is not None:
                    error_image[j, i] = np.sqrt(np.mean(
                        variance_image[y0:y1, x0:x1][goodpix]))
        else:
            raise ValueError(
                'mask_method {0} is not valid'.format(mask_method))

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
        if gain is not None:
            variance_image += image / gain
        flux_variance = ndimage.measurements.sum(variance_image,
                                               labels=segment_image,
                                               index=objids)
        flux_errors = np.sqrt(flux_variance)
        phot_table['flux_error'] = flux_errors

    bad_labels = (~np.isfinite(phot_table['xcen'])).nonzero()
    phot_table['flux'][bad_labels] = np.nan
    return phot_table
