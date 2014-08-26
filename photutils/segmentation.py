# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import copy
import numpy as np
from astropy.table import Table, Column
from skimage.measure._regionprops import _cached_property, _RegionProperties


__all__ = ['segment_props', 'segment_photometry']


class _SegmentProperties(object):
    def __init__(self, image, segment_image, label, slice):
        self.label = label
        self._slice = slice
        self._label_image = segment_image
        self._intensity_image = image
        self._cache_active = True

    image = _RegionProperties.image
    intensity_image = _RegionProperties.intensity_image
    _image_double = _RegionProperties._image_double
    moments = _RegionProperties.moments
    centroid = _RegionProperties.centroid
    local_centroid = _RegionProperties.local_centroid

    area = _RegionProperties.area
    bbox = _RegionProperties.bbox
    min_value = _RegionProperties.min_intensity
    max_value = _RegionProperties.max_intensity

    @_cached_property
    def region(self):
        return self.intensity_image[self.image]

    @_cached_property
    def min_position(self):
        return np.where(self.region == self.min_value)

    @_cached_property
    def max_position(self):
        return np.where(self.region == self.max_value)

    # TODO:  allow alternate centroid methods via input centroid_func:
    # npix = ndimage.labeled_comprehension(image, segment_image, label_ids,
    #                                      centroid_func, np.float32, np.nan)
    #centroids = ndimage.center_of_mass(image, segment_image, objids)
    #ycen, xcen = np.transpose(centroids)
    #npix = ndimage.labeled_comprehension(image, segment_image, objids, len,
    #                                     np.float32, np.nan)
    #radii = np.sqrt(npix / np.pi)



def segment_props(image, segment_image, mask_image=None,
                  mask_method='exclude', labels=None, out_table=False):
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


        * ``'xcen', 'ycen'``: object centroid (zero-based origin).
        * ``'area'``: the number pixels in the source segment.
        * ``'radius_equiv'``: the equivalent circular radius derived
          from the source ``area``.

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

    if not out_table:
        return objpropslist
    else:
        props_table = Table()
        data = [getattr(objprops, 'centroid') for objprops in objpropslist]
        xcen, ycen = np.transpose(data)

        props_table['xmin'] = Column(xmin)
        data = [getattr(objprops, 'bbox') for objprops in objpropslist]
        xmin, ymin, xmax, ymax = np.transpose(data)
        props_table['xmin'] = Column(xmin)
        props_table['xmax'] = Column(xmax)
        props_table['ymin'] = Column(ymin)
        props_table['ymax'] = Column(ymax)

        props = ['area', 'min_value', 'max_value', 'min_position',
                 'max_position']
        for prop in props:
            data = [getattr(objprops, prop) for objprops in objpropslist]
            props_table[prop] = Column(data)
        return props_table


def segment_photometry(data, segment_image, background=None, error=None,
                       gain=None, mask=None, mask_method='exclude',
                       labels=None):
    """
    Perform photometry of sources whose extents are defined by a labeled
    segmentation image.

    When the segmentation image is defined using a thresholded flux
    level (e.g., see `detect_sources`), this is equivalent to performing
    isophotal photometry.

    Parameters
    ----------
    data : array_like
        The 2D array on which to perform photometry.

    segment_image : array_like
        A 2D segmentation image where sources are marked by different
        positive integer values.  A value of zero is reserved for the
        background.

    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``image``.  If
        ``gain`` is input, then ``error`` should include all sources of
        "background" error but *exclude* the Poission error of the
        sources.  If ``gain`` is `None`, then the ``error_image`` is
        assumed to include *all* sources of error, including the
        Poission error of the sources.  ``error`` must have the same
        shape as ``image``.

    gain : float or array-like, optional
        Ratio of counts (e.g., electrons or photons) to the units of
        ``data``.  This is used to calculate the Poisson error of the
        sources.  If ``gain`` is input, then ``error`` should include
        all sources of "background" error but *exclude* the Poission
        error of the sources.  If ``gain`` is `None`, then the ``error``
        is assumed to include *all* sources of error, including the
        Poission error of the sources.  For example, if your input
        ``data`` is in units of ADU, then ``gain`` should represent
        electrons/ADU.  If your input ``data`` is in units of
        electrons/s then ``gain`` should be the exposure time.

    mask : array_like, bool, optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``image`` is masked
        when computing the photometry.  Use the ``mask_method`` keyword
        to select the method used to treat masked pixels.

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
        The background level of the input ``data``.  ``background`` may
        either be a scalar value or a 2D image with the same shape as
        the input ``data``.  If the input ``data`` has been
        background-subtracted, then set ``background`` to `None` (which
        is the default).

    labels : int, sequence of ints or None
        Subset of ``segment_image`` labels for which to perform the
        photometry.  If `None`, then photometry will be performed for
        all source segments.

    Returns
    -------
    table : `astropy.table.Table`
        A table of the photometry of the segmented sources containing
        the following columns:

        * ``'id'``: the source identification number corresponding to
          the object label in the ``segment_image``.
        * ``'segment_sum'``: the total flux within the source segment.
        * ``'segment_sum_errr'``: the 1-sigma flux error within the source
          segment.  Returned only if `error` is input.

    See Also
    --------
    detect_sources, segment_props
    """

    from scipy import ndimage
    if data.shape != segment_image.shape:
        raise ValueError('data and segment_image must have the same shape')
    if labels is None:
        label_ids = np.unique(segment_image[segment_image > 0])
    else:
        label_ids = np.atleast_1d(labels)

    data_iscopy = False
    if background is not None:
        if np.isscalar(background):
            bkgrd_image = np.zeros_like(data) + background
        else:
            if data.shape != background.shape:
                raise ValueError('If input background is 2D, then it must '
                                 'have the same shape as the input data.')
            bkgrd_image = background
        data = copy.deepcopy(data)
        data_iscopy = True
        data -= bkgrd_image

    if error is not None:
        if data.shape != error.shape:
            raise ValueError('data and error must have the same shape')
        variance = error**2

    if mask is not None:
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape')
        if not data_iscopy:
            data = copy.deepcopy(data)

        mask_idx = mask.nonzero()
        if mask_method == 'exclude':
            # masked pixel will not contribute to sums
            data[mask_idx] = 0.0
            if background is not None:
                bkgrd_image[mask_idx] = 0.0
            if error is not None:
                error[mask_idx] = 0.0
        elif mask_method == 'interpolate':
            for j, i in zip(mask_idx):
                y0, y1 = max(j - 1, 0), min(j + 2, data.shape[0])
                x0, x1 = max(i - 1, 0), min(i + 2, data.shape[1])
                goodpix = ~mask[y0:y1, x0:x1]
                data[j, i] = np.mean(data[y0:y1, x0:x1][goodpix])
                if background is not None:
                    bkgrd_image[j, i] = np.mean(
                        bkgrd_image[y0:y1, x0:x1][goodpix])
                if error is not None:
                    error[j, i] = np.sqrt(np.mean(
                        variance[y0:y1, x0:x1][goodpix]))
        else:
            raise ValueError(
                'mask_method "{0}" is not valid'.format(mask_method))

    segment_sum = ndimage.measurements.sum(data, labels=segment_image,
                                           index=label_ids)
    data = [label_ids, segment_sum]
    names = ('id', 'segment_sum')
    phot_table = Table(data, names=names)

    if error is not None:
        if gain is not None:
            variance += data / gain
        segment_sum_var = ndimage.measurements.sum(variance,
                                                   labels=segment_image,
                                                   index=label_ids)
        segment_sum_err = np.sqrt(segment_sum_var)
        phot_table['segment_sum_err'] = segment_sum_err

    if background is not None:
        background_sum = ndimage.measurements.sum(bkgrd_image,
                                                  labels=segment_image,
                                                  index=label_ids)
        background_mean = ndimage.measurements.mean(bkgrd_image,
                                                  labels=segment_image,
                                                  index=label_ids)
        phot_table['background_sum'] = background_sum
        phot_table['background_mean'] = background_mean

    #bad_labels = (~np.isfinite(phot_table['xcen'])).nonzero()
    #phot_table['flux'][bad_labels] = np.nan
    return phot_table
