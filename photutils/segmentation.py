# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import copy
import numpy as np
from astropy.table import Table, Column
from skimage.measure._regionprops import _cached_property, _RegionProperties


__all__ = ['SegmentProperties', 'segment_props', 'segment_photometry']


class SegmentProperties(object):
    def __init__(self, image, segment_image, label, slice):
        self.label = label
        self._slice = slice
        self._label_image = segment_image
        self._intensity_image = image
        self._cache_active = True

    image = _RegionProperties.image
    intensity_image = _RegionProperties.intensity_image
    _image_double = _RegionProperties._image_double
    local_centroid = _RegionProperties.local_centroid
    moments = _RegionProperties.moments
    moments_central = _RegionProperties.moments_central
    inertia_tensor = _RegionProperties.inertia_tensor
    inertia_tensor_eigvals = _RegionProperties.inertia_tensor_eigvals

    centroid = _RegionProperties.centroid
    min_value = _RegionProperties.min_intensity
    max_value = _RegionProperties.max_intensity
    area = _RegionProperties.area
    equivalent_diameter= _RegionProperties.equivalent_diameter
    perimeter = _RegionProperties.perimeter
    major_axis_length = _RegionProperties.major_axis_length
    minor_axis_length = _RegionProperties.minor_axis_length
    eccentricity = _RegionProperties.eccentricity
    orientation = _RegionProperties.orientation

    bbox = _RegionProperties.bbox
    coords = _RegionProperties.coords

    def __getitem__(self, key):
        return getattr(self, key, None)

    @_cached_property
    def equivalent_radius(self):
        return 0.5 * self.equivalent_diameter

    @_cached_property
    def region(self):
        return self.intensity_image[self.image]

    @_cached_property
    def min_position(self):
        return np.argwhere(self.region == self.min_value)

    @_cached_property
    def max_position(self):
        return np.argwhere(self.region == self.max_value)

    # TODO:  allow alternate centroid methods via input centroid_func:
    # npix = ndimage.labeled_comprehension(image, segment_image, label_ids,
    #                                      centroid_func, np.float32, np.nan)
    #centroids = ndimage.center_of_mass(image, segment_image, objids)
    #ycen, xcen = np.transpose(centroids)
    #npix = ndimage.labeled_comprehension(image, segment_image, objids, len,
    #                                     np.float32, np.nan)
    #radii = np.sqrt(npix / np.pi)


def segment_props(image, segment_image, mask=None, mask_method='exclude',
                  background=None, labels=None, output_table=False):
    """


    Parameters
    ----------
    data : array_like
        The 2D array on which to perform photometry.

    segment_image : array_like
        A 2D segmentation image where sources are marked by different
        positive integer values.  A value of zero is reserved for the
        background.

    mask : array_like, bool, optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``image`` is
        masked.  Use the ``mask_method`` keyword to select the method
        used to treat masked pixels.

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

    output_table : bool, optional
        If `True` then return an astropy `astropy.table.Table`,
        otherwise return a list of `SegmentProperties`.

    Returns
    -------
    output : `astropy.table.Table` or list of `SegmentProperties`.

        * If ``output_table = True``: `astropy.table.Table`
              A table of the photometry of the segmented sources
              containing the columns listed below.

        * If ``output_table = False``: list
              A list of `SegmentProperties`, one for each source
              segment.

    Notes
    -----
    The following properties can be accessed either as columns in an
    `astropy.table.Table` or as attributes or keys of
    `SegmentProperties`:

    **id** : int
        The source identification number corresponding to the object
        label in the ``segment_image``.

    **xcen**, **ycen** : float
        The ``x`` and ``y`` coordinates of the centroid within the
        source segment.

    **max_value**, **min_value** : float
        The minimum and maximum pixel values within the source segment.

    **area** : float
        The number pixels in the source segment.

    **equivalent_radius** : float
        The radius of a circle with the same ``area`` as the source
        segment.

    **perimeter** : float
        The perimeter of source segment, approximated using a line
        through the centers of the border pixels using a 4-connectivity.

    **major_axis_length** : float
        The length of the major axis of the ellipse that has the same
        normalized second central moments as the region.

    **minor_axis_length** : float
        The length of the minor axis of the ellipse that has the same
        normalized second central moments as the region.

    **eccentricity** : float
        The cccentricity of the ellipse that has the same second-moments
        as the region. The eccentricity is the ratio of the distance
        between its minor and major axis length. The value is between 0
        and 1.

    **orientation** : float
        The ngle between the X-axis and the major axis of the ellipse
        that has the same second-moments as the region. Ranging from
        -pi/2 to pi/2 in counter-clockwise direction.

    The following properties can be accessed only as columns in an
    `astropy.table.Table`:

    **xmin**, **xmax**, **ymin**, **ymax** : float
        The bounding box of the source segment.

    The following properties can be accessed only as attributes or keys
    of `SegmentProperties`:

    **min_position**, **max_position** : 2-tuple
        The image coordinates ``(y, x)`` of the minimum and maximum
        pixel values.

    **bbox** : 4-tuple
        The bounding box ``(min_row, min_col, max_row, max_col)`` of the
        source segment.

    **coords** : (N, 2) `numpy.ndarray`
        The coordinate list ``(row, col)`` of the source segment.

    **moments** :
        2nd order central moments

    **cutout_image** : `numpy.ndarray`
        A 2D cutout image based on the bounding box (``bbox``) of the
        source segment.

    **masked_cutout_image** : `numpy.ndarray`
        A 2D cutout image based on the bounding box (``bbox``) of the
        source segment, but including *only* the segmented pixels of the
        object.
    """

    from scipy import ndimage
    objslices = ndimage.find_objects(segment_image)
    objpropslist = []
    for i, objslice in enumerate(objslices):
        if objslice is None:
            continue
        label = i + 1
        objprops = SegmentProperties(image, segment_image, label, objslice)
        objpropslist.append(objprops)

    if not output_table:
        return objpropslist
    else:
        props_table = Table()
        ids = [getattr(obj, 'label') for obj in objpropslist]
        props_table['id'] = Column(ids)
        centroid = [getattr(objprops, 'centroid') for objprops in
                    objpropslist]
        xcen, ycen = np.transpose(centroid)
        props_table['xcen'] = Column(xcen)
        props_table['ycen'] = Column(ycen)
        bbox = [getattr(objprops, 'bbox') for objprops in objpropslist]
        xmin, ymin, xmax, ymax = np.transpose(bbox)
        props_table['xmin'] = Column(xmin)
        props_table['xmax'] = Column(xmax)
        props_table['ymin'] = Column(ymin)
        props_table['ymax'] = Column(ymax)

        props = ['min_value', 'max_value']
        for prop in props:
            data = [getattr(objprops, prop) for objprops in objpropslist]
            props_table[prop] = Column(data)

        #minpos = [getattr(objprops, 'min_position') for objprops in
        #          objpropslist]
        #xmin_position, ymin_position = np.transpose(minpos)
        #props_table['xmin_position'] = Column(xmin_position)
        #props_table['ymin_position'] = Column(ymin_position)

        #maxpos = [getattr(objprops, 'max_position') for objprops in
        #          objpropslist]
        #xmax_position, ymax_position = np.transpose(maxpos)
        #props_table['xmax_position'] = Column(xmax_position)
        #props_table['ymax_position'] = Column(ymax_position)

        props = ['area', 'equivalent_radius', 'perimeter',
                 'major_axis_length', 'minor_axis_length', 'eccentricity',
                 'orientation']
        for prop in props:
            data = [getattr(objprops, prop) for objprops in objpropslist]
            props_table[prop] = Column(data)

        return props_table


def segment_photometry(data, segment_image, error=None, gain=None,
                       mask=None, mask_method='exclude', background=None,
                       labels=None):
    """
    Perform photometry of sources whose extents are defined by a labeled
    segmentation image.

    When the segmentation image is defined using a thresholded flux
    level (e.g., see `detect_sources`), this is equivalent to performing
    isophotal photometry in `SExtractor`_.

    .. _SExtractor : http://www.astromatic.net/software/sextractor

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
        ``data`` used to calculate the Poisson error of the sources.  If
        ``gain`` is input, then ``error`` should include all sources of
        "background" error but *exclude* the Poission error of the
        sources.  If ``gain`` is `None`, then the ``error`` is assumed
        to include *all* sources of error, including the Poission error
        of the sources.  For example, if your input ``data`` is in units
        of ADU, then ``gain`` should represent electrons/ADU.  If your
        input ``data`` is in units of electrons/s then ``gain`` should
        be the exposure time.

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

        * ``'id'``: The source identification number corresponding to
          the object label in the ``segment_image``.
        * ``'segment_sum'``: The sum of image values within the source
          segment.
        * ``'segment_sum_err'``: The corresponding uncertainty in
          ``'segment_sum'`` values.  Returned only if ``error`` is
          input.
        * ``'background_sum'``: The sum of background values within the
          source segment.  Returned only if ``background`` is input.
        * ``'background_mean'``: The mean of background values within
          the source segment.  Returned only if ``background`` is input.

    See Also
    --------
    detect_sources, segment_props
    """

    from scipy import ndimage
    if segment_image.shape != data.shape:
        raise ValueError('segment_image and data must have the same shape')
    if labels is None:
        label_ids = np.unique(segment_image[segment_image > 0])
    else:
        label_ids = np.atleast_1d(labels)

    data_iscopy = False
    if background is not None:
        if np.isscalar(background):
            bkgrd_image = np.zeros_like(data) + background
        else:
            if background.shape != data.shape:
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
            # masked pixels will not contribute to sums
            data[mask_idx] = 0.0
            if background is not None:
                bkgrd_image[mask_idx] = 0.0
            if error is not None:
                variance[mask_idx] = 0.0
        elif mask_method == 'interpolate':
            for j, i in zip(*mask_idx):
                y0, y1 = max(j - 1, 0), min(j + 2, data.shape[0])
                x0, x1 = max(i - 1, 0), min(i + 2, data.shape[1])
                goodpix = ~mask[y0:y1, x0:x1]
                data[j, i] = np.mean(data[y0:y1, x0:x1][goodpix])
                if background is not None:
                    bkgrd_image[j, i] = np.mean(
                        bkgrd_image[y0:y1, x0:x1][goodpix])
                if error is not None:
                    variance[j, i] = np.sqrt(np.mean(
                        variance[y0:y1, x0:x1][goodpix]))
        else:
            raise ValueError(
                'mask_method "{0}" is not valid'.format(mask_method))

    segment_sum = ndimage.measurements.sum(data, labels=segment_image,
                                           index=label_ids)
    columns = [label_ids, segment_sum]
    names = ('id', 'segment_sum')
    phot_table = Table(columns, names=names)

    if error is not None:
        if gain is not None:
            if np.isscalar(gain):
                gain = np.broadcast_arrays(gain, data)[0]
            gain = np.asarray(gain)
            if gain.shape != data.shape:
                raise ValueError('If input gain is 2D, then it must have '
                                 'the same shape as the input data.')
            if np.any(gain <= 0):
                raise ValueError('gain must be positive everywhere')
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
    return phot_table
