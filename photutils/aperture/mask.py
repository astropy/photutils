# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines a class for aperture masks.
"""

import astropy.units as u
import numpy as np

__all__ = ['ApertureMask']


class ApertureMask:
    """
    Class for an aperture mask.

    Parameters
    ----------
    data : array_like
        A 2D array representing the fractional overlap of an aperture
        on the pixel grid. This should be the full-sized (i.e., not
        truncated) array that is the direct output of one of the
        low-level `photutils.geometry` functions.

    bbox : `photutils.aperture.BoundingBox`
        The bounding box object defining the aperture minimal bounding
        box.
    """

    def __init__(self, data, bbox):
        self.data = np.asanyarray(data)
        if self.data.shape != bbox.shape:
            raise ValueError('mask data and bounding box must have the same '
                             'shape')
        self.bbox = bbox
        self._mask = (self.data == 0)

    def __array__(self):
        """
        Array representation of the mask data array (e.g., for matplotlib).
        """

        return self.data

    @property
    def shape(self):
        """
        The shape of the mask data array.
        """

        return self.data.shape

    def _overlap_slices(self, shape):
        """
        Calculate the slices for the overlapping part of the bounding
        box and an array of the given shape.

        Parameters
        ----------
        shape : tuple of int
            The ``(ny, nx)`` shape of array where the slices are to be
            applied.

        Returns
        -------
        slices_large : tuple of slices
            A tuple of slice objects for each axis of the large array,
            such that ``large_array[slices_large]`` extracts the region
            of the large array that overlaps with the small array.

        slices_small : slice
            A tuple of slice objects for each axis of the small array,
            such that ``small_array[slices_small]`` extracts the region
            of the small array that is inside the large array.
        """

        if len(shape) != 2:
            raise ValueError('input shape must have 2 elements.')

        xmin = self.bbox.ixmin
        xmax = self.bbox.ixmax
        ymin = self.bbox.iymin
        ymax = self.bbox.iymax

        if xmin >= shape[1] or ymin >= shape[0] or xmax <= 0 or ymax <= 0:
            # no overlap of the aperture with the data
            return None, None

        slices_large = (slice(max(ymin, 0), min(ymax, shape[0])),
                        slice(max(xmin, 0), min(xmax, shape[1])))

        slices_small = (slice(max(-ymin, 0),
                              min(ymax - ymin, shape[0] - ymin)),
                        slice(max(-xmin, 0),
                              min(xmax - xmin, shape[1] - xmin)))

        return slices_large, slices_small

    def _to_image_partial_overlap(self, image):
        """
        Return an image of the mask in a 2D array, where the mask
        is not fully within the image (i.e., partial or no overlap).
        """

        # find the overlap of the mask on the output image shape
        slices_large, slices_small = self._overlap_slices(image.shape)

        if slices_small is None:
            return None  # no overlap

        # insert the mask into the output image
        image[slices_large] = self.data[slices_small]

        return image

    def to_image(self, shape):
        """
        Return an image of the mask in a 2D array of the given shape,
        taking any edge effects into account.

        Parameters
        ----------
        shape : tuple of int
            The ``(ny, nx)`` shape of the output array.

        Returns
        -------
        result : `~numpy.ndarray`
            A 2D array of the mask.
        """

        if len(shape) != 2:
            raise ValueError('input shape must have 2 elements.')

        image = np.zeros(shape)

        if self.bbox.ixmin < 0 or self.bbox.iymin < 0:
            return self._to_image_partial_overlap(image)

        try:
            image[self.bbox.slices] = self.data
        except ValueError:  # partial or no overlap
            image = self._to_image_partial_overlap(image)

        return image

    def cutout(self, data, fill_value=0., copy=False):
        """
        Create a cutout from the input data over the mask bounding box,
        taking any edge effects into account.

        Parameters
        ----------
        data : array_like
            A 2D array on which to apply the aperture mask.

        fill_value : float, optional
            The value used to fill pixels where the aperture mask does
            not overlap with the input ``data``.  The default is 0.

        copy : bool, optional
            If `True` then the returned cutout array will always be hold
            a copy of the input ``data``.  If `False` and the mask is
            fully within the input ``data``, then the returned cutout
            array will be a view into the input ``data``.  In cases
            where the mask partially overlaps or has no overlap with the
            input ``data``, the returned cutout array will always hold a
            copy of the input ``data`` (i.e., this keyword has no
            effect).

        Returns
        -------
        result : `~numpy.ndarray`
            A 2D array cut out from the input ``data`` representing the
            same cutout region as the aperture mask.  If there is a
            partial overlap of the aperture mask with the input data,
            pixels outside of the data will be assigned to
            ``fill_value``.  `None` is returned if there is no overlap
            of the aperture with the input ``data``.
        """

        data = np.asanyarray(data)
        if data.ndim != 2:
            raise ValueError('data must be a 2D array.')

        partial_overlap = False
        if self.bbox.ixmin < 0 or self.bbox.iymin < 0:
            partial_overlap = True

        if not partial_overlap:
            # try this for speed -- the result may still be a partial
            # overlap, in which case the next block will be triggered
            if copy:
                cutout = data[self.bbox.slices].copy()  # preserves Quantity
            else:
                cutout = data[self.bbox.slices]

        if partial_overlap or (cutout.shape != self.shape):
            slices_large, slices_small = self._overlap_slices(data.shape)

            if slices_small is None:
                return None  # no overlap

            # cutout is a copy
            cutout = np.zeros(self.shape, dtype=data.dtype)
            cutout[:] = fill_value
            cutout[slices_small] = data[slices_large]

            if isinstance(data, u.Quantity):
                cutout = u.Quantity(cutout, unit=data.unit)

        return cutout

    def multiply(self, data, fill_value=0.):
        """
        Multiply the aperture mask with the input data, taking any edge
        effects into account.

        The result is a mask-weighted cutout from the data.

        Parameters
        ----------
        data : array_like or `~astropy.units.Quantity`
            The 2D array to multiply with the aperture mask.

        fill_value : float, optional
            The value is used to fill pixels where the aperture mask
            does not overlap with the input ``data``.  The default is 0.

        Returns
        -------
        result : `~numpy.ndarray` or `None`
            A 2D mask-weighted cutout from the input ``data``.  If there
            is a partial overlap of the aperture mask with the input
            data, pixels outside of the data will be assigned to
            ``fill_value`` before being multipled with the mask.  `None`
            is returned if there is no overlap of the aperture with the
            input ``data``.
        """

        cutout = self.cutout(data, fill_value=fill_value)
        if cutout is None:
            return None
        else:
            weighted_cutout = cutout * self.data

            # needed to zero out non-finite data values outside of the
            # mask but within the bounding box
            weighted_cutout[self._mask] = 0.

            return weighted_cutout
