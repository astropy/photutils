# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines a class for aperture masks.
"""
import warnings

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
        Array representation of the mask data array (e.g., for
        matplotlib).
        """
        return self.data

    @property
    def shape(self):
        """
        The shape of the mask data array.
        """
        return self.data.shape

    def get_overlap_slices(self, shape):
        """
        Get slices for the overlapping part of the aperture mask and a
        2D array.

        Parameters
        ----------
        shape : 2-tuple of int
            The shape of the 2D array.

        Returns
        -------
        slices_large : tuple of slices or `None`
            A tuple of slice objects for each axis of the large array,
            such that ``large_array[slices_large]`` extracts the region
            of the large array that overlaps with the small array.
            `None` is returned if there is no overlap of the bounding
            box with the given image shape.

        slices_small : tuple of slices or `None`
            A tuple of slice objects for each axis of the aperture mask
            array such that ``small_array[slices_small]`` extracts the
            region that is inside the large array. `None` is returned if
            there is no overlap of the bounding box with the given image
            shape.
        """
        return self.bbox.get_overlap_slices(shape)

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

        # find the overlap of the mask on the output image shape
        slices_large, slices_small = self.get_overlap_slices(shape)

        if slices_small is None:
            return None  # no overlap

        # insert the mask into the output image
        image = np.zeros(shape)
        image[slices_large] = self.data[slices_small]
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
        result : `~numpy.ndarray` or `None`
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

        # find the overlap of the mask on the output image shape
        slices_large, slices_small = self.get_overlap_slices(data.shape)

        if slices_small is None:
            return None  # no overlap

        cutout_shape = (slices_small[0].stop - slices_small[0].start,
                        slices_small[1].stop - slices_small[1].start)

        if cutout_shape == self.shape:
            cutout = data[slices_large]
            if copy:
                cutout = np.copy(cutout)
            return cutout

        # cutout is always a copy for partial overlap
        if ~np.isfinite(fill_value):
            dtype = float
        else:
            dtype = data.dtype
        cutout = np.zeros(self.shape, dtype=dtype)
        cutout[:] = fill_value
        cutout[slices_small] = data[slices_large]

        if isinstance(data, u.Quantity):
            cutout <<= data.unit

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
            A 2D mask-weighted cutout from the input ``data``. If
            there is a partial overlap of the aperture mask with the
            input data, pixels outside of the data will be assigned to
            ``fill_value`` before being multiplied with the mask. `None`
            is returned if there is no overlap of the aperture with the
            input ``data``.
        """
        cutout = self.cutout(data, fill_value=fill_value)
        if cutout is None:
            return None
        else:
            # ignore multiplication with non-finite data values
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                weighted_cutout = cutout * self.data

            # fill values outside of the mask but within the bounding box
            weighted_cutout[self._mask] = fill_value

            return weighted_cutout

    def get_values(self, data, mask=None):
        """
        Get the mask-weighted pixel values from the data as a 1D array.

        If the ``ApertureMask`` was created with ``method='center'``,
        (where the mask weights are only 1 or 0), then the returned
        values will simply be pixel values extracted from the data.

        Parameters
        ----------
        data : array_like or `~astropy.units.Quantity`
            The 2D array from which to get mask-weighted values.

        mask : array_like (bool), optional
            A boolean mask with the same shape as ``data`` where a
            `True` value indicates the corresponding element of ``data``
            is not returned in the result.

        Returns
        -------
        result : `~numpy.ndarray`
            A 1D array of mask-weighted pixel values from the input
            ``data``. If there is no overlap of the aperture with the
            input ``data``, the result will be an empty array with shape
            (0,).
        """
        slc_large, slc_small = self.get_overlap_slices(data.shape)
        if slc_large is None:
            return np.array([])
        cutout = data[slc_large]
        apermask = self.data[slc_small]
        pixel_mask = (apermask > 0)  # good pixels

        if mask is not None:
            if mask.shape != data.shape:
                raise ValueError('mask and data must have the same shape')
            pixel_mask &= ~mask[slc_large]

        # ignore multiplication with non-finite data values
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return (cutout * apermask)[pixel_mask]
