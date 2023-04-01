# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for generating 2D image cutouts.
"""

import numpy as np
from astropy.nddata import extract_array, overlap_slices
from astropy.utils import lazyproperty

from photutils.aperture import BoundingBox

__all__ = ['CutoutImage']


class CutoutImage:
    """
    Create a cutout object from a 2D array.

    The returned object will contain a 2D cutout array.  If
    ``copy=False`` (default), the cutout array is a view into the
    original ``data`` array, otherwise the cutout array will contain a
    copy of the original data.

    Parameters
    ----------
    data : `~numpy.ndarray`
        The 2D data array from which to extract the cutout array.

    position : 2 tuple
        The ``(y, x)`` position of the center of the cutout array with
        respect to the ``data`` array.

    shape : 2 tuple of int
        The shape of the cutout array along each axis in ``(ny, nx)``
        order.

    mode : {'trim', 'partial', 'strict'}, optional
        The mode used for creating the cutout data array. For the
        ``'partial'`` and ``'trim'`` modes, a partial overlap
        of the cutout array and the input ``data`` array is
        sufficient. For the ``'strict'`` mode, the cutout array
        has to be fully contained within the ``data`` array,
        otherwise an `~astropy.nddata.utils.PartialOverlapError`
        is raised. In all modes, non-overlapping arrays will raise
        a `~astropy.nddata.utils.NoOverlapError`. In ``'partial'``
        mode, positions in the cutout array that do not overlap with
        the ``data`` array will be filled with ``fill_value``. In
        ``'trim'`` mode only the overlapping elements are returned, thus
        the resulting cutout array may be smaller than the requested
        ``shape``.

    fill_value : float or int, optional
        If ``mode='partial'``, the value to fill pixels in the
        cutout array that do not overlap with the input ``data``.
        ``fill_value`` must have the same ``dtype`` as the input
        ``data`` array.

    copy : bool, optional
        If `False` (default), then the cutout data will be a view into
        the original ``data`` array. If `True`, then the cutout data
        will hold a copy of the original ``data`` array.

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.utils import CutoutImage
    >>> data = np.arange(20.0).reshape(5, 4)
    >>> cutout = CutoutImage(data, (2, 2), (3, 3))
    >>> print(cutout.data)  # doctest: +FLOAT_CMP
    [[ 5.  6.  7.]
     [ 9. 10. 11.]
     [13. 14. 15.]]

    >>> cutout2 = CutoutImage(data, (0, 0), (3, 3), mode='partial')
    >>> print(cutout2.data)  # doctest: +FLOAT_CMP
    [[nan nan nan]
     [nan  0.  1.]
     [nan  4.  5.]]
    """

    def __init__(self, data, position, shape, mode='trim', fill_value=np.nan,
                 copy=False):
        self.position = position
        self.input_shape = shape
        self.mode = mode
        self.fill_value = fill_value
        self.copy = copy

        data = np.asanyarray(data)
        self._overlap_slices = overlap_slices(data.shape, shape, position,
                                              mode=mode)
        self.data = self._make_cutout(data)
        self.shape = self.data.shape

    def _make_cutout(self, data):
        cutout_data = extract_array(data, self.input_shape, self.position,
                                    mode=self.mode, fill_value=self.fill_value,
                                    return_position=False)
        if self.copy:
            cutout_data = np.copy(cutout_data)
        return cutout_data

    def __array__(self, dtype=None):
        """
        Array representation of the cutout data array (e.g., for
        matplotlib).
        """
        return np.asarray(self.data, dtype=dtype)

    def __str__(self):
        cls_name = f'<{self.__class__.__module__}.{self.__class__.__name__}>'
        props = f'Shape: {self.data.shape}'
        return f'{cls_name}\n' + props

    def __repr__(self):
        return self.__str__()

    @lazyproperty
    def slices_original(self):
        """
        A tuple of slice objects in axis order for the minimal bounding
        box of the cutout with respect to the original array.

        For ``mode='partial'``, the slices are for the valid
        (non-filled) cutout values.
        """
        return self._overlap_slices[0]

    @lazyproperty
    def slices_cutout(self):
        """
        A tuple of slice objects in axis order for the minimal bounding
        box of the cutout with respect to the cutout array.

        For ``mode='partial'``, the slices are for the valid
        (non-filled) cutout values.
        """
        return self._overlap_slices[1]

    def _calc_bbox(self, slices):
        """
        Calculate the `~photutils.aperture.BoundingBox` of the
        rectangular bounding box from the input slices.
        """
        return BoundingBox(ixmin=slices[1].start, ixmax=slices[1].stop,
                           iymin=slices[0].start, iymax=slices[0].stop)

    @lazyproperty
    def bbox_original(self):
        """
        The `~photutils.aperture.BoundingBox` of the minimal rectangular
        region of the cutout array with respect to the original array.

        For ``mode='partial'``, the bounding box indices are for the
        valid (non-filled) cutout values.
        """
        return self._calc_bbox(self.slices_original)

    @lazyproperty
    def bbox_cutout(self):
        """
        The `~photutils.aperture.BoundingBox` of the minimal rectangular
        region of the cutout array with respect to the cutout array.

        For ``mode='partial'``, the bounding box indices are for the
        valid (non-filled) cutout values.
        """
        return self._calc_bbox(self.slices_cutout)

    def _calc_xyorigin(self, slices):
        """
        Calculate the (x, y) origin, taking into account partial
        overlaps.
        """
        xorigin, yorigin = (slices[1].start, slices[0].start)

        if self.mode == 'partial':
            yorigin -= self.slices_cutout[0].start
            xorigin -= self.slices_cutout[1].start

        return np.array((xorigin, yorigin))

    @lazyproperty
    def xyorigin(self):
        """
        A `~numpy.ndarray` containing the ``(x, y)`` integer index of
        the origin pixel of the cutout with respect to the original
        array.

        The origin index will be negative for cutouts with partial
        overlaps.
        """
        return self._calc_xyorigin(self.slices_original)
