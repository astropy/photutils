# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines a class for a rectangular bounding box.
"""

import math

import numpy as np

__all__ = ['BoundingBox']


class BoundingBox:
    """
    A rectangular bounding box in integer (not float) pixel indices.

    Parameters
    ----------
    ixmin, ixmax, iymin, iymax : int
        The bounding box pixel indices.  Note that the upper values
        (``iymax`` and ``ixmax``) are exclusive as for normal slices in
        Python.  The lower values (``ixmin`` and ``iymin``) must not be
        greater than the respective upper values (``ixmax`` and
        ``iymax``).

    Examples
    --------
    When constructing a BoundingBox, it's better to use keyword
    arguments for readability:

    >>> from photutils.aperture import BoundingBox
    >>> bbox = BoundingBox(ixmin=1, ixmax=10, iymin=2, iymax=20)
    >>> bbox
    BoundingBox(ixmin=1, ixmax=10, iymin=2, iymax=20)

    Sometimes it's useful to check if two bounding boxes are the same:

    >>> bbox == BoundingBox(ixmin=1, ixmax=10, iymin=2, iymax=20)
    True
    >>> bbox == BoundingBox(ixmin=7, ixmax=10, iymin=2, iymax=20)
    False

    The "center" and "shape" attributes can be useful when working with
    numpy arrays:

    >>> bbox.center  # numpy order: (y, x)
    (10.5, 5.0)
    >>> bbox.shape  # numpy order: (y, x)
    (18, 9)

    The "extent" is useful when plotting the BoundingBox with
    matplotlib:

    >>> bbox.extent  # matplotlib order: (xmin, xmax, ymin, ymax)
    (0.5, 9.5, 1.5, 19.5)
    """

    def __init__(self, ixmin, ixmax, iymin, iymax):
        for value in (ixmin, ixmax, iymin, iymax):
            if not isinstance(value, (int, np.integer)):
                raise TypeError('ixmin, ixmax, iymin, and iymax must all be '
                                'integers')

        if ixmin > ixmax:
            raise ValueError('ixmin must be <= ixmax')
        if iymin > iymax:
            raise ValueError('iymin must be <= iymax')

        self.ixmin = ixmin
        self.ixmax = ixmax
        self.iymin = iymin
        self.iymax = iymax

    @classmethod
    def from_float(cls, xmin, xmax, ymin, ymax):
        """
        Return the smallest bounding box that fully contains a given
        rectangle defined by float coordinate values.

        Following the pixel index convention, an integer index
        corresponds to the center of a pixel and the pixel edges span
        from (index - 0.5) to (index + 0.5).  For example, the pixel
        edge spans of the following pixels are:

        - pixel 0: from -0.5 to 0.5
        - pixel 1: from 0.5 to 1.5
        - pixel 2: from 1.5 to 2.5

        In addition, because `BoundingBox` upper limits are exclusive
        (by definition), 1 is added to the upper pixel edges.  See
        examples below.

        Parameters
        ----------
        xmin, xmax, ymin, ymax : float
            Float coordinates defining a rectangle.  The lower values
            (``xmin`` and ``ymin``) must not be greater than the
            respective upper values (``xmax`` and ``ymax``).

        Returns
        -------
        bbox : `BoundingBox` object
            The minimal ``BoundingBox`` object fully containing the
            input rectangle coordinates.

        Examples
        --------
        >>> from photutils.aperture import BoundingBox
        >>> BoundingBox.from_float(xmin=1.0, xmax=10.0, ymin=2.0, ymax=20.0)
        BoundingBox(ixmin=1, ixmax=11, iymin=2, iymax=21)

        >>> BoundingBox.from_float(xmin=1.4, xmax=10.4, ymin=1.6, ymax=10.6)
        BoundingBox(ixmin=1, ixmax=11, iymin=2, iymax=12)
        """
        ixmin = math.floor(xmin + 0.5)
        ixmax = math.ceil(xmax + 0.5)
        iymin = math.floor(ymin + 0.5)
        iymax = math.ceil(ymax + 0.5)

        return cls(ixmin, ixmax, iymin, iymax)

    def __eq__(self, other):
        if not isinstance(other, BoundingBox):
            raise TypeError('Can compare BoundingBox only to another '
                            'BoundingBox.')

        return ((self.ixmin == other.ixmin)
                and (self.ixmax == other.ixmax)
                and (self.iymin == other.iymin)
                and (self.iymax == other.iymax))

    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersection(other)

    def __repr__(self):
        return (f'{self.__class__.__name__}(ixmin={self.ixmin}, '
                f'ixmax={self.ixmax}, iymin={self.iymin}, '
                f'iymax={self.iymax})')

    @property
    def center(self):
        """
        The ``(y, x)`` center of the bounding box.
        """
        return (0.5 * (self.iymax - 1 + self.iymin),
                0.5 * (self.ixmax - 1 + self.ixmin))

    @property
    def shape(self):
        """
        The ``(ny, nx)`` shape of the bounding box.
        """
        return self.iymax - self.iymin, self.ixmax - self.ixmin

    def get_overlap_slices(self, shape):
        """
        Get slices for the overlapping part of the bounding box and an
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
            A tuple of slice objects for each axis of an array enclosed
            by the bounding box such that ``small_array[slices_small]``
            extracts the region that is inside the large array. `None`
            is returned if there is no overlap of the bounding box with
            the given image shape.
        """
        if len(shape) != 2:
            raise ValueError('input shape must have 2 elements.')

        xmin = self.ixmin
        xmax = self.ixmax
        ymin = self.iymin
        ymax = self.iymax

        if xmin >= shape[1] or ymin >= shape[0] or xmax <= 0 or ymax <= 0:
            # no overlap of the bounding box with the input shape
            return None, None

        slices_large = (slice(max(ymin, 0), min(ymax, shape[0])),
                        slice(max(xmin, 0), min(xmax, shape[1])))
        slices_small = (slice(max(-ymin, 0),
                              min(ymax - ymin, shape[0] - ymin)),
                        slice(max(-xmin, 0),
                              min(xmax - xmin, shape[1] - xmin)))

        return slices_large, slices_small

    @property
    def extent(self):
        """
        The extent of the mask, defined as the ``(xmin, xmax, ymin,
        ymax)`` bounding box from the bottom-left corner of the
        lower-left pixel to the upper-right corner of the upper-right
        pixel.

        The upper edges here are the actual pixel positions of the
        edges, i.e., they are not "exclusive" indices used for python
        indexing. This is useful for plotting the bounding box using
        Matplotlib.
        """
        return (self.ixmin - 0.5, self.ixmax - 0.5,
                self.iymin - 0.5, self.iymax - 0.5)

    def as_artist(self, **kwargs):
        """
        Return a `matplotlib.patches.Rectangle` that represents the
        bounding box.

        Parameters
        ----------
        **kwargs : dict
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.

        Returns
        -------
        result : `matplotlib.patches.Rectangle`
            A matplotlib rectangular patch.

        Examples
        --------
        .. plot::
            :include-source:

            import matplotlib.pyplot as plt
            import numpy as np
            from photutils.aperture import BoundingBox

            bbox = BoundingBox(2, 7, 3, 8)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            rng = np.random.default_rng(0)
            ax.imshow(rng.random((10, 10)), interpolation='nearest',
                      cmap='viridis')
            ax.add_patch(bbox.as_artist(facecolor='none', edgecolor='white',
                         lw=2.))
        """
        from matplotlib.patches import Rectangle

        return Rectangle(xy=(self.extent[0], self.extent[2]),
                         width=self.shape[1], height=self.shape[0], **kwargs)

    def to_aperture(self):
        """
        Return a `~photutils.aperture.RectangularAperture` that
        represents the bounding box.
        """
        # prevent circular import
        from photutils.aperture.rectangle import RectangularAperture

        xypos = self.center[::-1]  # xy order
        height, width = self.shape
        return RectangularAperture(xypos, w=width, h=height, theta=0.0)

    def plot(self, ax=None, origin=(0, 0), **kwargs):
        """
        Plot the `BoundingBox` on a matplotlib `~matplotlib.axes.Axes`
        instance.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes` or `None`, optional
            The matplotlib axes on which to plot.  If `None`, then the
            current `~matplotlib.axes.Axes` instance is used.

        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        **kwargs : dict
            Any keyword arguments accepted by `matplotlib.patches.Patch`.

        Returns
        -------
        patch : `matplotlib.patches.Patch`
            The matplotlib patch object for the plotted bounding box.
            The patch can be used, for example, when adding a plot
            legend.
        """
        aper = self.to_aperture()
        return aper.plot(ax=ax, origin=origin, **kwargs)[0]

    def union(self, other):
        """
        Return a `BoundingBox` representing the union of this
        `BoundingBox` with another `BoundingBox`.

        Parameters
        ----------
        other : `BoundingBox`
            The `BoundingBox` to join with this one.

        Returns
        -------
        result : `BoundingBox`
            A `BoundingBox` representing the union of the input
            `BoundingBox` with this one.
        """
        if not isinstance(other, BoundingBox):
            raise TypeError('BoundingBox can be joined only with another '
                            'BoundingBox.')

        ixmin = min((self.ixmin, other.ixmin))
        ixmax = max((self.ixmax, other.ixmax))
        iymin = min((self.iymin, other.iymin))
        iymax = max((self.iymax, other.iymax))

        return BoundingBox(ixmin=ixmin, ixmax=ixmax, iymin=iymin, iymax=iymax)

    def intersection(self, other):
        """
        Return a `BoundingBox` representing the intersection of this
        `BoundingBox` with another `BoundingBox`.

        Parameters
        ----------
        other : `BoundingBox`
            The `BoundingBox` to intersect with this one.

        Returns
        -------
        result : `BoundingBox`
            A `BoundingBox` representing the intersection of the input
            `BoundingBox` with this one.
        """
        if not isinstance(other, BoundingBox):
            raise TypeError('BoundingBox can be intersected only with '
                            'another BoundingBox.')

        ixmin = max(self.ixmin, other.ixmin)
        ixmax = min(self.ixmax, other.ixmax)
        iymin = max(self.iymin, other.iymin)
        iymax = min(self.iymax, other.iymax)
        if ixmax < ixmin or iymax < iymin:
            return None

        return BoundingBox(ixmin=ixmin, ixmax=ixmax, iymin=iymin, iymax=iymax)
