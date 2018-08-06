# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy.io.fits.util import _is_int


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
    >>> from photutils import BoundingBox

    >>> # constructing a BoundingBox like this is cryptic:
    >>> bbox = BoundingBox(1, 10, 2, 20)

    >>> # it's better to use keyword arguments for readability:
    >>> bbox = BoundingBox(ixmin=1, ixmax=10, iymin=2, iymax=20)
    >>> bbox  # nice repr, useful for interactive work
    BoundingBox(ixmin=1, ixmax=10, iymin=2, iymax=20)

    >>> # sometimes it's useful to check if two bounding boxes are the same
    >>> bbox == BoundingBox(ixmin=1, ixmax=10, iymin=2, iymax=20)
    True
    >>> bbox == BoundingBox(ixmin=7, ixmax=10, iymin=2, iymax=20)
    False

    >>> # "shape" and "slices" can be useful when working with numpy arrays
    >>> bbox.shape  # numpy order: (y, x)
    (18, 9)
    >>> bbox.slices  # numpy order: (y, x)
    (slice(2, 20, None), slice(1, 10, None))

    >>> # "extent" is useful when plotting the BoundingBox with matplotlib
    >>> bbox.extent  # matplotlib order: (x, y)
    (0.5, 9.5, 1.5, 19.5)
    """

    def __init__(self, ixmin, ixmax, iymin, iymax):
        if not _is_int(ixmin):
            raise TypeError('ixmin must be an integer')
        if not _is_int(ixmax):
            raise TypeError('ixmax must be an integer')
        if not _is_int(iymin):
            raise TypeError('iymin must be an integer')
        if not _is_int(iymax):
            raise TypeError('iymax must be an integer')

        if ixmin > ixmax:
            raise ValueError('ixmin must be <= ixmax')
        if iymin > iymax:
            raise ValueError('iymin must be <= iymax')

        self.ixmin = ixmin
        self.ixmax = ixmax
        self.iymin = iymin
        self.iymax = iymax

    @classmethod
    def _from_float(cls, xmin, xmax, ymin, ymax):
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
        >>> from photutils import BoundingBox
        >>> BoundingBox._from_float(xmin=1.0, xmax=10.0, ymin=2.0, ymax=20.0)
        BoundingBox(ixmin=1, ixmax=11, iymin=2, iymax=21)

        >>> BoundingBox._from_float(xmin=1.4, xmax=10.4, ymin=1.6, ymax=10.6)
        BoundingBox(ixmin=1, ixmax=11, iymin=2, iymax=12)
        """

        ixmin = int(np.floor(xmin + 0.5))
        ixmax = int(np.ceil(xmax + 0.5))
        iymin = int(np.floor(ymin + 0.5))
        iymax = int(np.ceil(ymax + 0.5))

        return cls(ixmin, ixmax, iymin, iymax)

    def __eq__(self, other):
        if not isinstance(other, BoundingBox):
            raise TypeError('Can compare BoundingBox only to another '
                            'BoundingBox.')

        return (
            (self.ixmin == other.ixmin) and
            (self.ixmax == other.ixmax) and
            (self.iymin == other.iymin) and
            (self.iymax == other.iymax)
        )

    def __repr__(self):
        data = self.__dict__
        data['name'] = self.__class__.__name__
        fmt = ('{name}(ixmin={ixmin}, ixmax={ixmax}, iymin={iymin}, '
               'iymax={iymax})')
        return fmt.format(**data)

    @property
    def shape(self):
        """
        The ``(ny, nx)`` shape of the bounding box.
        """

        return self.iymax - self.iymin, self.ixmax - self.ixmin

    @property
    def slices(self):
        """
        The bounding box as a tuple of `slice` objects.

        The slice tuple is in numpy axis order (i.e. ``(y, x)``) and
        therefore can be used to slice numpy arrays.
        """

        return (slice(self.iymin, self.iymax), slice(self.ixmin, self.ixmax))

    @property
    def extent(self):
        """
        The extent of the mask, defined as the ``(xmin, xmax, ymin,
        ymax)`` bounding box from the bottom-left corner of the
        lower-left pixel to the upper-right corner of the upper-right
        pixel.

        The upper edges here are the actual pixel positions of the
        edges, i.e. they are not "exclusive" indices used for python
        indexing.  This is useful for plotting the bounding box using
        Matplotlib.
        """

        return (
            self.ixmin - 0.5,
            self.ixmax - 0.5,
            self.iymin - 0.5,
            self.iymax - 0.5,
        )

    def as_patch(self, **kwargs):
        """
        Return a `matplotlib.patches.Rectangle` that represents the
        bounding box.

        Parameters
        ----------
        kwargs
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
            from photutils import BoundingBox
            bbox = BoundingBox(2, 7, 3, 8)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            np.random.seed(12345)
            ax.imshow(np.random.random((10, 10)), interpolation='nearest',
                      cmap='viridis')
            ax.add_patch(bbox.as_patch(facecolor='none', edgecolor='white',
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

        from .rectangle import RectangularAperture

        xpos = (self.extent[1] + self.extent[0]) / 2.
        ypos = (self.extent[3] + self.extent[2]) / 2.
        xypos = (xpos, ypos)
        h, w = self.shape

        return RectangularAperture(xypos, w=w, h=h, theta=0.)

    def plot(self, origin=(0, 0), ax=None, fill=False, **kwargs):
        """
        Plot the `BoundingBox` on a matplotlib `~matplotlib.axes.Axes`
        instance.

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        ax : `matplotlib.axes.Axes` instance, optional
            If `None`, then the current `~matplotlib.axes.Axes` instance
            is used.

        fill : bool, optional
            Set whether to fill the aperture patch.  The default is
            `False`.

        kwargs
            Any keyword arguments accepted by `matplotlib.patches.Patch`.
        """

        aper = self.to_aperture()
        aper.plot(origin=origin, ax=ax, fill=fill, **kwargs)
