# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define classes to estimate local background using a circular annulus
aperture.
"""

import numpy as np

from photutils.aperture import CircularAnnulus
from photutils.background import MedianBackground
from photutils.utils._repr import make_repr

__all__ = ['LocalBackground']


class LocalBackground:
    """
    Class to compute a local background using a circular annulus
    aperture.

    Parameters
    ----------
    inner_radius : float
        The inner radius of the circular annulus in pixels.

    outer_radius : float
        The outer radius of the circular annulus in pixels.

    bkg_estimator : callable, optional
        A callable object (a function or e.g., an instance of any
        `~photutils.background.BackgroundBase` subclass) used to
        estimate the background in each aperture. The callable object
        must take in a 2D `~numpy.ndarray` or `~numpy.ma.MaskedArray`
        and have an ``axis`` keyword. The default is an instance of
        `~photutils.background.MedianBackground` with sigma clipping
        (i.e., sigma-clipped median).

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.background import LocalBackground
    >>> data = np.ones((101, 101))
    >>> local_bkg = LocalBackground(5, 10)
    >>> bkg = local_bkg(data, 50, 50)
    >>> print(bkg)  # doctest: +FLOAT_CMP
    1.0

    >>> # Multiple positions
    >>> x = [30, 50, 70]
    >>> y = [30, 50, 70]
    >>> bkg = local_bkg(data, x, y)
    >>> print(bkg)  # doctest: +FLOAT_CMP
    [1. 1. 1.]
    """

    def __init__(self, inner_radius, outer_radius, bkg_estimator=None):
        if inner_radius <= 0:
            msg = 'inner_radius must be positive.'
            raise ValueError(msg)
        if outer_radius <= 0:
            msg = 'outer_radius must be positive.'
            raise ValueError(msg)
        if outer_radius <= inner_radius:
            msg = 'outer_radius must be greater than inner_radius.'
            raise ValueError(msg)

        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        if bkg_estimator is None:
            bkg_estimator = MedianBackground()
        self.bkg_estimator = bkg_estimator

    def __repr__(self):
        params = ('inner_radius', 'outer_radius', 'bkg_estimator')
        return make_repr(self, params)

    def to_aperture(self, x, y):
        """
        Return a `~photutils.aperture.CircularAnnulus` instance
        representing the local background annulus at the given
        positions.

        Parameters
        ----------
        x, y : float or 1D float `~numpy.ndarray`
            The aperture center (x, y) position(s) at which to create
            the annulus aperture.

        Returns
        -------
        apertures : `~photutils.aperture.CircularAnnulus` instance
            The circular annulus aperture(s) at the given position(s).

        Examples
        --------
        >>> from photutils.background import LocalBackground
        >>> local_bkg = LocalBackground(5, 10)
        >>> aperture = local_bkg.to_aperture(50, 50)
        >>> aperture  # doctest: +FLOAT_CMP
        <CircularAnnulus([[50., 50.]], r_in=5.0, r_out=10.0)>

        >>> # Multiple positions
        >>> aperture = local_bkg.to_aperture([30, 70], [40, 80])
        >>> aperture  # doctest: +FLOAT_CMP
        <CircularAnnulus([[30., 40.],
                          [70., 80.]], r_in=5.0, r_out=10.0)>
        >>> print(len(aperture.positions))
        2
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        positions = np.array(list(zip(x, y, strict=True)))
        return CircularAnnulus(positions, self.inner_radius,
                               self.outer_radius)

    def __call__(self, data, x, y, mask=None):
        """
        Measure the local background in a circular annulus.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The 2D array on which to measure the local background.

        x, y : float or 1D float `~numpy.ndarray`
            The aperture center (x, y) position(s) at which to measure
            the local background.

        mask : 2D bool `~numpy.ndarray`, optional
            A boolean mask with the same shape as ``data`` where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked data are excluded from all calculations.

        Returns
        -------
        value : float or 1D float `~numpy.ndarray`
            The local background values. If all pixels in an annulus
            are masked or outside the data bounds, the corresponding
            value will be NaN.

        Examples
        --------
        >>> import numpy as np
        >>> from photutils.background import LocalBackground
        >>> data = np.ones((101, 101))
        >>> local_bkg = LocalBackground(5, 10)
        >>> bkg = local_bkg(data, 50, 50)
        >>> print(bkg)  # doctest: +FLOAT_CMP
        1.0

        >>> # Multiple positions
        >>> bkg = local_bkg(data, [30, 50], [40, 60])
        >>> print(bkg)  # doctest: +FLOAT_CMP
        [1. 1.]

        >>> # Position outside data returns NaN
        >>> bkg = local_bkg(data, -50, -50)
        >>> print(np.isnan(bkg))
        True
        """
        apertures = self.to_aperture(x, y)
        apermasks = apertures.to_mask(method='center')

        n_apertures = len(apermasks)
        bkg = np.empty(n_apertures)
        for i, apermask in enumerate(apermasks):
            values = apermask.get_values(data, mask=mask)
            bkg[i] = self.bkg_estimator(values)

        if bkg.size == 1:
            bkg = bkg[0]

        return bkg
