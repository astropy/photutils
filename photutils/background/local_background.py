# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines classes to estimate local background using a
circular annulus aperture.
"""

import numpy as np

from photutils.aperture import CircularAnnulus
from photutils.background import MedianBackground

__all__ = ['LocalBackground']


class LocalBackground:
    def __init__(self, inner_radius, outer_radius,
                 bkg_estimator=MedianBackground()):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.bkg_estimator = bkg_estimator
        self._aperture = CircularAnnulus((0, 0), inner_radius, outer_radius)

    def __call__(self, data, x, y, mask=None):
        """
        Measure the local background in a circular annulus.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The 2D array on which to measure the local background.

        x, y : float or 1D float `~numpy.nddarray`
            The aperture center (x, y) position(s) at which to measure
            the local background.

        mask : 2D bool `~numpy.ndarray`, optional
            A boolean mask with the same shape as ``data`` where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked data are excluded from all calculations.

        Returns
        -------
        value : float or 1D float `~numpy.ndarray`
            The local background values.
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        self._aperture.positions = np.array(list(zip(x, y)))
        apermasks = self._aperture.to_mask(method='center')

        bkg = []
        for apermask in apermasks:
            values = apermask.get_values(data, mask=mask)
            bkg.append(self.bkg_estimator(values))
        bkg = np.array(bkg)

        if bkg.size == 1:
            bkg = bkg[0]

        return bkg
