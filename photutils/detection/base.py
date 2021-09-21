# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module implements the base classes for detecting stars in an
astronomical image. Each star-finding class should define a method
called ``find_stars`` that finds stars in an image.
"""

import abc

__all__ = ['StarFinderBase']


class StarFinderBase(metaclass=abc.ABCMeta):
    """
    Abstract base class for star finders.
    """

    def __call__(self, data, mask=None):
        return self.find_stars(data, mask=mask)

    @abc.abstractmethod
    def find_stars(self, data, mask=None):
        """
        Find stars in an astronomical image.

        Parameters
        ----------
        data : 2D array_like
            The 2D image array.

        mask : 2D bool array, optional
            A boolean mask with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked pixels are ignored when searching for
            stars.

        Returns
        -------
        table : `~astropy.table.Table` or `None`
            A table of found stars. If no stars are found then `None` is
            returned.
        """
        raise NotImplementedError('Needs to be implemented in a subclass.')
