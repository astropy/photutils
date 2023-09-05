# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides classes to perform grouping of stars.
"""

from collections import defaultdict

import numpy as np

__all__ = ['SourceGrouper']


class SourceGrouper:
    """
    Class to group sources into clusters based on a minimum separation
    distance.

    The groups are formed using hierarchical agglomerative
    clustering with a distance criterion, calling the
    `scipy.cluster.hierarchy.fclusterdata` function.

    Parameters
    ----------
    min_separation : float
        The minimum distance (in pixels) such that any two sources
        separated by less than this distance will be placed in the same
        group if the ``min_size`` criteria is also met.
    """
    def __init__(self, min_separation):
        self.min_separation = min_separation

    def __call__(self, x, y):
        """
        Group sources into clusters based on a minimum distance
        criteria.

        Parameters
        ----------
        x, y : 1D float `~numpy.ndarray`
            The 1D arrays of the x and y centroid coordinates of the
            sources.

        Returns
        -------
        result : 1D int `~numpy.ndarray`
            A 1D array of the groups, in the same order as the input x
            and y coordinates.
        """
        return self._group_sources(x, y)

    def _group_sources(self, x, y):
        """
        Group sources into clusters based on a minimum distance
        criteria.

        Parameters
        ----------
        x, y : 1D float `~numpy.ndarray`
            The 1D arrays of the x and y centroid coordinates of the
            sources.

        Returns
        -------
        result : 1D int `~numpy.ndarray`
            A 1D array of the groups, in the same order as the input x
            and y coordinates.
        """
        from scipy.cluster.hierarchy import fclusterdata

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        if x.shape != y.shape:
            raise ValueError('x and y must have the same shape')
        if x.shape == (0,):  # no sources
            raise ValueError('x and y must not be empty')
        if x.shape == (1,):  # single source -> single group
            return np.array([1])

        xypos = np.transpose((x, y))
        group_id = fclusterdata(xypos, t=self.min_separation,
                                criterion='distance')

        # reorder the group_ids so that that unique group_ids start from 1
        # and increase (this matches the output of DBSCAN)
        mapping = defaultdict(lambda: len(mapping) + 1)
        return np.array([mapping[group] for group in group_id])
