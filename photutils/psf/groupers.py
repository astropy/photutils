# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides classes to perform grouping of stars.
"""

import numpy as np

__all__ = ['SourceGrouper']


class SourceGrouper:
    """
    Class to group sources into clusters based on a minimum separation
    distance.

    This class uses the `"Density-based Spatial Clustering
    of Applications with Noise" clustering algorithm
    <https://scikit-learn.org/stable/modules/clustering.html#dbscan>`_
    from `scikit-learn <https://scikit-learn.org/>`_.

    Parameters
    ----------
    min_separation : float
        The minimum distance (in pixels) such that any two sources
        separated by less than this distance will be placed in the same
        group if the ``min_size`` criteria is also met.

    min_size : int, optional
        The minimum number of sources necessary to form a group.

    metric : str or callable (default='euclidean'), optional
        The metric to use when calculating distance between each pair of
        sources.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        The algorithm to use to find nearest neighbors.

    leaf_size : int, optional (default = 30)
        The leaf size passed to the BallTree or cKDTree function.
        Changing ``leaf_size`` will not affect the results of a query,
        but can significantly impact the speed of a query and the memory
        required to store the constructed tree.

    References
    ----------
    [1] scikit-learn DBSCAN.
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

    Notes
    -----
    ``min_separation`` corresponds to ``eps`` in `sklearn.cluster.DBSCAN
    <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.D
    BSCAN.html>`_.
    """
    def __init__(self, min_separation, *, min_size=1, metric='euclidean',
                 algorithm='auto', leaf_size=30):
        self.min_separation = min_separation
        self.min_size = min_size
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size

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
        from sklearn.cluster import DBSCAN

        xypos = np.transpose((x, y))
        dbscan = DBSCAN(eps=self.min_separation, min_samples=self.min_size,
                        metric=self.metric, algorithm=self.algorithm,
                        leaf_size=self.leaf_size)
        group_id = dbscan.fit(xypos).labels_ + 1

        return group_id
