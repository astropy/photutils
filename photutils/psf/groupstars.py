# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides classes to perform grouping of stars.
"""

import abc

import numpy as np
from astropy.table import Column
from astropy.utils.decorators import deprecated

__all__ = ['DAOGroup', 'DBSCANGroup', 'GroupStarsBase']


@deprecated('1.9.0', alternative='`photutils.psf.SourceGrouper`')
class GroupStarsBase(metaclass=abc.ABCMeta):
    """
    This base class provides the basic interface for subclasses that
    are capable of classifying stars in groups.
    """

    def __call__(self, starlist):
        """
        Classify stars into groups.

        Parameters
        ----------
        starlist : `~astropy.table.Table`
            List of star positions. Columns named as ``x_0`` and
            ``y_0``, which corresponds to the centroid coordinates of
            the sources, must be provided.

        Returns
        -------
        group_starlist : `~astropy.table.Table`
            ``starlist`` with an additional column named ``group_id``
            whose unique values represent groups of mutually overlapping
            stars.
        """
        return self.group_stars(starlist)

    @abc.abstractmethod
    def group_stars(self, starlist):
        """
        Classify stars into groups.

        Parameters
        ----------
        starlist : `~astropy.table.Table`
            List of star positions. Columns named as ``x_0`` and
            ``y_0``, which corresponds to the centroid coordinates of
            the sources, must be provided.

        Returns
        -------
        group_starlist : `~astropy.table.Table`
            ``starlist`` with an additional column named ``group_id``
            whose unique values represent groups of mutually overlapping
            stars.
        """
        raise NotImplementedError('Needs to be implemented in a subclass.')


@deprecated('1.9.0', alternative='`photutils.psf.SourceGrouper`')
class DAOGroup(GroupStarsBase):
    """
    This class implements the DAOGROUP algorithm presented by
    Stetson (1987).

    The method ``group_stars`` divides an entire starlist into sets of
    distinct, self-contained groups of mutually overlapping stars.
    It accepts as input a list of stars and determines which stars are close
    enough to be capable of adversely influencing each others' profile fits.

    Parameters
    ----------
    crit_separation : float or int
        Distance, in units of pixels, such that any two stars separated by
        less than this distance will be placed in the same group.

    See Also
    --------
    photutils.detection.DAOStarFinder

    Notes
    -----
    Assuming the psf fwhm to be known, ``crit_separation`` may be set to
    k*fwhm, for some positive real k.

    References
    ----------
    [1] Stetson, Astronomical Society of the Pacific, Publications,
        (ISSN 0004-6280), vol. 99, March 1987, p. 191-222.
        Available at:
        https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract
    """

    def __init__(self, crit_separation):
        self.crit_separation = crit_separation

    @property
    def crit_separation(self):
        return self._crit_separation

    @crit_separation.setter
    def crit_separation(self, crit_separation):
        if not isinstance(crit_separation, (float, int)):
            raise ValueError('crit_separation is expected to be either float'
                             f'or int. Received {type(crit_separation)}.')

        if crit_separation < 0.0:
            raise ValueError('crit_separation is expected to be a positive '
                             f'real number. Got {crit_separation}.')

        self._crit_separation = crit_separation

    def group_stars(self, starlist):
        cstarlist = starlist.copy()

        if 'id' not in cstarlist.colnames:
            cstarlist.add_column(Column(name='id',
                                        data=np.arange(len(cstarlist)) + 1))
        cstarlist.add_column(Column(name='group_id',
                                    data=np.zeros(len(cstarlist),
                                                  dtype=int)))

        if not np.array_equal(cstarlist['id'], np.arange(len(cstarlist)) + 1):
            raise ValueError('id column must be an integer-valued sequence '
                             f'starting from 1. Got {cstarlist["id"]}')

        n = 1
        while (cstarlist['group_id'] == 0).sum() > 0:
            init_star = cstarlist[np.where(cstarlist['group_id'] == 0)[0][0]]
            index = self.find_group(init_star,
                                    cstarlist[cstarlist['group_id'] == 0])
            cstarlist['group_id'][index - 1] = n
            k = 1
            K = len(index)
            while k < K:
                init_star = cstarlist[cstarlist['id'] == index[k]]
                tmp_index = self.find_group(
                    init_star, cstarlist[cstarlist['group_id'] == 0])
                if len(tmp_index) > 0:
                    cstarlist['group_id'][tmp_index - 1] = n
                    index = np.append(index, tmp_index)
                    K = len(index)
                k += 1
            n += 1

        return cstarlist

    def find_group(self, star, starlist):
        """
        Find the ids of those stars in ``starlist`` which are at a
        distance less than ``crit_separation`` from ``star``.

        Parameters
        ----------
        star : `~astropy.table.Row`
            Star which will be either the head of a cluster or an
            isolated one.

        starlist : `~astropy.table.Table`
            List of star positions. Columns named as ``x_0`` and
            ``y_0``, which corresponds to the centroid coordinates of
            the sources, must be provided.

        Returns
        -------
        result : `~numpy.ndarray`
            Array containing the ids of those stars which are at a
            distance less than ``crit_separation`` from ``star``.
        """
        star_distance = np.hypot(star['x_0'] - starlist['x_0'],
                                 star['y_0'] - starlist['y_0'])
        distance_criteria = star_distance < self.crit_separation
        return np.asarray(starlist[distance_criteria]['id'])


@deprecated('1.9.0', alternative='`photutils.psf.SourceGrouper`')
class DBSCANGroup(GroupStarsBase):
    """
    Class to create star groups according to a distance criteria using
    the Density-based Spatial Clustering of Applications with Noise
    (DBSCAN) from scikit-learn.

    Parameters
    ----------
    crit_separation : float or int
        Distance, in units of pixels, such that any two stars separated
        by less than this distance will be placed in the same group.
    min_samples : int, optional (default=1)
        Minimum number of stars necessary to form a group.
    metric : str or callable (default='euclidean')
        The metric to use when calculating distance between each pair of
        stars.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        The algorithm to be used to actually find nearest neighbors.
    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or cKDTree.

    Notes
    -----
    * The attribute ``crit_separation`` corresponds to ``eps`` in
      `sklearn.cluster.DBSCAN
      <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN>`_.

    * This class provides more general algorithms than
      `photutils.psf.DAOGroup`.  More precisely,
      `photutils.psf.DAOGroup` is a special case of
      `photutils.psf.DBSCANGroup` when ``min_samples=1`` and
      ``metric=euclidean``.  Additionally, `photutils.psf.DBSCANGroup`
      may be faster than `photutils.psf.DAOGroup`.

    References
    ----------
    [1] Scikit Learn DBSCAN.
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
    """

    def __init__(self, crit_separation, *, min_samples=1, metric='euclidean',
                 algorithm='auto', leaf_size=30):
        self.crit_separation = crit_separation
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size

    def group_stars(self, starlist):
        from sklearn.cluster import DBSCAN

        cstarlist = starlist.copy()

        if 'id' not in cstarlist.colnames:
            cstarlist.add_column(Column(name='id',
                                        data=np.arange(len(cstarlist)) + 1))

        if not np.array_equal(cstarlist['id'], np.arange(len(cstarlist)) + 1):
            raise ValueError('id column must be an integer-valued sequence '
                             'starting from 1. Got {cstarlist["id"]}')

        pos_stars = np.transpose((cstarlist['x_0'], cstarlist['y_0']))
        dbscan = DBSCAN(eps=self.crit_separation,
                        min_samples=self.min_samples, metric=self.metric,
                        algorithm=self.algorithm, leaf_size=self.leaf_size)
        cstarlist['group_id'] = (dbscan.fit(pos_stars).labels_
                                 + np.ones(len(cstarlist), dtype=int))
        return cstarlist
