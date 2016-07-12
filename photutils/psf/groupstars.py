# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Module which provides classes to perform source grouping."""

from __future__ import division
import abc
import numpy as np
from astropy.table import Column, Table, vstack


__all__ = ['DAOGroup', 'GroupStarsBase']


class GroupStarsBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def group_stars(self, starlist):
        pass


class DAOGroup(GroupStarsBase):
    """
    This is class implements the DAOGROUP algorithm presented by
    Stetson (1987).

    The method ``group_stars`` divides an entire starlist into sets of
    distinct, self-contained groups of mutually overlapping stars.
    It accepts as input a list of stars and determines which stars are close
    enough to be capable of adversely influencing each others' profile fits.

    Attributes
    ---------- 
    crit_separation : float or int
        Distance, in units of pixels, such that any two stars separated by
        less than this distance will be placed in the same group. 

    Notes
    -----
    Assuming the psf fwhm to be known, ``crit_separation`` may be set to
    k*fwhm, for some positive real k.

    See
    ---
    `~daofind`

    References
    ----------
    [1] Stetson, Astronomical Society of the Pacific, Publications,
        (ISSN 0004-6280), vol. 99, March 1987, p. 191-222.
        Available at: http://adsabs.harvard.edu/abs/1987PASP...99..191S
    """

    def __init__(self, crit_separation):
        self.crit_separation = crit_separation
    
    def __call__(self, starlist):
        """
        Parameters
        ----------
        starlist : `~astropy.table.Table`
            List of stars positions. Columns named as ``x_0`` and ``y_0``,
            which corresponds to the centroid coordinates of the sources,
            must be provided.

        Returns
        -------
        group_starlist : `~astropy.table.Table`
            ``starlist`` with an additional column named ``group_id`` whose
            unique values represent groups of mutually overlapping stars.
        """

        return self.group_stars(starlist)

    @property
    def crit_separation(self):
        return self._crit_separation
    
    @crit_separation.setter
    def crit_separation(self, crit_separation):
        if not isinstance(crit_separation, (float, int)):
            raise ValueError('crit_separation is expected to be either '+
                             'float or int. Received {}.'\
                             .format(type(crit_separation)))
        elif crit_separation < 0.0:
            raise ValueError('crit_separation is expected to be a positive '+
                             'real number. Got {}'.format(crit_separation))
        else:
            self._crit_separation = crit_separation
 
    def group_stars(self, starlist):
        cstarlist = starlist.copy()

        if 'id' not in cstarlist.colnames:
            cstarlist.add_column(Column(name='id',
                                        data=np.arange(len(cstarlist)) + 1))
        cstarlist.add_column(Column(name='group_id',\
                                data=np.zeros(len(cstarlist), dtype=np.int)))

        if not np.array_equal(cstarlist['id'], np.arange(len(cstarlist)) +1):
            raise ValueError('id colum must be an integer-valued ' +
                             'sequence starting from 1. ' +
                             'Got {}'.format(cstarlist['id']))

        n = 1
        while (cstarlist['group_id'] == 0).sum() > 0:
            init_star = cstarlist[np.where(cstarlist['group_id'] == 0)[0][0]]
            index = self.find_group(init_star,
                                    cstarlist[cstarlist['group_id'] == 0])
            cstarlist['group_id'][index-1] = n
            k = 1
            K = len(index)
            while k < K:
                init_star = cstarlist[cstarlist['id'] == index[k]]
                tmp_index = self.find_group(init_star,\
                                        cstarlist[cstarlist['group_id'] == 0])
                if len(tmp_index) > 0:
                    cstarlist['group_id'][tmp_index-1] = n
                    index = np.append(index, tmp_index)
                    K = len(index)
                k += 1
            n += 1

        return cstarlist

    def find_group(self, star, starlist):
        """
        Find the ids of those stars in ``starlist`` which are at a distance less
        than ``crit_separation`` from ``star``.

        Parameters
        ----------
        star : `~astropy.table.Row`
            Star which will be either the head of a cluster or an isolated one.
        starlist : `~astropy.table.Table`
            List of stars positions. Columns named as ``x_0`` and ``y_0``,
            which corresponds to the centroid coordinates of the sources,
            must be provided.
        
        Returns
        -------
        Array containing the ids of those stars which are at a distance less
        than ``crit_separation`` from ``star``.
        """
        
        star_distance = np.hypot(star['x_0'] - starlist['x_0'],
                                 star['y_0'] - starlist['y_0'])
        distance_criteria = star_distance < self.crit_separation
        return np.asarray(starlist[distance_criteria]['id'])
