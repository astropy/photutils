# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions for implementing the DAOGROUP algorithm proposed by Stetson in
Astronomical Society of the Pacific, Publications, (ISSN 0004-6280),
vol. 99, March 1987, p. 191-222.
Available at: http://adsabs.harvard.edu/abs/1987PASP...99..191S
"""

from __future__ import division
import numpy as np
from astropy.table import Column, Table, vstack


class GroupFinder(metaclass=abc.ABCMeta):
    @abstractmethod
    def __call__(self, starlist):
        pass


class DAOGroup(GroupFinder)
    """
    This is an implementation of the DAOGROUP algorithm presented by
    Stetson (1987).

    daogroup divides an entire starlist into sets of distinct, self-contained
    groups of mutually overlapping stars. It accepts as input a list of stars
    and determines which stars are close enough to be capable of adversely
    influencing each others' profile fits.

    Parameters
    ----------
    starlist : `~astropy.table.Table`
        List of stars positions. Columns named as 'x_0' and 'y_0' must be
        provided.
    crit_separation : float or int
        Distance, in units of pixels, such that any two stars separated by
        less than this distance will be placed in the same group.

    Returns
    -------
    group_starlist : list of `~astropy.table.Table`
        Each `~astropy.table.Table` in the list corresponds to a group of
        mutually overlapping stars.

    Notes
    -----
    Assuming the psf fwhm to be known, 'crit_separation' may be set to
    k*fwhm, for some positive real k.

    See
    ---
    `~daofind`
    """

    def __init__(self, crit_separation):
        self.crit_separation = crit_separation

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
 
    def __call__(self, starlist):
        group_starlist = []
        cstarlist = starlist.copy()

        if 'id' not in cstarlist.colnames:
            cstarlist.add_column(Column(name='id',
                                        data=np.arange(len(cstarlist))))
        
        while len(cstarlist) is not 0:
            init_group = _find_group(cstarlist[0], cstarlist,
                                     self.crit_separation)
            assigned_stars_ids = np.intersect1d(cstarlist['id'],
                                                init_group['id'],
                                                assume_unique=True)
            cstarlist = _remove_stars(cstarlist, assigned_stars_ids)
            n = 1
            N = len(init_group)
            while(n < N):    
                tmp_group = _find_group(init_group[n], cstarlist,
                                        self.crit_separation)
                if len(tmp_group) > 0:
                    assigned_stars_ids = np.intersect1d(cstarlist['id'],
                                                        tmp_group['id'],
                                                        assume_unique=True)
                    cstarlist = _remove_stars(cstarlist, assigned_stars_ids)
                    init_group = vstack([init_group, tmp_group])
                    N = len(init_group)
                n = n + 1
            group_starlist.append(init_group)
        return group_starlist

def _find_group(star, starlist, crit_separation):
    """
    Find those stars in `starlist` which are at a distance of
    `crit_separation` from `star`.

    Parameters
    ----------
    star : `~astropy.table.Row`
        Star which will be either the head of a cluster or an isolated one.
    
    starlist : `~astropy.table.Table`

    Returns
    -------
    `~astropy.table.table.Table` containing those stars which are at a distance
    less than `crit_separation` from `star`.
    """
    
    star_distance = np.hypot(star['x_0'] - starlist['x_0'],
                             star['y_0'] - starlist['y_0'])
    distance_criteria = star_distance < crit_separation
    return starlist[distance_criteria]

def _remove_stars(starlist, stars_ids):
    """
    Remove stars from `starlist` whose ids are in `stars_ids`.

    Parameters
    ----------
    starlist : `~astropy.table.Table`
        Star list from which stars will be removed.

    stars_ids : numpy.ndarray
        IDs of the stars which will be removed.

    Returns
    -------
    Reduced `~astropy.table.Table` containing only the stars whose ids are not
    listed in `stars_ids`.
    """
    
    for i in range(len(stars_ids)):
        starlist.remove_rows(np.where(starlist['id'] == stars_ids[i])[0])
    return starlist
