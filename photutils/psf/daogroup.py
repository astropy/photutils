# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions for implementing the DAOGROUP algorithm proposed by Stetson 1987.
"""

from __future__ import division
import numpy as np
from astropy.table import Column, Table, vstack

__all__ = ['daogroup']

def daogroup(starlist, crit_separation):
    """
    This is an implementation which follows the DAO GROUP algorithm presented
    by Stetson (1987).

    GROUP divides an entire starlist into sets of distinct, self-contained
    groups of mutually overlapping stars.

    GROUP accepts as input a list of stars and determines which stars are
    close enough to be capable of adversely influencing each others' profile
    fits.

    Parameters
    ----------
    starlist : `~astropy.table.Table` or array-like
        List of stars positions.
        If `~astropy.table.Table`, columns should be named 'x_0' and 'y_0'.
        TODO: If array-like, it should be either (x_0, y_0).
        If 'starlist' only contains x_0 and y_0, 'crit_separation' must be
        provided.
    crit_separation : float or int
        Distance, in units of pixels, such that any two stars separated by
        less than this distance will be placed in the same group.
        TODO: If None, 'flux_0' must be provided in 'starlist'.
    
    Returns
    -------
    group_starlist : list of `~astropy.table.Table`
        Each `~astropy.table.Table` in the list corresponds to a group of
        mutually overlapping starts.

    Notes
    -----
    Assuming the psf fwhm to be known, 'crit_separation' may be set to
    k*fwhm, for some positive real k.

    See
    ---
    `~daofind`
    """

    if not isinstance(crit_separation, (float, int)):
        raise ValueError('crit_separation is expected to be either float or' +
                         ' int. Received {}.'.format(type(crit_separation)))
    if crit_separation < 0.0:
        raise ValueError('crit_separation is expected to be a positive' +
                         'real number.')

    ## write a method that varifies whether the starlist given by the user
    ## is valid

    group_starlist = []
    cstarlist = starlist.copy()

    if 'id' not in cstarlist.colnames:
        cstarlist.add_column(Column(name='id',
                                    data=np.arange(len(cstarlist))))
    
    while len(cstarlist) is not 0:
        init_group = _find_group(cstarlist[0], cstarlist, crit_separation)
        assigned_stars_ids = np.intersect1d(cstarlist['id'], init_group['id'],
                                            assume_unique=True)
        cstarlist = _remove_stars(cstarlist, assigned_stars_ids)
        n = 1
        N = len(init_group)
        while(n < N):    
            tmp_group = _find_group(init_group[n], cstarlist, crit_separation)
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
    star : `~astropy.table.row.Row`
        Star which will be either the head of a cluster or an isolated one.
    
    starlist : `~astropy.table.table.Table`

    Returns
    -------
    `~astropy.table.table.Table` containing those stars which are at a distance
    of `crit_separation` from `star`.
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
    starlist : `~astropy.table.table.Table`
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
