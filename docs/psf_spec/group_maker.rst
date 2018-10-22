GroupMaker
==========

Documented as the ``__call__`` method of ``GroupStarsBase`` - see
https://photutils.readthedocs.io/en/stable/api/photutils.psf.groupstars.GroupStarsBase.html.
The API for the group maker may be subject to change if ``group_stars`` requires changes
to accommodate future revisions to the PSF fitting process -- primarily it would require
updates if the PSF fitting is extended to include non-point sources and "scene maker"
functionality. These large changes would subsequently require significant changes to the
PSF fitting routines (e.g., ``IterativelySubtractedPSFPhotometry``) as a whole.

A function which groups stars within some critical separation, returning potentially
overlapping sources with an additonal column indicating their common group members.
Subclasses of ``GroupStarsBase`` may require further input parameters, such as 
``crit_separation`` required for ``DAOGroup``.

The main functionality of this routine is within ``group_stars``, which accepts
``starlist``, a `~astropy.table.Table` of sources with centroid positions, and returns
``group_starlist``, the same `~astropy.table.Table` passed to the routine but with an
additional column indicating the group number of the sources. These sources are grouped
by the individual criteria specified within the ``group_stars`` function defined by the
``GroupStarsBase`` subclass in question.

Parameters
----------

starlist : `~astropy.table.Table`
    List of star positions; ``x_0`` and ``y_0``, the centroids of the sources, must be
    provided.

Returns
-------

group_starlist : `~astropy.table.Table`
    ``starlist`` with an additional column appended -- ``group_id`` gives unique
    group number of mutually overlapping sources.


Methods
-------

find_group
^^^^^^^^^^^

Convenience function provided by ``DAOGroup`` returning all objects within
``crit_separation`` from a given star from ``starlist``.

Parameters
""""""""""

star : `~astropy.table.Row`
    Single star from ``starlist`` whose mutual group members are required.
starlist : `~astropy.table.Table`
    List of star positions; ``x_0`` and ``y_0``, the centroids of the sources, must be
    provided.

Returns
"""""""

`~numpy.array` with the IDs of all stars with distance less than ``crit_separation`` to ``star``.


Example Usage
-------------

Here we create a ``DAOGroup`` list of overlapping sources, then find all sources within 3 pixels
of the first source in the list.::

    from photutils.psf.groupstars import DAOGroup
    group = DAOGroup(starlist, crit_separation=3)
    stargroups = group.group_stars(starlist)
    id_overlap = group.find_stars(starlist[0], starlist)
