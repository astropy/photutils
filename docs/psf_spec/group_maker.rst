GroupMaker
==========

Documented as the ``__call__`` method of ``GroupStarsBase`` - see
https://photutils.readthedocs.io/en/stable/api/photutils.psf.groupstars.GroupStarsBase.html
API may potentially change if group_stars is updated to scene_maker extending PSF fitting to
non-point source objects; however, it is likely that the fundamental inputs and outputs
remain at least functionally similar to those shown here. Large changes to the GroupMaker
call may require significant changes to the PSF Photometry fitting routines (e.g.,
``IterativelySubtractedPSFPhotometry``).

A function which groups stars within some critical separation, returning potentially
overlapping sources with an additonal column indicating their common group members.
Subclasses of ``GroupStarsBase`` may require further input parameters, such as 
``crit_separation`` required for ``DAOGroup``.

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
