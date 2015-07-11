# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions for creating a cut out of the data and a mask of the same cut out
corresponding to the aperture"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import warnings
import astropy.units as u
from astropy.utils.exceptions import AstropyUserWarning


__all__ = []



def get_cutouts(data, positions, radius, error,
                           pixelwise_error, method, subpixels):

    extents = np.zeros((len(positions), 4), dtype=int)

    extents[:, 0] = positions[:, 0] - radius + 0.5
    extents[:, 1] = positions[:, 0] + radius + 1.5
    extents[:, 2] = positions[:, 1] - radius + 0.5
    extents[:, 3] = positions[:, 1] + radius + 1.5

	from .aperture_funcs import get_phot_extents

    no_overlap, absolute_extent, centered_extent = get_phot_extents(data, positions,
                                                       extents)

	#Check that some of the apertures overlap the data
    if np.sum(no_overlap):
        warnings.warn("The aperture at position {0} does not have any "
                      "overlap with the data"
                      .format(positions[no_overlap]),
                      AstropyUserWarning)
        if np.sum(no_overlap) == len(positions):
        	#FIXME: this should return something of the same form of the final output
            #return (, )
            pass

    if method == 'center':
        use_exact = 0
        subpixels = 1
    elif method == 'subpixel':
        use_exact = 0
    else:
        use_exact = 1
        subpixels = 1

    fractional_overlap_mask = get_fractinoal_overlap_mask(absolute_extent, centered_extent,
    														~no_overlap, use_exact, subpixels)

def get_fractional_overlap_mask(absolute_extent, centered_extent, overlap_arr, use_exact,
								subpixels):
    x_min, x_max, y_min, y_max = absolute_extent
    x_pmin, x_pmax, y_pmin, y_pmax = centered_extent
    from .geometry import circular_overlap_grid

    for i, overlap in enumerate(overlap_arr):

        if overlap:
            fraction = circular_overlap_grid(x_pmin[i], x_pmax[i],
                                             y_pmin[i], y_pmax[i],
                                             x_max[i] - x_min[i],
                                             y_max[i] - y_min[i],
                                             radius, use_exact, subpixels)

    return fraction
