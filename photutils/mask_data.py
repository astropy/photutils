# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions for creating a cut out of the data and a mask of the same cut out
corresponding to the aperture"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import warnings
import astropy.units as u
from astropy.utils.exceptions import AstropyUserWarning


__all__ = ['get_cutouts']



def get_cutouts(data, circular_aperture, use_exact = 0, subpixels = 5):
    '''
    Create a square cutout of each aperture and an associated mask which denotes the
    aperture with 1 for inside, 0 for outside, and a fraction for the edges.

    Parameters:
    -----------
    data : Image array
    circular_aperture: photutils aperture object
    use_exact : (default = 0) refers to the keyword in .geometry.circular_overlap_grid.
            If use_exact = 1, a hard boundary containing only the pixels fully contained
            within the aperture. if use_exact = 0, calculates the fractional pixel
            contained within the pixel
    subpixels : (default = 5) if use_exact = 0 this is used to determine the fraction of
            the pixel contained within the aperture.

    Returns:
    -----------
    img_mask_list : a list with one tuple for each object position in the circular_aperture
        array. Each tuple contains a 2D array with the cutout of the image which contains
        the complete aperture and a 2D array with a boolean mask which is True inside the
        aperture, False outside the aperture, and gives a fraction inside the aperture for
        edge pixels.



    '''

    positions = circular_aperture.positions
    radius = circular_aperture.r
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
            return [(np.array([]), np.array([]))]

    #Compute cutout for each object in image
    img_mask_list = []
    absolute_extent = reformat_extents(absolute_extent)
    centered_extent = reformat_extents(centered_extent)
    for indiv_obj in range(len(no_overlap)):

        fractional_overlap_mask = get_fractional_overlap_mask(absolute_extent[indiv_obj],
                                                                centered_extent[indiv_obj],
                                                                ~no_overlap[indiv_obj],
                                                                radius, use_exact,
                                                                subpixels)
        img_stamp = cutout_image(data, absolute_extent[indiv_obj], ~no_overlap[indiv_obj])
        img_mask_list.append((img_stamp, fractional_overlap_mask))
    return img_mask_list

def get_fractional_overlap_mask(indiv_absolute_extent, indiv_centered_extent, overlap,
                                radius, use_exact, subpixels):
    '''
    given a set of coordinates, calculate the masked array for the cutout. Returns an
    empty array if there is no overlap between the data and the aperture.
    '''


    from .geometry import circular_overlap_grid

    x_min, x_max, y_min, y_max = indiv_absolute_extent
    x_pmin, x_pmax, y_pmin, y_pmax = indiv_centered_extent

    if overlap:
        return circular_overlap_grid(x_pmin, x_pmax,
                                             y_pmin, y_pmax,
                                             x_max - x_min,
                                             y_max - y_min,
                                             radius, use_exact, subpixels)
    else:
        #Should this raise a warning or have we already done this once and that is enough?
        return np.array([])


def reformat_extents(extents):
	'''
	Extents are returned as a list of xmin, list of xmax, list of ymin, list of ymax, I
	would like to group them by object as (xmin, xmax, ymin, ymax) for each object.
	'''
	obj_group_extents = []
	for xmin, xmax, ymin, ymax in zip(extents[0], extents[1], extents[2], extents[3]):
		obj_group_extents.append((xmin, xmax, ymin, ymax))
	return obj_group_extents


def cutout_image(data, indiv_absolute_extent, overlap):
    '''
    Given a set of coordinates, return a cutout corresponding to the square containing
    the circular aperture at a given position
    '''
    if overlap:
        x_min, x_max, y_min, y_max = indiv_absolute_extent
        return data[y_min:y_max, x_min:x_max]
    else:
        return np.array([])