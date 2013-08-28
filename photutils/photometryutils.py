"""
This module includes helper functions for photometry. 
"""
import numpy as np
from astropy import log

__all__ = ['extract_array_2D', 'add_array_2D']


def _get_slices(large_array_shape, small_array_shape, position):
    """
    Get slices for a given small and large array shape and position. 
    """
    large_width = large_array_shape[1]
    large_height = large_array_shape[0]
    y_min, x_min = np.array(position) - np.array(small_array_shape) // 2
    y_max, x_max = np.array(position) + np.array(small_array_shape) // 2 + 1

    # Set up slices in x direction
    if x_min < 0:
        log.debug('Left side out of range')
        s_x = slice(0, x_max)
        b_x = slice(-x_min, x_max - x_min)

    elif x_max > large_width:
        log.debug('Right side out of range')
        s_x = slice(x_min, large_width)
        b_x = slice(0, large_width - x_min)

    elif x_min < 0 and x_max > large_width:
        s_x = slice(0, large_width)
        b_x = slice(-x_min, large_width - x_min)

    else:
        s_x = slice(x_min, x_max)
        b_x = slice(0, x_max - x_min)

    # Set up slices in y direction
    if y_min < 0:
        log.debug('Bottom side out of range')
        s_y = slice(0, y_max)
        b_y = slice(-y_min, y_max - y_min)

    elif y_max > large_height:
        log.debug('Top side out of range')
        s_y = slice(y_min, large_height)
        b_y = slice(0, large_height - y_min)

    elif y_min < 0 and y_max > large_height:
        s_y = slice(0, large_height)
        b_y = slice(-left_bottom, large_height - y_min)

    else:
        s_y = slice(y_min, y_max)
        b_y = slice(0, y_max - y_min)

    return s_x, s_y, b_x, b_y


def extract_array_2D(array_large, shape, position):
    """
    Extract smaller array of given shape and position out of a larger array.

    Parameters
    ----------
    array_large : ndarray
        Array to extract another array from.
    shape : tuple
        Shape of the extracted array.
    position : tuple
        x and y position of the center of the small array.
    """
    # Check if one array is larger than the other
    if array_large.shape > shape:
        s_y_large, s_x_large, s_y_small, s_x_small = _get_slices(array_large.shape, shape, position)
        return array_large[s_y_large, s_x_large]


def add_array_2D(array_large, array_small, position):
    """
    Add a smaller 2D arrays at a given position in a larger 2D array.

    Parameters
    ----------
    array_large : ndarray

    array_small : ndarray
    position : tuple
        x and y position of the center of the small array.
    """
    # Check if one array is larger than the other
    if array_large.shape > array_small.shape:
        s_y_large, s_x_large, s_y_small, s_x_small = _get_slices(array_large.shape, array_small.shape, position)
        res = array_large[s_y_large, s_x_large] + array_small[s_y_small, s_x_small]
        return res