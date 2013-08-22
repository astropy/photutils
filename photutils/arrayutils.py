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
    left_bottom = np.array(position) - np.array(small_array_shape) // 2
    right_top = np.array(position) + np.array(small_array_shape) // 2 + 1

    # Set up slices in x direction
    if left_bottom[1] < 0:
        log.debug('Left side out of range')
        s_x = slice(0, right_top[1])
        b_x = slice(-left_bottom[1], right_top[1] - left_bottom[1])

    elif right_top[1] > large_width:
        log.debug('Right side out of range')
        s_x = slice(left_bottom[1], large_width)
        b_x = slice(0, large_width - left_bottom[1])

    elif left_bottom[1] < 0 and right_top[1] > large_width:
        s_x = slice(0, large_width)
        b_x = slice(-left_bottom[1], large_width - left_bottom[1])

    else:
        s_x = slice(left_bottom[1], right_top[1])
        b_x = slice(0, right_top[1] - left_bottom[1])

    # Set up slices in y direction
    if left_bottom[0] < 0:
        log.debug('Bottom side out of range')
        s_y = slice(0, right_top[0])
        b_y = slice(-left_bottom[0], right_top[0] - left_bottom[0])

    elif right_top[0] > large_height:
        log.debug('Top side out of range')
        s_y = slice(left_bottom[0], large_height)
        b_y = slice(0, large_height - left_bottom[0])

    elif left_bottom[0] < 0 and right_top[0] > large_height:
        s_y = slice(0, large_height)
        b_y = slice(-left_bottom, large_height - left_bottom[0])

    else:
        s_y = slice(left_bottom[0], right_top[0])
        b_y = slice(0, right_top[0] - left_bottom[0])

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
    # Check if sizes of array match
    if array_large.shape > array_small.shape:
        s_y_large, s_x_large, s_y_small, s_x_small = _get_slices(array_large.shape, array_small.shape, position)
        res = array_large[s_y_large, s_x_large] + array_small[s_y_small, s_x_small]
        return res