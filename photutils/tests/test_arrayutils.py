import pytest

import numpy as np

from ..arrayutils import extract_array_2D, add_array_2D


def test_extract_array_2D():
    """
    Test extract_array utility function.
    
    Test by extracting an array of ones out of an array of zeros.
    """
    large_test_array = np.zeros((11, 11))
    small_test_array = np.ones((5, 5))
    large_test_array[3:8, 3:8] = small_test_array
    extracted_array = extract_array_2D(large_test_array, (5, 5), (5, 5))
    assert np.all(extracted_array == small_test_array)


def test_add_array_2D():
    """
    Test add_array_2D utility function.
    
    Test by adding an array of ones out of an array of zeros.
    """
    large_test_array = np.zeros((11, 11))
    small_test_array = np.ones((5, 5))
    large_test_array_ref = large_test_array.copy()
    large_test_array_ref[3:8, 3:8] += small_test_array
    
    added_array = add_array_2D(large_test_array, small_test_array, (5, 5))
    assert np.all(added_array == large_test_array_ref)
