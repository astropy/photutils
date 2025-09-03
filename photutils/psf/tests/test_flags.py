# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the flags module.
"""

import numpy as np
import pytest

from photutils.psf import decode_psf_flags


def test_decode_psf_flags():
    """
    Test the decode_psf_flags standalone function.
    """
    # Test single flag value with no flags set
    decoded = decode_psf_flags(0)
    assert decoded == []
    assert isinstance(decoded, list)

    # Test single flag value with one bit set
    decoded = decode_psf_flags(1)
    assert decoded == ['npixfit_partial']

    decoded = decode_psf_flags(2)
    assert decoded == ['outside_bounds']

    decoded = decode_psf_flags(4)
    assert decoded == ['negative_flux']

    decoded = decode_psf_flags(8)
    assert decoded == ['no_convergence']

    decoded = decode_psf_flags(16)
    assert decoded == ['no_covariance']

    decoded = decode_psf_flags(32)
    assert decoded == ['near_bound']

    decoded = decode_psf_flags(64)
    assert decoded == ['no_overlap']

    decoded = decode_psf_flags(128)
    assert decoded == ['fully_masked']

    decoded = decode_psf_flags(256)
    assert decoded == ['too_few_pixels']

    # Test combination of flags
    decoded = decode_psf_flags(5)  # bits 1 and 4
    assert set(decoded) == {'npixfit_partial', 'negative_flux'}
    assert len(decoded) == 2

    decoded = decode_psf_flags(136)  # bits 8 and 128
    assert set(decoded) == {'no_convergence', 'fully_masked'}
    assert len(decoded) == 2

    # Test with all flags set
    all_flags = 1 + 2 + 4 + 8 + 16 + 32 + 64 + 128 + 256  # 511
    decoded = decode_psf_flags(all_flags)
    expected_all = ['npixfit_partial', 'outside_bounds', 'negative_flux',
                    'no_convergence', 'no_covariance', 'near_bound',
                    'no_overlap', 'fully_masked', 'too_few_pixels']
    assert set(decoded) == set(expected_all)
    assert len(decoded) == 9

    # Test with array input
    flags_array = [0, 1, 2, 5]
    decoded_list = decode_psf_flags(flags_array)
    assert len(decoded_list) == 4
    assert isinstance(decoded_list, list)

    # Check individual results
    assert decoded_list[0] == []
    assert decoded_list[1] == ['npixfit_partial']
    assert decoded_list[2] == ['outside_bounds']
    assert set(decoded_list[3]) == {'npixfit_partial', 'negative_flux'}

    # Test with numpy array
    flags_np = np.array([8, 16, 32])
    decoded_list = decode_psf_flags(flags_np)
    assert len(decoded_list) == 3
    assert decoded_list[0] == ['no_convergence']
    assert decoded_list[1] == ['no_covariance']
    assert decoded_list[2] == ['near_bound']

    # Test with 0-d numpy array (scalar array)
    flag_scalar = np.array(64)
    decoded = decode_psf_flags(flag_scalar)
    assert isinstance(decoded, list)
    assert decoded == ['no_overlap']

    # Test membership operations (common usage pattern)
    issues = decode_psf_flags(136)
    assert 'no_convergence' in issues
    assert 'fully_masked' in issues
    assert 'negative_flux' not in issues

    # Test error conditions
    with pytest.raises(TypeError, match='Flag value must be an integer'):
        decode_psf_flags(3.14)

    with pytest.raises(TypeError, match='Flag value must be an integer'):
        decode_psf_flags('invalid')

    with pytest.raises(TypeError, match='Flag value must be an integer'):
        decode_psf_flags([1, 2.5, 3])


def test_decode_psf_flags_practical_usage():
    """
    Test practical usage patterns for decode_psf_flags.
    """
    # Simulate some typical flag values
    typical_flags = [0, 1, 8, 64, 136, 256]

    # Test batch processing
    all_issues = decode_psf_flags(typical_flags)
    assert len(all_issues) == len(typical_flags)

    # Test filtering for specific conditions
    convergence_issues = [i for i, issues in enumerate(all_issues)
                          if 'no_convergence' in issues]
    expected_conv_indices = [2, 4]  # flags 8 and 136 have convergence issues
    assert convergence_issues == expected_conv_indices

    # Test counting issues
    issue_counts = {}
    for issues in all_issues:
        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    # Verify expected counts
    assert issue_counts.get('no_convergence', 0) == 2  # flags 8 and 136
    assert issue_counts.get('fully_masked', 0) == 1  # flag 136
    assert issue_counts.get('npixfit_partial', 0) == 1  # flag 1

    # Test boolean context (empty list is falsy)
    clean_sources = [i for i, issues in enumerate(all_issues) if not issues]
    assert 0 in clean_sources  # flag 0 should have no issues

    # Test string formatting for reporting
    for i, issues in enumerate(all_issues):
        if issues:
            report = f"Source {i}: {', '.join(issues)}"
            assert isinstance(report, str)
            assert str(i) in report


def test_decode_psf_flags_edge_cases():
    """
    Test edge cases for decode_psf_flags.
    """
    # Test with very large flag value (all bits set + extra)
    large_flag = 2**16 - 1  # Much larger than our defined flags
    decoded = decode_psf_flags(large_flag)
    expected_all = ['npixfit_partial', 'outside_bounds', 'negative_flux',
                    'no_convergence', 'no_covariance', 'near_bound',
                    'no_overlap', 'fully_masked', 'too_few_pixels']
    assert set(decoded) == set(expected_all)

    # Test with negative flag (should still work)
    # Note: This is technically invalid input but numpy ints can be negative
    negative_flag = -1  # All bits set in two's complement
    decoded = decode_psf_flags(negative_flag)
    assert len(decoded) == 9  # All our flags should be detected
    # Test with empty array
    empty_array = np.array([], dtype=int)
    decoded = decode_psf_flags(empty_array)
    assert decoded == []

    # Test with 2D array (should flatten)
    flag_2d = np.array([[0, 1], [8, 136]])
    decoded = decode_psf_flags(flag_2d)
    assert len(decoded) == 4  # Flattened to 4 elements
    assert decoded[0] == []
    assert decoded[1] == ['npixfit_partial']
    assert decoded[2] == ['no_convergence']
    assert set(decoded[3]) == {'no_convergence', 'fully_masked'}
