# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the flags module.
"""

import numpy as np
import pytest

from photutils.psf import IterativePSFPhotometry, PSFPhotometry
from photutils.psf.flags import (PSF_FLAGS, _PSFFlagDefinition, _PSFFlags,
                                 _update_decode_docstring, decode_psf_flags)


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

    match = 'Flag value must be a non-negative integer'
    with pytest.raises(ValueError, match=match):
        decode_psf_flags(-2)

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


def test_psf_flags_singleton():
    """
    Test _PSFFlags singleton behavior.
    """
    # Test that PSF_FLAGS is accessible and is a _PSFFlags instance
    assert isinstance(PSF_FLAGS, _PSFFlags)

    # Test that multiple references point to the same object
    flags1 = PSF_FLAGS
    flags2 = PSF_FLAGS
    assert flags1 is flags2

    # Test that creating a new instance works independently
    new_flags = _PSFFlags()
    assert isinstance(new_flags, _PSFFlags)
    assert new_flags is not PSF_FLAGS  # Different instances


def test_psf_flags_constants():
    """
    Test _PSFFlags constant access.
    """
    # Test all flag constants exist and have correct values
    expected_constants = {
        'NPIXFIT_PARTIAL': 1,
        'OUTSIDE_BOUNDS': 2,
        'NEGATIVE_FLUX': 4,
        'NO_CONVERGENCE': 8,
        'NO_COVARIANCE': 16,
        'NEAR_BOUND': 32,
        'NO_OVERLAP': 64,
        'FULLY_MASKED': 128,
        'TOO_FEW_PIXELS': 256,
    }

    for const_name, expected_value in expected_constants.items():
        assert hasattr(PSF_FLAGS, const_name)
        actual_value = getattr(PSF_FLAGS, const_name)
        assert actual_value == expected_value
        assert isinstance(actual_value, int)


def test_psf_flags_properties():
    """
    Test _PSFFlags property access methods.
    """
    # Test bit_values property
    bit_values = PSF_FLAGS.bit_values
    expected_bits = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert set(bit_values) == set(expected_bits)
    assert len(bit_values) == 9

    # Test names property
    names = PSF_FLAGS.names
    expected_names = [
        'npixfit_partial', 'outside_bounds', 'negative_flux',
        'no_convergence', 'no_covariance', 'near_bound',
        'no_overlap', 'fully_masked', 'too_few_pixels',
    ]
    assert set(names) == set(expected_names)
    assert len(names) == 9

    # Test flag_dict property
    flag_dict = PSF_FLAGS.flag_dict
    assert isinstance(flag_dict, dict)
    assert len(flag_dict) == 9
    for bit_val, name in flag_dict.items():
        assert bit_val in expected_bits
        assert name in expected_names

    # Test all_flags property
    all_flags = PSF_FLAGS.all_flags
    assert isinstance(all_flags, list)
    assert len(all_flags) == 9
    for flag_def in all_flags:
        assert isinstance(flag_def, _PSFFlagDefinition)


def test_psf_flags_get_methods():
    """
    Test _PSFFlags getter methods.
    """
    # Test get_name
    assert PSF_FLAGS.get_name(1) == 'npixfit_partial'
    assert PSF_FLAGS.get_name(8) == 'no_convergence'
    assert PSF_FLAGS.get_name(256) == 'too_few_pixels'

    # Test get_bit_value
    assert PSF_FLAGS.get_bit_value('npixfit_partial') == 1
    assert PSF_FLAGS.get_bit_value('no_convergence') == 8
    assert PSF_FLAGS.get_bit_value('too_few_pixels') == 256

    # Test get_description
    desc1 = PSF_FLAGS.get_description(1)
    assert 'npixfit smaller than full fit_shape region' in desc1

    desc8 = PSF_FLAGS.get_description(8)
    assert 'possible non-convergence' in desc8

    # Test get_detailed_description
    detailed1 = PSF_FLAGS.get_detailed_description(1)
    assert 'number of fitted pixels' in detailed1
    assert 'partial PSF fitting' in detailed1

    detailed8 = PSF_FLAGS.get_detailed_description(8)
    assert 'algorithm may not have converged' in detailed8


def test_psf_flags_get_definition():
    """
    Test _PSFFlags get_definition method.
    """
    # Test get_definition by bit value
    def_by_bit = PSF_FLAGS.get_definition(1)
    assert isinstance(def_by_bit, _PSFFlagDefinition)
    assert def_by_bit.bit_value == 1
    assert def_by_bit.name == 'npixfit_partial'

    # Test get_definition by name
    def_by_name = PSF_FLAGS.get_definition('npixfit_partial')
    assert isinstance(def_by_name, _PSFFlagDefinition)
    assert def_by_name.bit_value == 1
    assert def_by_name.name == 'npixfit_partial'

    # Test that both methods return the same object
    assert def_by_bit is def_by_name

    # Test error cases
    with pytest.raises(KeyError, match='No flag with bit value 999'):
        PSF_FLAGS.get_definition(999)

    with pytest.raises(KeyError, match="No flag with name 'invalid'"):
        PSF_FLAGS.get_definition('invalid')

    with pytest.raises(TypeError, match='identifier must be int'):
        PSF_FLAGS.get_definition(3.14)


def test_psf_flag_definition():
    """
    Test _PSFFlagDefinition dataclass.
    """
    # Create a flag definition
    flag_def = _PSFFlagDefinition(
        bit_value=1,
        name='test_flag',
        description='test description',
        detailed_description='detailed test description',
    )

    # Test attributes
    assert flag_def.bit_value == 1
    assert flag_def.name == 'test_flag'
    assert flag_def.description == 'test description'
    assert flag_def.detailed_description == 'detailed test description'

    # Test immutability (frozen dataclass)
    with pytest.raises(AttributeError):
        flag_def.bit_value = 2

    # Test equality
    flag_def2 = _PSFFlagDefinition(
        bit_value=1,
        name='test_flag',
        description='test description',
        detailed_description='detailed test description',
    )
    assert flag_def == flag_def2

    # Test inequality
    flag_def3 = _PSFFlagDefinition(
        bit_value=2,
        name='test_flag',
        description='test description',
        detailed_description='detailed test description',
    )
    assert flag_def != flag_def3


def test_psf_flags_integration_with_decode():
    """
    Test integration between _PSFFlags and decode_psf_flags.
    """
    # Test that decode_psf_flags uses PSF_FLAGS internally
    test_flags = [PSF_FLAGS.NPIXFIT_PARTIAL, PSF_FLAGS.NO_CONVERGENCE,
                  PSF_FLAGS.FULLY_MASKED]

    decoded = decode_psf_flags(test_flags)
    assert len(decoded) == 3
    assert decoded[0] == ['npixfit_partial']
    assert decoded[1] == ['no_convergence']
    assert decoded[2] == ['fully_masked']

    # Test combined flags
    combined = PSF_FLAGS.NO_CONVERGENCE | PSF_FLAGS.FULLY_MASKED
    decoded_combined = decode_psf_flags(combined)
    assert set(decoded_combined) == {'no_convergence', 'fully_masked'}

    # Test all constants work with decode
    for const_name in ['NPIXFIT_PARTIAL', 'OUTSIDE_BOUNDS', 'NEGATIVE_FLUX',
                       'NO_CONVERGENCE', 'NO_COVARIANCE', 'NEAR_BOUND',
                       'NO_OVERLAP', 'FULLY_MASKED', 'TOO_FEW_PIXELS']:
        const_value = getattr(PSF_FLAGS, const_name)
        decoded_const = decode_psf_flags(const_value)
        assert len(decoded_const) == 1

        # The decoded name should match the constant name (lowercase)
        expected_name = const_name.lower()
        assert decoded_const[0] == expected_name


def test_psf_flags_completeness():
    """
    Test that _PSFFlags covers all expected flag scenarios.
    """
    # Test that we have the expected number of flags
    assert len(PSF_FLAGS.all_flags) == 9

    # Test that bit values are powers of 2
    for bit_val in PSF_FLAGS.bit_values:
        assert bit_val > 0
        assert (bit_val & (bit_val - 1)) == 0  # Power of 2 check

    # Test that bit values are unique
    bit_values = PSF_FLAGS.bit_values
    assert len(bit_values) == len(set(bit_values))

    # Test that names are unique
    names = PSF_FLAGS.names
    assert len(names) == len(set(names))

    # Test that all names are valid Python identifiers (for compatibility)
    for name in names:
        assert name.isidentifier()
        assert '_' in name or name.islower()  # Snake_case convention

    # Test that all flags can be combined without conflicts
    all_combined = 0
    for bit_val in PSF_FLAGS.bit_values:
        all_combined |= bit_val

    decoded_all = decode_psf_flags(all_combined)
    assert len(decoded_all) == 9
    assert set(decoded_all) == set(PSF_FLAGS.names)


def test_psf_classes_docstrings():
    """
    Test that the PSF classes have dynamic flag documentation.
    """
    classes_to_test = [PSFPhotometry, IterativePSFPhotometry]

    for cls in classes_to_test:
        docstring = cls.__call__.__doc__

        # Should have flags section
        assert '* ``flags`` : bitwise flag values' in docstring

        # Should have all dynamic flag descriptions
        dynamic_flags = [
            'npixfit smaller than full fit_shape region',
            'fitted position outside input image bounds',
            'non-positive flux',
            'possible non-convergence',
            'missing parameter covariance',
            'fitted parameter near a bound',
            'no overlap with data',
            'fully masked source',
            'too few pixels for fitting',
        ]

        for flag_desc in dynamic_flags:
            msg = f"Missing flag description in {cls.__name__}: {flag_desc}"
            assert flag_desc in docstring, msg


def test_decode_psf_flags_docstring():
    """
    Test that the decode_psf_flags function has dynamic flag
    documentation.
    """
    docstring = decode_psf_flags.__doc__

    # Should not have placeholder
    assert '<flag descriptions>' not in docstring

    # Should have all expected flag names in the expected format
    expected_flags = [
        "``'npixfit_partial'`` : bit 1",
        "``'outside_bounds'`` : bit 2",
        "``'negative_flux'`` : bit 4",
        "``'no_convergence'`` : bit 8",
        "``'no_covariance'`` : bit 16",
        "``'near_bound'`` : bit 32",
        "``'no_overlap'`` : bit 64",
        "``'fully_masked'`` : bit 128",
        "``'too_few_pixels'`` : bit 256",
    ]

    for flag_desc in expected_flags:
        msg = f"Missing flag in docstring: {flag_desc}"
        assert flag_desc in docstring, msg

    # Should have flag descriptions
    expected_descriptions = [
        'npixfit smaller than full fit_shape region',
        'fitted position outside input image bounds',
        'non-positive flux',
        'possible non-convergence',
        'missing parameter covariance',
        'fitted parameter near a bound',
        'no overlap with data',
        'fully masked source',
        'too few pixels for fitting',
    ]

    for desc in expected_descriptions:
        assert desc in docstring, f"Missing description: {desc}"


def test_update_decode_docstring_noop():
    """
    Test that the update_decode_docstring decorator is a no-op if no
    docstring exists.
    """
    @_update_decode_docstring
    def test_func(data):
        pass

    docstring = test_func.__doc__
    assert docstring is None
