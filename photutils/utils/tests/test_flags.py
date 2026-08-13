# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _flags module.
"""

from typing import ClassVar

import numpy as np
import pytest
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.utils._flags import (FlagDefinition, FlagRegistry, decode_flags,
                                    define_flag_docstring,
                                    update_flag_docstring)


class _ExampleFlags(FlagRegistry):
    """
    A minimal flag registry for testing the shared machinery.
    """

    FLAG_DEFINITIONS: ClassVar = [
        FlagDefinition(1, 'one', 'first flag', 'The first flag.'),
        FlagDefinition(2, 'two', 'second flag', 'The second flag.'),
        FlagDefinition(4, 'four', 'third flag', 'The third flag.'),
    ]
    domain: ClassVar = 'example'
    _DEPRECATED_FLAG_NAMES: ClassVar = {'old_one': 'one'}
    _DEPRECATED_CONSTANT_NAMES: ClassVar = {'OLD_ONE': 'ONE'}
    _DEPRECATED_SINCE: ClassVar = '3.0'
    _DEPRECATED_UNTIL: ClassVar = '4.0'


@pytest.fixture
def registry():
    """
    Provide an example flag registry instance.
    """
    return _ExampleFlags()


class TestFlagRegistry:
    """
    Tests for the FlagRegistry base class.
    """

    def test_constants(self, registry):
        """
        Test that uppercase constants are created for each flag.
        """
        assert registry.ONE == 1
        assert registry.TWO == 2
        assert registry.FOUR == 4

    def test_all_flags(self, registry):
        """
        Test that all_flags returns a copy of the flag definitions.
        """
        flags = registry.all_flags
        assert flags == _ExampleFlags.FLAG_DEFINITIONS
        flags.append(None)
        assert len(registry.all_flags) == 3

    def test_bit_values(self, registry):
        """
        Test the bit_values property.
        """
        assert registry.bit_values == [1, 2, 4]

    def test_names(self, registry):
        """
        Test the names property.
        """
        assert registry.names == ['one', 'two', 'four']

    def test_flag_dict(self, registry):
        """
        Test the flag_dict property.
        """
        assert registry.flag_dict == {1: 'one', 2: 'two', 4: 'four'}

    def test_get_definition_by_bit_value(self, registry):
        """
        Test get_definition with int and numpy integer bit values.
        """
        assert registry.get_definition(2).name == 'two'
        assert registry.get_definition(np.int64(4)).name == 'four'

    def test_get_definition_by_name(self, registry):
        """
        Test get_definition with a flag name.
        """
        definition = registry.get_definition('one')
        assert definition.bit_value == 1
        assert definition.description == 'first flag'

    def test_get_definition_deprecated_name(self, registry):
        """
        Test that a deprecated flag name warns and resolves to the new
        name.
        """
        match = "'old_one' is deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match) as record:
            definition = registry.get_definition('old_one')
        assert definition.name == 'one'
        msg = str(record[0].message)
        assert 'version 3.0' in msg
        assert 'version 4.0' in msg

    def test_get_definition_unknown_bit_value(self, registry):
        """
        Test that an unknown bit value raises KeyError.
        """
        match = 'No flag with bit value 8'
        with pytest.raises(KeyError, match=match):
            registry.get_definition(8)

    def test_get_definition_unknown_name(self, registry):
        """
        Test that an unknown flag name raises KeyError.
        """
        match = "No flag with name 'unknown'"
        with pytest.raises(KeyError, match=match):
            registry.get_definition('unknown')

    @pytest.mark.parametrize('identifier', [1.5, True, None, [1]])
    def test_get_definition_invalid_type(self, registry, identifier):
        """
        Test that non-int, non-str identifiers raise TypeError.
        """
        match = 'identifier must be int'
        with pytest.raises(TypeError, match=match):
            registry.get_definition(identifier)

    def test_get_name(self, registry):
        """
        Test get_name.
        """
        assert registry.get_name(2) == 'two'

    def test_get_bit_value(self, registry):
        """
        Test get_bit_value.
        """
        assert registry.get_bit_value('two') == 2

    def test_get_description(self, registry):
        """
        Test get_description.
        """
        assert registry.get_description(4) == 'third flag'

    def test_get_detailed_description(self, registry):
        """
        Test get_detailed_description.
        """
        assert registry.get_detailed_description(4) == 'The third flag.'

    def test_deprecated_constant(self, registry):
        """
        Test that a deprecated constant name warns and resolves to the
        new constant.
        """
        match = "'OLD_ONE' attribute was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            assert registry.OLD_ONE == 1

    def test_unknown_attribute(self, registry):
        """
        Test that an unknown attribute raises AttributeError.
        """
        match = "no attribute 'UNKNOWN'"
        with pytest.raises(AttributeError, match=match):
            _ = registry.UNKNOWN


class TestDefineFlagDocstring:
    """
    Tests for the define_flag_docstring function.
    """

    def test_bullet_list(self, registry):
        """
        Test the generated bullet list content.
        """
        lines = define_flag_docstring(registry)
        assert lines[0] == ''
        assert lines[1] == '* **0** : No flags set.'
        assert lines[2] == "* **1** (``'one'``) : The first flag."
        assert len(lines) == 5

    def test_indent(self, registry):
        """
        Test the indent keyword.
        """
        lines = define_flag_docstring(registry, indent=4)
        assert lines[1] == '    * **0** : No flags set.'

    def test_invalid_registry(self):
        """
        Test that a non-FlagRegistry input raises TypeError.
        """
        match = 'registry must be an instance of FlagRegistry'
        with pytest.raises(TypeError, match=match):
            define_flag_docstring('not_a_registry')


class TestUpdateFlagDocstring:
    """
    Tests for the update_flag_docstring function.
    """

    def test_placeholder_replaced(self, registry):
        """
        Test that the placeholder is replaced with the bullet list.
        """
        def func():
            """
            Flags.

            <flag_descriptions>
            """

        func = update_flag_docstring(func, registry)
        assert '<flag_descriptions>' not in func.__doc__
        assert "* **1** (``'one'``) : The first flag." in func.__doc__

    def test_no_placeholder(self, registry):
        """
        Test that a docstring without the placeholder is unchanged.
        """
        def func():
            """
            Flags.
            """

        docstring = func.__doc__
        func = update_flag_docstring(func, registry)
        assert func.__doc__ == docstring

    def test_no_docstring(self, registry):
        """
        Test that a function without a docstring is returned unchanged.
        """
        def func():
            pass

        func = update_flag_docstring(func, registry)
        assert func.__doc__ is None


class TestDecodeFlags:
    """
    Tests for the decode_flags function.
    """

    def test_scalar_zero(self, registry):
        """
        Test that a zero flag value decodes to an empty list.
        """
        assert decode_flags(0, registry) == []

    def test_scalar(self, registry):
        """
        Test decoding scalar flag values.
        """
        assert decode_flags(1, registry) == ['one']
        assert decode_flags(5, registry) == ['one', 'four']
        assert decode_flags(7, registry) == ['one', 'two', 'four']

    def test_return_bit_values(self, registry):
        """
        Test the return_bit_values keyword.
        """
        assert decode_flags(5, registry, return_bit_values=True) == [1, 4]

    def test_0d_array(self, registry):
        """
        Test decoding a 0-d array flag value.
        """
        assert decode_flags(np.array(3), registry) == ['one', 'two']

    def test_1d_array(self, registry):
        """
        Test decoding a 1D array of flag values.
        """
        decoded = decode_flags(np.array([0, 1, 6]), registry)
        assert decoded == [[], ['one'], ['two', 'four']]

    def test_2d_array(self, registry):
        """
        Test that decoding a 2D array preserves the input shape as
        nested lists.
        """
        flags = np.array([[0, 1], [2, 5]])
        decoded = decode_flags(flags, registry)
        assert decoded == [[[], ['one']], [['two'], ['one', 'four']]]

    def test_negative_flag(self, registry):
        """
        Test that a negative flag value raises ValueError.
        """
        match = 'Flag value must be a non-negative integer'
        with pytest.raises(ValueError, match=match):
            decode_flags(-1, registry)

    @pytest.mark.parametrize('flag_value', [1.5, True, np.True_])
    def test_invalid_type(self, registry, flag_value):
        """
        Test that non-integer flag values (including bools) raise
        TypeError.
        """
        match = 'Flag value must be an integer'
        with pytest.raises(TypeError, match=match):
            decode_flags(flag_value, registry)
