# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _deprecation module.
"""

import warnings

import numpy as np
import pytest
from astropy.table import QTable, Table, TableMergeError, join, unique
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.utils._deprecation import (DeprecatedColumnQTable,
                                          DeprecatedColumnTable,
                                          _future_column_names_var,
                                          create_deprecated_table_from_data,
                                          create_empty_deprecated_qtable,
                                          deprecated, deprecated_getattr,
                                          deprecated_positional_kwargs,
                                          deprecated_renamed_argument,
                                          use_future_column_names)

DEPRECATION_MAP = {'old': 'new', 'old_b': 'new_b'}


@pytest.fixture
def raw_data():
    """
    Provide a raw data dictionary for table creation.
    """
    return {'old': [3, 2, 1], 'old_b': [4, 5, 6], 'stable': [7, 8, 9]}


class TestDeprecatedColumn:
    """
    Tests for DeprecatedColumnTable and DeprecatedColumnQTable.
    """

    def test_creation_and_type(self, raw_data):
        """
        Test that the factory creates the correct object type.
        """
        table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
        assert isinstance(table, DeprecatedColumnTable)
        assert not isinstance(table, DeprecatedColumnQTable)
        assert set(table.colnames) == {'new', 'new_b', 'stable'}

        qtable = create_deprecated_table_from_data(
            raw_data, DEPRECATION_MAP, use_qtable=True)
        assert isinstance(qtable, DeprecatedColumnQTable)

    def test_masked_creation(self, raw_data):
        """
        Test that kwargs like "masked" are passed through correctly.
        """
        table = create_deprecated_table_from_data(
            raw_data, DEPRECATION_MAP, masked=True,
        )
        assert isinstance(table, DeprecatedColumnTable)
        assert table.masked is True
        table['new'].mask[0] = True
        assert np.all(table['new'].mask == [True, False, False])

    def test_getitem_access(self, raw_data):
        """
        Test deprecated access via __getitem__.
        """
        table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)

        match = "'old' was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            col = table['old']
        assert np.all(col == table['new'])

        match = "'old_b' was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            sub_table = table[['stable', 'old_b']]
        assert sub_table.colnames == ['stable', 'new_b']

    def test_setitem_assignment(self, raw_data):
        """
        Test deprecated assignment via __setitem__.
        """
        table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
        match = "'old' was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            table['old'] = [100, 200, 300]
        assert np.all(table['new'] == [100, 200, 300])

    def test_delitem_and_remove(self, raw_data):
        """
        Test deprecated deletion via __delitem__ and remove methods.
        """
        table1 = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
        match = "'old' was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            del table1['old']
        assert 'new' not in table1.colnames

        table2 = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
        with pytest.warns(AstropyDeprecationWarning, match=match):
            table2.remove_column('old')
        assert 'new' not in table2.colnames

    def test_keep_columns(self, raw_data):
        """
        Test deprecated use in keep_columns.
        """
        table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
        match = "'old' was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            table.keep_columns(['stable', 'old'])
        assert set(table.colnames) == {'stable', 'new'}

    def test_rename_methods(self, raw_data):
        """
        Test deprecated use in rename_column and rename_columns.
        """
        table1 = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
        match = "'old' was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            table1.rename_column('old', 'final_name_1')
        assert 'final_name_1' in table1.colnames
        assert 'new' not in table1.colnames

        table2 = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
        with pytest.warns(AstropyDeprecationWarning):
            table2.rename_columns(['old', 'old_b'], ['final1', 'final2'])
        assert set(table2.colnames) == {'final1', 'final2', 'stable'}

    def test_data_operations(self, raw_data):
        """
        Test deprecated use in sort, group_by, and unique.
        """
        table_sort = create_deprecated_table_from_data(raw_data,
                                                       DEPRECATION_MAP)
        match = "'old' was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            table_sort.sort('old')
        assert table_sort['new'][0] == 1

        table_group = create_deprecated_table_from_data(raw_data,
                                                        DEPRECATION_MAP)
        match = "'old_b' was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            groups = table_group.group_by('old_b')
        assert len(groups.groups) == 3

        table_unique = create_deprecated_table_from_data(raw_data,
                                                         DEPRECATION_MAP)
        match = "'old' was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            unique_table = unique(table_unique, keys='old')
        assert len(unique_table) == 3

    def test_join(self, raw_data, recwarn):
        """
        Test deprecated use in the standalone join function.
        """
        table1 = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
        table2 = Table({'new': [1, 3], 'extra': [1.1, 3.3]})

        match = "Left table does not have key column 'old'"
        with pytest.raises(TableMergeError, match=match):
            join(table1, table2, keys='old')

        # Test that it works correctly with the new name and issues no
        # warnings
        joined = join(table1, table2, keys='new')
        assert 'extra' in joined.colnames
        assert len(joined) == 2
        assert len(recwarn) == 0

    def test_indexing(self, raw_data, recwarn):
        """
        Test deprecated use in add_index and remove_indices.
        """
        table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
        match = "'old' was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            table.add_index('old')

        assert len(table.indices) == 1
        assert table.indices[0].columns[0].name == 'new'

        # The `pytest.warns` context manager consumes the warning. Now we can
        # test that the next operation issues no new warnings.
        table.remove_indices('new')
        assert not table.indices
        assert len(recwarn) == 0

    def test_non_string_access_no_warning(self, raw_data, recwarn):
        """
        Test that non-string access does not trigger warnings.

        This ensures that row access via integers or slices does not
        incorrectly engage the name translation logic.
        """
        table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)

        # Access a row by integer
        row = table[0]
        assert row['new'] == 3

        # Slice rows
        sliced = table[0:2]
        assert len(sliced) == 2

        # Assert that no warnings were recorded during these operations
        assert len(recwarn) == 0

    def test_contains(self, raw_data):
        """
        Test deprecated use in ``in`` checks.
        """
        table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
        match = "'old' was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            assert 'old' in table

        # Non-deprecated column: no warning
        assert 'stable' in table

        # Missing column: no warning
        assert 'missing' not in table

    def test_copy_preserves_deprecation(self, raw_data):
        """
        Test that copy() preserves deprecation behavior.
        """
        table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
        copied = table.copy()

        assert isinstance(copied, DeprecatedColumnTable)
        match = "'old' was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            col = copied['old']
        assert np.all(col == [3, 2, 1])

        # Ensure it's a true copy; modifying original doesn't affect copy
        table['new'][0] = 999
        assert copied['new'][0] == 3

    def test_remove_columns(self, raw_data):
        """
        Test deprecated use in remove_columns (plural).
        """
        table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
        with pytest.warns(AstropyDeprecationWarning):
            table.remove_columns(['old', 'old_b'])
        assert table.colnames == ['stable']

    def test_replace_column(self, raw_data):
        """
        Test deprecated use in replace_column.
        """
        table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
        match = "'old' was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            table.replace_column('old', [100, 200, 300])
        assert np.all(table['new'] == [100, 200, 300])

    def test_empty_deprecation_map(self, raw_data, recwarn):
        """
        Test a table with an empty deprecation map (no deprecated names).
        """
        table = create_deprecated_table_from_data(raw_data, {})  # empty map
        # All operations should work without any warnings
        assert set(table.colnames) == {'old', 'old_b', 'stable'}
        col = table['old']
        assert np.all(col == [3, 2, 1])
        table.sort('old')
        assert 'old' in table
        assert len(recwarn) == 0

    def test_add_index_list(self, raw_data):
        """
        Test deprecated use in add_index with a list of column names.
        """
        table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
        with pytest.warns(AstropyDeprecationWarning):
            table.add_index(['old', 'old_b'])
        assert len(table.indices) == 1
        index_col_names = [c.name for c in table.indices[0].columns]
        assert index_col_names == ['new', 'new_b']

    def test_remove_indices_deprecated(self, raw_data):
        """
        Test deprecated use in remove_indices.
        """
        table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
        table.add_index('new')
        assert len(table.indices) == 1
        with pytest.warns(AstropyDeprecationWarning):
            table.remove_indices('old')
        assert len(table.indices) == 0

    def test_copy_with_none_deprecation_map(self):
        """
        Test that copy() works when deprecation_map is None.
        """
        table = DeprecatedColumnTable({'a': [1, 2, 3]})
        table.deprecation_map = None
        copied = table.copy()
        assert copied.deprecation_map is None
        assert np.all(copied['a'] == [1, 2, 3])

    def test_create_empty_deprecated_qtable(self):
        """
        Test creating an empty QTable and adding columns incrementally.
        """
        table = create_empty_deprecated_qtable(DEPRECATION_MAP)
        assert isinstance(table, DeprecatedColumnQTable)
        assert len(table) == 0

        # Add columns using new names
        table['new'] = [1, 2, 3]
        table['stable'] = [4, 5, 6]

        # Access via deprecated name
        match = "'old' was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            col = table['old']
        assert np.all(col == [1, 2, 3])

    def test_slice_preserves_deprecation(self, raw_data):
        """
        Test that slicing preserves deprecation behavior.
        """
        table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
        sliced = table[0:2]

        assert isinstance(sliced, DeprecatedColumnTable)
        match = "'old' was deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            col = sliced['old']
        assert np.all(col == [3, 2])

    def test_translate_names_non_string(self, raw_data, recwarn):
        """
        Test that _translate_names passes through non-string/non-sequence
        values unchanged (e.g., a Table used as group_by keys).
        """
        table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)

        # group_by accepts a Table as keys; _translate_names should pass
        # it through without modification or warnings
        key_table = Table({'new': [3, 2, 1]})
        groups = table.group_by(key_table)
        assert len(groups.groups) == 3
        assert len(recwarn) == 0


class TestFutureColumnNames:
    """
    Tests for the ``photutils.future_column_names`` opt-in flag.
    """

    def setup_method(self):
        import photutils
        self._original = photutils.future_column_names
        photutils.future_column_names = True

    def teardown_method(self):
        import photutils
        photutils.future_column_names = self._original

    def test_from_data_returns_plain_table(self, raw_data):
        """
        Test that create_deprecated_table_from_data returns a plain
        Table when the flag is set.
        """
        table = create_deprecated_table_from_data(
            raw_data, DEPRECATION_MAP)
        assert type(table) is Table
        assert set(table.colnames) == {'new', 'new_b', 'stable'}

    def test_from_data_returns_plain_qtable(self, raw_data):
        """
        Test that create_deprecated_table_from_data returns a plain
        QTable when the flag is set and use_qtable=True.
        """
        table = create_deprecated_table_from_data(
            raw_data, DEPRECATION_MAP, use_qtable=True)
        assert type(table) is QTable

    def test_from_data_no_warnings(self, raw_data, recwarn):
        """
        Test that accessing columns on a plain table created with the
        flag does not issue deprecation warnings.
        """
        table = create_deprecated_table_from_data(
            raw_data, DEPRECATION_MAP)
        _ = table['new']
        assert len(recwarn) == 0

    def test_from_data_old_name_raises(self, raw_data):
        """
        Test that accessing a deprecated name on a plain table created
        with the flag raises KeyError.
        """
        table = create_deprecated_table_from_data(
            raw_data, DEPRECATION_MAP)
        with pytest.raises(KeyError):
            table['old']

    def test_empty_returns_plain_qtable(self):
        """
        Test that create_empty_deprecated_qtable returns a plain QTable
        when the flag is set.
        """
        table = create_empty_deprecated_qtable(DEPRECATION_MAP)
        assert type(table) is QTable
        assert len(table) == 0


class TestUseFutureColumnNames:
    """
    Tests for the ``use_future_column_names`` context manager.
    """

    def test_context_manager_returns_plain_table(self, raw_data):
        """
        Test that the context manager makes
        create_deprecated_table_from_data return a plain Table.
        """
        with use_future_column_names():
            table = create_deprecated_table_from_data(
                raw_data, DEPRECATION_MAP)
        assert type(table) is Table
        assert set(table.colnames) == {'new', 'new_b', 'stable'}

    def test_context_manager_returns_plain_qtable(self, raw_data):
        """
        Test that the context manager makes
        create_deprecated_table_from_data return a plain QTable.
        """
        with use_future_column_names():
            table = create_deprecated_table_from_data(
                raw_data, DEPRECATION_MAP, use_qtable=True)
        assert type(table) is QTable

    def test_context_manager_empty_qtable(self):
        """
        Test that the context manager makes
        create_empty_deprecated_qtable return a plain QTable.
        """
        with use_future_column_names():
            table = create_empty_deprecated_qtable(DEPRECATION_MAP)
        assert type(table) is QTable
        assert len(table) == 0

    def test_global_unchanged_after_context(self):
        """
        Test that the global flag is unchanged after the context
        manager exits.
        """
        import photutils

        original = photutils.future_column_names
        with use_future_column_names():
            pass
        assert photutils.future_column_names == original

    def test_outside_context_uses_global(self, raw_data):
        """
        Test that outside the context manager, the global flag is
        respected and deprecated table behavior is used.
        """
        import photutils

        assert not photutils.future_column_names
        with use_future_column_names():
            pass
        # Outside the context, should use the deprecated table
        table = create_deprecated_table_from_data(
            raw_data, DEPRECATION_MAP)
        assert type(table) is DeprecatedColumnTable

    def test_nested_context_managers(self, raw_data):
        """
        Test that nested context managers work correctly with different
        values.
        """
        with use_future_column_names(enabled=True):
            table1 = create_deprecated_table_from_data(
                raw_data, DEPRECATION_MAP)
            assert type(table1) is Table

            with use_future_column_names(enabled=False):
                table2 = create_deprecated_table_from_data(
                    raw_data, DEPRECATION_MAP)
                assert type(table2) is DeprecatedColumnTable

            # Back to the outer context
            table3 = create_deprecated_table_from_data(
                raw_data, DEPRECATION_MAP)
            assert type(table3) is Table

    def test_context_manager_disabled(self, raw_data):
        """
        Test that use_future_column_names(enabled=False) forces the
        deprecated table behavior even if the global flag is True.
        """
        import photutils

        original = photutils.future_column_names
        try:
            photutils.future_column_names = True
            with use_future_column_names(enabled=False):
                table = create_deprecated_table_from_data(
                    raw_data, DEPRECATION_MAP)
                assert type(table) is DeprecatedColumnTable
        finally:
            photutils.future_column_names = original

    def test_restores_on_exception(self):
        """
        Test that the context manager restores state even if an
        exception occurs.
        """
        sentinel_before = _future_column_names_var.get()
        msg = 'test error'
        with use_future_column_names(), pytest.raises(ValueError, match=msg):
            raise ValueError(msg)
        assert _future_column_names_var.get() == sentinel_before


@deprecated_positional_kwargs('1.0', until='2.0')
def _example_func(a, b=10, c=20):
    """
    Example function for testing deprecated_positional_kwargs.
    """
    return a + b + c


class TestDeprecatedPositionalKwargs:
    """
    Tests for the deprecated_positional_kwargs decorator.
    """

    def test_no_warning_at_limit(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = _example_func(1)
        assert result == 31

    def test_no_warning_keyword_only(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = _example_func(1, b=5, c=3)
        assert result == 9

    def test_warns_when_exceeded(self):
        match = "'_example_func'"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            result = _example_func(1, 2)
        assert result == 23

    def test_warning_message_versions(self):
        with pytest.warns(AstropyDeprecationWarning) as record:
            _example_func(1, 2, 3)
        msg = str(record[0].message)
        assert '1.0' in msg
        assert '2.0' in msg

    def test_warning_names_single(self):
        with pytest.warns(AstropyDeprecationWarning) as record:
            _example_func(1, 2)
        msg = str(record[0].message)
        assert "Passing 'b' positionally" in msg
        assert "'c'" not in msg
        assert 'Pass it as a keyword argument' in msg
        assert 'b=...' in msg

    def test_warning_names_two(self):
        with pytest.warns(AstropyDeprecationWarning) as record:
            _example_func(1, 2, 3)
        msg = str(record[0].message)
        assert "'b' and 'c'" in msg
        assert 'Pass them as keyword arguments' in msg
        assert 'b=..., c=...' in msg

    def test_warning_names_three(self):
        @deprecated_positional_kwargs('1.0')
        def _func(a, b=1, c=2, d=3):
            return a + b + c + d

        with pytest.warns(AstropyDeprecationWarning) as record:
            _func(1, 2, 3, 4)
        msg = str(record[0].message)
        assert "'b', 'c', and 'd'" in msg
        assert 'Pass them as keyword arguments' in msg
        assert 'b=..., c=..., d=...' in msg

    def test_return_value_preserved(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert _example_func(5, 3, 2) == 10
        assert _example_func(5) == 35

    def test_preserves_metadata(self):
        assert _example_func.__name__ == '_example_func'
        assert 'Example function' in _example_func.__doc__

    def test_zero_positional(self):
        @deprecated_positional_kwargs('1.5', until='2.5')
        def _no_pos(x=0):
            return x

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = _no_pos(x=42)
        assert result == 42

        with pytest.warns(AstropyDeprecationWarning) as record:
            result = _no_pos(42)
        assert result == 42
        msg = str(record[0].message)
        assert "'x'" in msg
        assert 'Pass it as a keyword argument' in msg
        assert 'x=...' in msg

    def test_no_until(self):
        @deprecated_positional_kwargs('3.0')
        def _func(a, b=10):
            return a + b

        with pytest.warns(AstropyDeprecationWarning) as record:
            result = _func(1, 2)
        assert result == 3
        msg = str(record[0].message)
        assert '3.0' in msg
        assert 'a future version' in msg
        assert "'b'" in msg
        assert 'b=...' in msg

    def test_until_keyword_only(self):
        match = 'takes 1 positional argument'
        with pytest.raises(TypeError, match=match):
            # until passed positionally
            deprecated_positional_kwargs('1.0', '2.0')

    def test_since_until_int(self):
        @deprecated_positional_kwargs(3, until=4)
        def _func(a, b=10):
            return a + b

        with pytest.warns(AstropyDeprecationWarning) as record:
            result = _func(1, 2)
        assert result == 3
        msg = str(record[0].message)
        assert '3' in msg
        assert '4' in msg

    def test_multiple_required_args(self):
        @deprecated_positional_kwargs('1.0')
        def _func(a, b, c=10):
            return a + b + c

        # Two required positional args should not warn
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = _func(1, 2)
        assert result == 13

        # Third (optional) arg passed positionally should warn
        with pytest.warns(AstropyDeprecationWarning) as record:
            result = _func(1, 2, 3)
        assert result == 6
        msg = str(record[0].message)
        assert "'c'" in msg
        assert "Passing 'c'" in msg
        assert "'a'" not in msg
        assert "'b'" not in msg

    def test_positional_only_params(self):
        @deprecated_positional_kwargs('1.0')
        def _func(a, /, b=10):
            return a + b

        # Positional-only arg should not warn
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = _func(1)
        assert result == 11

        # Optional arg passed positionally should warn
        with pytest.warns(AstropyDeprecationWarning) as record:
            result = _func(1, 2)
        assert result == 3
        msg = str(record[0].message)
        assert "'b'" in msg


@deprecated_renamed_argument('b', 'new', '1.0', until='2.0')
def _example_func2(a, new, c=20):
    """
    Example function for testing deprecated_renamed_argument.
    """
    return a + new + c


@deprecated_renamed_argument('b', 'new', '1.0', until=None)
def _example_func3(a, new, c=20):
    """
    Example function for testing deprecated_renamed_argument
    with no "until" version specified.
    """
    return a + new + c


def test_deprecated_renamed_argument():
    # Test that using the new name works without warnings
    result = _example_func2(1, new=5, c=3)
    assert result == 9

    # Test that using the old name issues a warning and still works
    with pytest.warns(AstropyDeprecationWarning) as record:
        result = _example_func2(1, b=5, c=3)
    assert result == 9
    msg = str(record[0].message)
    assert "'b' was deprecated" in msg
    assert "'new'" in msg
    assert 'version 2.0' in msg

    # Test that if until=None, the warning is issued but no end version
    # is mentioned
    with pytest.warns(AstropyDeprecationWarning) as record:
        result = _example_func3(1, b=5, c=3)
    assert result == 9
    msg = str(record[0].message)
    assert 'deprecated' in msg.lower()
    assert 'future version' in msg


def test_deprecated_renamed_argument_always_warns():
    # Test that the warning is issued on every call, not just the first
    # time from a given call site.
    with pytest.warns(AstropyDeprecationWarning):
        _example_func2(1, b=5)
    with pytest.warns(AstropyDeprecationWarning):
        _example_func2(1, b=5)


class TestColumnDeprecationUntil:
    """
    Tests for the ``until`` parameter in column deprecation.
    """

    def test_until_in_warning_message(self):
        """
        Test that the removal version appears in the warning message.
        """
        table = create_deprecated_table_from_data(
            {'old': [1]}, DEPRECATION_MAP, until='5.0')
        with pytest.warns(AstropyDeprecationWarning) as record:
            _ = table['old']
        msg = str(record[0].message)
        assert 'version 5.0' in msg

    def test_until_none(self):
        """
        Test that without ``until``, the message says "a future
        version".
        """
        table = create_deprecated_table_from_data(
            {'old': [1]}, DEPRECATION_MAP)
        with pytest.warns(AstropyDeprecationWarning) as record:
            _ = table['old']
        msg = str(record[0].message)
        assert 'a future version' in msg

    def test_future_column_names(self):
        """
        Test that the warning mentions ``future_column_names``.
        """
        table = create_deprecated_table_from_data(
            {'old': [1]}, DEPRECATION_MAP)
        with pytest.warns(AstropyDeprecationWarning) as record:
            _ = table['old']
        msg = str(record[0].message)
        assert 'future_column_names' in msg

    def test_until_preserved_on_copy(self):
        """
        Test that copy() preserves the ``until`` value.
        """
        table = create_deprecated_table_from_data(
            {'old': [1]}, DEPRECATION_MAP, until='5.0')
        copied = table.copy()
        with pytest.warns(AstropyDeprecationWarning) as record:
            _ = copied['old']
        msg = str(record[0].message)
        assert 'version 5.0' in msg

    def test_until_preserved_on_slice(self):
        """
        Test that slicing preserves the ``until`` value.
        """
        table = create_deprecated_table_from_data(
            {'old': [1, 2]}, DEPRECATION_MAP, until='5.0')
        sliced = table[0:1]
        with pytest.warns(AstropyDeprecationWarning) as record:
            _ = sliced['old']
        msg = str(record[0].message)
        assert 'version 5.0' in msg

    def test_empty_qtable_until(self):
        """
        Test that create_empty_deprecated_qtable passes ``until``.
        """
        table = create_empty_deprecated_qtable(
            DEPRECATION_MAP, until='6.0')
        table['new'] = [1, 2]
        with pytest.warns(AstropyDeprecationWarning) as record:
            _ = table['old']
        msg = str(record[0].message)
        assert 'version 6.0' in msg

    def test_since_in_warning_message(self):
        """
        Test that the deprecation version appears in the warning
        message.
        """
        table = create_deprecated_table_from_data(
            {'old': [1]}, DEPRECATION_MAP, since='3.0')
        with pytest.warns(AstropyDeprecationWarning) as record:
            _ = table['old']
        msg = str(record[0].message)
        assert 'in version 3.0' in msg

    def test_since_none(self):
        """
        Test that without ``since``, the message does not mention a
        deprecation version.
        """
        table = create_deprecated_table_from_data(
            {'old': [1]}, DEPRECATION_MAP)
        with pytest.warns(AstropyDeprecationWarning) as record:
            _ = table['old']
        msg = str(record[0].message)
        assert 'was deprecated.' in msg
        assert 'was deprecated in version' not in msg

    def test_since_and_until(self):
        """
        Test that both ``since`` and ``until`` appear in the warning.
        """
        table = create_deprecated_table_from_data(
            {'old': [1]}, DEPRECATION_MAP, since='3.0', until='4.0')
        with pytest.warns(AstropyDeprecationWarning) as record:
            _ = table['old']
        msg = str(record[0].message)
        assert 'in version 3.0' in msg
        assert 'version 4.0' in msg

    def test_since_preserved_on_copy(self):
        """
        Test that copy() preserves the ``since`` value.
        """
        table = create_deprecated_table_from_data(
            {'old': [1]}, DEPRECATION_MAP, since='3.0', until='4.0')
        copied = table.copy()
        with pytest.warns(AstropyDeprecationWarning) as record:
            _ = copied['old']
        msg = str(record[0].message)
        assert 'in version 3.0' in msg
        assert 'version 4.0' in msg

    def test_since_preserved_on_slice(self):
        """
        Test that slicing preserves the ``since`` value.
        """
        table = create_deprecated_table_from_data(
            {'old': [1, 2]}, DEPRECATION_MAP, since='3.0', until='4.0')
        sliced = table[0:1]
        with pytest.warns(AstropyDeprecationWarning) as record:
            _ = sliced['old']
        msg = str(record[0].message)
        assert 'in version 3.0' in msg

    def test_empty_qtable_since(self):
        """
        Test that create_empty_deprecated_qtable passes ``since``.
        """
        table = create_empty_deprecated_qtable(
            DEPRECATION_MAP, since='3.0', until='4.0')
        table['new'] = [1, 2]
        with pytest.warns(AstropyDeprecationWarning) as record:
            _ = table['old']
        msg = str(record[0].message)
        assert 'in version 3.0' in msg
        assert 'version 4.0' in msg


class _ExampleObj:
    """
    A helper class for testing ``deprecated_getattr``.
    """

    def __init__(self):
        self.new_attr = 42
        self._deprecated_attrs = {'old_attr': 'new_attr'}

    def __getattr__(self, name):
        return deprecated_getattr(self, name, self._deprecated_attrs)


class _ExampleObjSinceUntil:
    """
    A helper class for testing ``deprecated_getattr`` with since/until.
    """

    def __init__(self):
        self.new_attr = 42
        self._deprecated_attrs = {'old_attr': 'new_attr'}

    def __getattr__(self, name):
        return deprecated_getattr(self, name, self._deprecated_attrs,
                                  since='3.0', until='4.0')


class TestDeprecatedGetattr:
    """
    Tests for the ``deprecated_getattr`` helper function.
    """

    def test_deprecated_access_warns(self):
        """
        Test that accessing a deprecated attribute issues a warning.
        """
        obj = _ExampleObj()
        match = "'old_attr'.*deprecated"
        with pytest.warns(AstropyDeprecationWarning, match=match):
            val = obj.old_attr
        assert val == 42

    def test_new_name_no_warning(self):
        """
        Test that the new attribute does not trigger a warning.
        """
        obj = _ExampleObj()
        assert obj.new_attr == 42

    def test_unknown_attr_raises(self):
        """
        Test that an unknown attribute raises AttributeError.
        """
        obj = _ExampleObj()
        match = 'no attribute'
        with pytest.raises(AttributeError, match=match):
            _ = obj.nonexistent

    def test_message_no_since_no_until(self):
        """
        Test the default message (no "since", no "until").
        """
        obj = _ExampleObj()
        with pytest.warns(AstropyDeprecationWarning) as record:
            _ = obj.old_attr
        msg = str(record[0].message)
        assert "'old_attr'" in msg
        assert "'new_attr'" in msg
        assert 'a future version' in msg
        assert 'in version' not in msg

    def test_message_with_since_and_until(self):
        """
        Test the message includes "since" and "until" versions.
        """
        obj = _ExampleObjSinceUntil()
        with pytest.warns(AstropyDeprecationWarning) as record:
            _ = obj.old_attr
        msg = str(record[0].message)
        assert 'in version 3.0' in msg
        assert 'version 4.0' in msg

    def test_message_with_since_only(self):
        """
        Test the message when only "since" is provided.
        """
        obj = _ExampleObj()
        dep_map = {'x': 'y'}
        obj.y = 99
        with pytest.warns(AstropyDeprecationWarning) as record:
            val = deprecated_getattr(obj, 'x', dep_map, since='2.0')
        assert val == 99
        msg = str(record[0].message)
        assert 'in version 2.0' in msg
        assert 'a future version' in msg

    def test_message_with_until_only(self):
        """
        Test the message when only "until" is provided.
        """
        obj = _ExampleObj()
        dep_map = {'x': 'y'}
        obj.y = 99
        with pytest.warns(AstropyDeprecationWarning) as record:
            val = deprecated_getattr(obj, 'x', dep_map, until='5.0')
        assert val == 99
        msg = str(record[0].message)
        assert 'version 5.0' in msg
        # since was not given, so "deprecated in version" should not appear
        assert 'deprecated in version' not in msg


@deprecated('1.0', until='2.0')
def _example_func4(a, b=10, c=20):
    """
    Example function for testing deprecated_positional_kwargs.
    """
    return a + b + c


@deprecated('1.0')
def _example_func5(a, b=10, c=20):
    """
    Example function for testing deprecated_positional_kwargs.
    """
    return a + b + c


def test_deprecated():
    """
    Test the basic functionality of the @deprecated decorator.
    """
    with pytest.warns(AstropyDeprecationWarning) as record:
        result = _example_func4(1, 2, 3)
    assert result == 6
    msg = str(record[0].message)
    assert 'version 1.0' in msg
    assert 'version 2.0' in msg

    with pytest.warns(AstropyDeprecationWarning) as record:
        result = _example_func5(1, 2, 3)
    assert result == 6
    msg = str(record[0].message)
    assert 'version 1.0' in msg
    assert 'a future version' in msg
