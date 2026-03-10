# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _deprecation module.
"""

import numpy as np
import pytest
from astropy.table import Table, TableMergeError, join, unique
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.utils._deprecation import (DeprecatedColumnQTable,
                                          DeprecatedColumnTable,
                                          create_deprecated_table_from_data,
                                          create_empty_deprecated_qtable)

DEPRECATION_MAP = {'old': 'new', 'old_b': 'new_b'}


@pytest.fixture
def raw_data():
    """
    Provide a raw data dictionary for table creation.
    """
    return {'old': [3, 2, 1], 'old_b': [4, 5, 6], 'stable': [7, 8, 9]}


def test_creation_and_type(raw_data):
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


def test_masked_creation(raw_data):
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


def test_getitem_access(raw_data):
    """
    Test deprecated access via __getitem__.
    """
    table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)

    match = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match):
        col = table['old']
    assert np.all(col == table['new'])

    match = "'old_b' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match):
        sub_table = table[['stable', 'old_b']]
    assert sub_table.colnames == ['stable', 'new_b']


def test_setitem_assignment(raw_data):
    """
    Test deprecated assignment via __setitem__.
    """
    table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
    match = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match):
        table['old'] = [100, 200, 300]
    assert np.all(table['new'] == [100, 200, 300])


def test_delitem_and_remove(raw_data):
    """
    Test deprecated deletion via __delitem__ and remove methods.
    """
    table1 = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
    match = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match):
        del table1['old']
    assert 'new' not in table1.colnames

    table2 = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
    with pytest.warns(AstropyDeprecationWarning, match=match):
        table2.remove_column('old')
    assert 'new' not in table2.colnames


def test_keep_columns(raw_data):
    """
    Test deprecated use in keep_columns.
    """
    table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
    match = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match):
        table.keep_columns(['stable', 'old'])
    assert set(table.colnames) == {'stable', 'new'}


def test_rename_methods(raw_data):
    """
    Test deprecated use in rename_column and rename_columns.
    """
    table1 = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
    match = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match):
        table1.rename_column('old', 'final_name_1')
    assert 'final_name_1' in table1.colnames
    assert 'new' not in table1.colnames

    table2 = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
    with pytest.warns(AstropyDeprecationWarning):
        table2.rename_columns(['old', 'old_b'], ['final1', 'final2'])
    assert set(table2.colnames) == {'final1', 'final2', 'stable'}


def test_data_operations(raw_data):
    """
    Test deprecated use in sort, group_by, and unique.
    """
    table_sort = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
    match = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match):
        table_sort.sort('old')
    assert table_sort['new'][0] == 1

    table_group = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
    match = "'old_b' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match):
        groups = table_group.group_by('old_b')
    assert len(groups.groups) == 3

    table_unique = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
    match = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match):
        unique_table = unique(table_unique, keys='old')
    assert len(unique_table) == 3


def test_join(raw_data, recwarn):
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


def test_indexing(raw_data, recwarn):
    """
    Test deprecated use in add_index and remove_indices.
    """
    table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
    match = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match):
        table.add_index('old')

    assert len(table.indices) == 1
    assert table.indices[0].columns[0].name == 'new'

    # The `pytest.warns` context manager consumes the warning. Now we can
    # test that the next operation issues no new warnings.
    table.remove_indices('new')
    assert not table.indices
    assert len(recwarn) == 0


def test_non_string_access_no_warning(raw_data, recwarn):
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


def test_contains(raw_data):
    """
    Test deprecated use in ``in`` checks.
    """
    table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
    match = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match):
        assert 'old' in table

    # Non-deprecated column: no warning
    assert 'stable' in table

    # Missing column: no warning
    assert 'missing' not in table


def test_copy_preserves_deprecation(raw_data):
    """
    Test that copy() preserves deprecation behavior.
    """
    table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
    copied = table.copy()

    assert isinstance(copied, DeprecatedColumnTable)
    match = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match):
        col = copied['old']
    assert np.all(col == [3, 2, 1])

    # Ensure it's a true copy; modifying original doesn't affect copy
    table['new'][0] = 999
    assert copied['new'][0] == 3


def test_remove_columns(raw_data):
    """
    Test deprecated use in remove_columns (plural).
    """
    table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
    with pytest.warns(AstropyDeprecationWarning):
        table.remove_columns(['old', 'old_b'])
    assert table.colnames == ['stable']


def test_replace_column(raw_data):
    """
    Test deprecated use in replace_column.
    """
    table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
    match = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match):
        table.replace_column('old', [100, 200, 300])
    assert np.all(table['new'] == [100, 200, 300])


def test_empty_deprecation_map(raw_data, recwarn):
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


def test_add_index_list(raw_data):
    """
    Test deprecated use in add_index with a list of column names.
    """
    table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
    with pytest.warns(AstropyDeprecationWarning):
        table.add_index(['old', 'old_b'])
    assert len(table.indices) == 1
    index_col_names = [c.name for c in table.indices[0].columns]
    assert index_col_names == ['new', 'new_b']


def test_create_empty_deprecated_qtable():
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
    match = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match):
        col = table['old']
    assert np.all(col == [1, 2, 3])


def test_slice_preserves_deprecation(raw_data):
    """
    Test that slicing preserves deprecation behavior.
    """
    table = create_deprecated_table_from_data(raw_data, DEPRECATION_MAP)
    sliced = table[0:2]

    assert isinstance(sliced, DeprecatedColumnTable)
    match = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match):
        col = sliced['old']
    assert np.all(col == [3, 2])


def test_translate_names_non_string(raw_data, recwarn):
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
