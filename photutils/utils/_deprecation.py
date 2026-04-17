# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A module to create Astropy Tables with deprecated column names.

It is designed to create new table objects from raw data, rather than
modifying existing tables.

The primary function, ``create_deprecated_table_from_data``, handles
the data renaming and constructs an instance of a custom ``Table`` or
``QTable`` subclass that correctly handles all deprecated name access.

Note that standalone Astropy functions like ``join`` inspect
``colnames`` directly and do not trigger the deprecation mapping. Users
must use the new column names when calling such functions.
"""

import inspect
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

from astropy.table import QTable, Table
from astropy.utils.decorators import deprecated as astropy_deprecated
from astropy.utils.decorators import (
    deprecated_renamed_argument as astropy_deprecated_renamed_argument)
from astropy.utils.exceptions import AstropyDeprecationWarning

_SENTINEL = object()
_future_column_names_var = ContextVar(
    'photutils_future_column_names', default=_SENTINEL,
)


def _get_future_column_names():
    """
    Return the effective value of ``future_column_names``.

    A context-local override (set via `use_future_column_names`) takes
    precedence over the global ``photutils.future_column_names`` flag.

    Returns
    -------
    result : bool
        Whether future column names are enabled.
    """
    import photutils

    val = _future_column_names_var.get()
    if val is not _SENTINEL:
        return val
    return photutils.future_column_names


@contextmanager
def use_future_column_names(enabled=True):
    """
    Context manager to temporarily override ``future_column_names``.

    Within the ``with`` block, photutils functions will behave as
    though ``photutils.future_column_names`` is set to enabled,
    without modifying the global flag. This is safe to use in
    multi-threaded and async code because the override is stored in a
    `~contextvars.ContextVar`.

    Parameters
    ----------
    enabled : bool, optional
        The value to use inside the block. The default is `True`.

    Examples
    --------
    >>> import photutils
    >>> from photutils import use_future_column_names
    >>> photutils.future_column_names  # global default
    False
    >>> with use_future_column_names():
    ...     # inside here, tables use new column names only
    ...     pass
    >>> photutils.future_column_names  # unchanged
    False
    """
    token = _future_column_names_var.set(enabled)
    try:
        yield
    finally:
        _future_column_names_var.reset(token)


def deprecated(since, *, alternative=None, until=None):
    """
    Decorator to mark a function or method as deprecated.

    This is a wrapper around `astropy.utils.decorators.deprecated` that
    allows for an optional ``until`` parameter to specify when the
    deprecated functionality will be removed. If ``until`` is provided,
    the warning message will include both the deprecation version and the
    removal version.

    Parameters
    ----------
    since : str or int
        The version in which the function or method was deprecated.

    alternative : str or None, optional
        An optional string describing an alternative function or method
        to use instead of the deprecated one. If `None`, no alternative
        is mentioned in the warning message.

    until : str or int, optional
        The version in which the deprecated functionality will be removed.
        If `None`, the removal version is not mentioned in the warning
        message.

    Returns
    -------
    decorator : function
        A decorator function that can be applied to any function or method
        to mark it as deprecated.
    """
    if until is None:
        message = (f'This function was deprecated in version {since} and will '
                   'be removed in a future version.')
    else:
        remove_version = 'version ' + str(until)
        message = (f'This function was deprecated in version {since} and will '
                   f'be removed in {remove_version}.')

    if alternative is not None:
        message += f' Use {alternative} instead.'

    return astropy_deprecated(since, message=message)


def deprecated_renamed_argument(old_name, new_name, since, *, until=None):
    """
    Decorator to warn when a renamed argument is used.

    This is a wrapper around
    `astropy.utils.decorators.deprecated_renamed_argument` that allows
    for an optional ``until`` parameter to specify when the old argument
    name will be removed. If ``until`` is provided, the warning message
    will include both the deprecation version and the removal version.

    Parameters
    ----------
    old_name : str
        The old (deprecated) argument name.

    new_name : str or None
        The new argument name that should be used instead, or `None` if
        the argument has been removed entirely.

    since : str or int
        The version in which the argument was renamed or removed.

    until : str or int, optional
        The version in which the old argument name will be removed. If
        `None`, the removal version is not mentioned in the warning
        message.

    Returns
    -------
    decorator : function
        A decorator function that can be applied to any function to warn
        about the use of a renamed argument.
    """
    if until is None:
        return astropy_deprecated_renamed_argument(
            old_name, new_name, since)

    remove_version = 'version ' + str(until)
    message = (f"'{old_name}' was deprecated in version {since} and will "
               f'be removed in {remove_version}.')
    if new_name is not None:
        message += f" Use argument '{new_name}' instead."
    return astropy_deprecated_renamed_argument(
        old_name, new_name, since, message=message)


def deprecated_getattr(instance, name, deprecated_map, *, since=None,
                       until=None):
    """
    Handle deprecated attribute access on an instance.

    This is a helper function for ``__getattr__`` methods on classes
    that have deprecated attribute names. It checks if ``name`` is in
    ``deprecated_map`` and, if so, issues a deprecation warning and
    returns the value of the new attribute. Otherwise, it raises an
    `AttributeError`.

    Parameters
    ----------
    instance : object
        The instance on which the attribute was accessed.

    name : str
        The attribute name that was accessed.

    deprecated_map : dict
        A dictionary mapping old (deprecated) attribute names to their
        new attribute names.

    since : str or int, optional
        The version in which the attribute was deprecated. If `None`,
        the deprecation version is not mentioned in the warning message.

    until : str or int, optional
        The version in which the old attribute name will be removed.
        If `None`, the removal version is not mentioned in the warning
        message.

    Returns
    -------
    value : object
        The value of the new attribute.

    Raises
    ------
    AttributeError
        If ``name`` is not in ``deprecated_map``.
    """
    if name in deprecated_map:
        new_name = deprecated_map[name]
        since_str = ''
        if since is not None:
            since_str = f' in version {since}'
        if until is not None:
            remove_str = 'version ' + str(until)
        else:
            remove_str = 'a future version'
        warn_msg = (f'The {name!r} attribute was deprecated{since_str}; '
                    f'use {new_name!r} instead. It will be removed in '
                    f'{remove_str}.')
        warnings.warn(warn_msg, AstropyDeprecationWarning, stacklevel=3)
        return getattr(instance, new_name)

    msg = f'{type(instance).__name__!r} object has no attribute {name!r}'
    raise AttributeError(msg)


def deprecated_positional_kwargs(since, *, until=None):
    """
    Decorator to warn when optional arguments are passed positionally.

    Parameters that have no default value (i.e., required parameters)
    are allowed positionally. Parameters with default values (i.e.,
    optional parameters) will trigger a deprecation warning if passed
    positionally.

    Parameters
    ----------
    since : str or int
        The version in which passing optional arguments positionally is
        deprecated.

    until : str or int, optional
        The version in which passing optional arguments positionally
        will be removed. If `None`, the removal version is not mentioned
        in the warning message.

    Returns
    -------
    decorator : function
        A decorator function that can be applied to any function to warn
        about positional arguments.
    """
    def decorator(func):  # numpydoc ignore=GL08
        since_str = str(since)
        until_str = str(until) if until is not None else None
        sig = inspect.signature(func)
        n_positional = 0
        param_names = []
        for name, param in sig.parameters.items():
            param_names.append(name)
            if (param.kind in (inspect.Parameter.POSITIONAL_ONLY,
                               inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    and param.default is inspect.Parameter.empty):
                n_positional += 1

        @wraps(func)
        def wrapper(*args, **kwargs):  # numpydoc ignore=GL08
            if len(args) > n_positional:
                extra_names = param_names[n_positional:len(args)]
                quoted = [f"'{name}'" for name in extra_names]
                if len(quoted) == 1:
                    params_str = quoted[0]
                    pronoun = 'it'
                    kwarg_noun = 'a keyword argument'
                elif len(quoted) == 2:
                    params_str = f'{quoted[0]} and {quoted[1]}'
                    pronoun = 'them'
                    kwarg_noun = 'keyword arguments'
                else:
                    params_str = (', '.join(quoted[:-1])
                                  + f', and {quoted[-1]}')
                    pronoun = 'them'
                    kwarg_noun = 'keyword arguments'
                examples_str = ', '.join(f'{name}=...' for name in extra_names)
                remove_str = 'a future version'
                if until_str is not None:
                    remove_str = f'version {until_str}'
                msg = (f'Passing {params_str} positionally to '
                       f"'{func.__name__}' is deprecated as of version "
                       f'{since_str} and will be removed in {remove_str}. '
                       f'Pass {pronoun} as {kwarg_noun} instead '
                       f'(e.g., {examples_str}).')
                warnings.warn(msg, AstropyDeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator


class DeprecatedColumnMixin:
    """
    A mixin to handle deprecated column names in Astropy tables.

    This mixin overrides common table methods to intercept calls
    using old column names. It translates them to new names, issues a
    deprecation warning, and then calls the original parent method via
    ``super()``. This works correctly because instances are created from
    this class directly, ensuring a valid method resolution order.
    """

    deprecation_map = None
    _deprecation_since = None
    _deprecation_until = None

    def _warn_deprecated(self, name, new_name, stacklevel=4):
        """
        Issue a deprecation warning for a column name.

        Parameters
        ----------
        name : str
            The deprecated column name.

        new_name : str
            The new column name.

        stacklevel : int, optional
            The stack level for the warning. The default is 4.
        """
        since_str = ''
        if self._deprecation_since is not None:
            since_str = f' in version {self._deprecation_since}'
        if self._deprecation_until is not None:
            remove_str = 'version ' + str(self._deprecation_until)
        else:
            remove_str = 'a future version'
        msg = (f"The column name '{name}' was deprecated{since_str}. Use "
               f"'{new_name}' instead. It will be removed in "
               f'{remove_str}. Once you have updated your code to use '
               f"'{new_name}', set photutils.future_column_names = True "
               'to opt into a standard QTable without the deprecated '
               'column name mapping.')
        warnings.warn(msg, AstropyDeprecationWarning,
                      stacklevel=stacklevel)

    def _translate_name(self, name, stacklevel=4):
        """
        Translate a single name, issue a warning, and return the new
        name.

        Parameters
        ----------
        name : str
            The column name to be translated.

        stacklevel : int, optional
            The stack level for the warning. The default is 4.

        Returns
        -------
        result : str
            The translated new column name, or the original name if it
            is not deprecated.
        """
        if self.deprecation_map and name in self.deprecation_map:
            new_name = self.deprecation_map[name]
            self._warn_deprecated(name, new_name, stacklevel=stacklevel)
            return new_name
        return name

    def _translate_names(self, names, stacklevel=4):
        """
        Translate a single name or a list/tuple of names.

        Parameters
        ----------
        names : str or list or tuple
            The column name(s) to be translated.

        stacklevel : int, optional
            The stack level for the warning. The default is 4.

        Returns
        -------
        str or list or tuple
            The translated new column name(s).
        """
        if isinstance(names, (list, tuple)):
            return [self._translate_name(name, stacklevel=stacklevel)
                    for name in names]

        if not isinstance(names, str):
            return names

        return self._translate_name(names, stacklevel=stacklevel)

    def __contains__(self, name):
        """
        Override for ``in`` checks.
        """
        if (isinstance(name, str) and self.deprecation_map
                and name in self.deprecation_map):
            new_name = self.deprecation_map[name]
            self._warn_deprecated(name, new_name, stacklevel=3)
            return new_name in self.colnames
        return name in self.colnames

    def __getitem__(self, item):
        """
        Override for item access.
        """
        if isinstance(item, (str, list, tuple)):
            item = self._translate_names(item)
        result = super().__getitem__(item)
        if isinstance(result, type(self)) and self.deprecation_map:
            result.deprecation_map = self.deprecation_map
            result._deprecation_since = self._deprecation_since
            result._deprecation_until = self._deprecation_until
        return result

    def __setitem__(self, item, value):
        """
        Override for item assignment.
        """
        if isinstance(item, str):
            item = self._translate_names(item)
        super().__setitem__(item, value)

    def __delitem__(self, item):
        """
        Override for item deletion.
        """
        if isinstance(item, str):
            item = self._translate_names(item)
        super().__delitem__(item)

    def keep_columns(self, names):
        """
        Override for keeping specified columns.

        Parameters
        ----------
        names : list or tuple
            A list or tuple of column names to keep.
        """
        names = self._translate_names(names)
        super().keep_columns(names)

    def remove_column(self, name):
        """
        Override for column removal.

        Parameters
        ----------
        name : str
            The name of the column to be removed.
        """
        name = self._translate_names(name)
        super().remove_column(name)

    def remove_columns(self, names):
        """
        Override for multiple column removal.

        Parameters
        ----------
        names : list or tuple
            A list or tuple of column names to be removed.
        """
        names = self._translate_names(names)
        super().remove_columns(names)

    def rename_column(self, name, new_name):
        """
        Override for column renaming.

        Parameters
        ----------
        name : str
            The current name of the column to be renamed.

        new_name : str
            The new name for the column.
        """
        name = self._translate_names(name)
        super().rename_column(name, new_name)

    def rename_columns(self, names, new_names):
        """
        Override for multiple column renaming.

        Parameters
        ----------
        names : list or tuple
            A list or tuple of current column names to be renamed.

        new_names : list or tuple
            A list or tuple of new names for the columns.
        """
        names = self._translate_names(names)
        super().rename_columns(names, new_names)

    def replace_column(self, name, col, **kwargs):
        """
        Override for column replacement.

        Parameters
        ----------
        name : str
            The current name of the column to be replaced.

        col : `Column` or `MaskedColumn`
            The new column to replace the existing one.

        **kwargs : dict, optional
            Additional keyword arguments passed to the parent method.
        """
        name = self._translate_names(name)
        super().replace_column(name, col, **kwargs)

    def add_index(self, names):
        """
        Override for index addition.

        Parameters
        ----------
        names : str or list or tuple
            The name(s) of the column(s) to be indexed.
        """
        names = self._translate_names(names)
        super().add_index(names)

    def remove_indices(self, names):
        """
        Override for index removal.

        Parameters
        ----------
        names : str or list or tuple
            The name(s) of the column(s) whose indices are to be
            removed.
        """
        names = self._translate_names(names)
        super().remove_indices(names)

    def sort(self, keys, **kwargs):
        """
        Override for sorting.

        Parameters
        ----------
        keys : str or list or tuple
            The name(s) of the column(s) to sort by.

        **kwargs : dict, optional
            Additional keyword arguments (e.g., ``kind``, ``reverse``)
            passed to the parent method.
        """
        keys = self._translate_names(keys)
        super().sort(keys, **kwargs)

    def group_by(self, keys, **kwargs):
        """
        Override for grouping.

        Parameters
        ----------
        keys : str or list or tuple
            The name(s) of the column(s) to group by.

        **kwargs : dict, optional
            Additional keyword arguments passed to the parent method.
        """
        keys = self._translate_names(keys)
        return super().group_by(keys, **kwargs)

    def copy(self, copy_data=True):
        """
        Override to preserve the deprecation map on copy.

        Parameters
        ----------
        copy_data : bool, optional
            Whether to copy the data. The default is `True`.

        Returns
        -------
        result : `DeprecatedColumnTable` or `DeprecatedColumnQTable`
            A copy of the table with the deprecation map preserved.
        """
        new_table = super().copy(copy_data=copy_data)
        new_table.deprecation_map = (self.deprecation_map.copy()
                                     if self.deprecation_map
                                     else None)
        new_table._deprecation_since = self._deprecation_since
        new_table._deprecation_until = self._deprecation_until
        return new_table


class DeprecatedColumnTable(DeprecatedColumnMixin, Table):
    """
    An Astropy Table with built-in support for deprecated names.
    """


class DeprecatedColumnQTable(DeprecatedColumnMixin, QTable):
    """
    An Astropy QTable with built-in support for deprecated names.
    """


def create_empty_deprecated_qtable(deprecation_map, *, since=None,
                                   until=None, **kwargs):
    """
    Create an empty `DeprecatedColumnQTable`.

    This is useful when building a table column by column rather than
    from a complete data dictionary.

    If ``photutils.future_column_names`` is `True`, a standard
    `~astropy.table.QTable` is returned instead, with no deprecation
    behavior.

    Parameters
    ----------
    deprecation_map : dict
        A dictionary mapping old (deprecated) names to new names.

    since : str or int, optional
        The version in which the column names were deprecated. If
        `None`, the deprecation version is not mentioned in the
        warning message.

    until : str or int, optional
        The version in which the old column names will be removed. If
        `None`, the removal version is not mentioned in the warning
        message.

    **kwargs : dict, optional
        Any other keywords accepted by the `~astropy.table.QTable`
        constructor (e.g., ``meta={...}``).

    Returns
    -------
    table : `DeprecatedColumnQTable` or `~astropy.table.QTable`
        A new empty QTable instance. If
        ``photutils.future_column_names`` is `True`, a standard
        `~astropy.table.QTable` is returned.

    Examples
    --------
    Create an empty table and add columns incrementally:

    >>> import warnings
    >>> from photutils.utils._deprecation import (
    ...     create_empty_deprecated_qtable)
    >>> dep_map = {'xcentroid': 'x_centroid'}
    >>> table = create_empty_deprecated_qtable(dep_map)
    >>> table['x_centroid'] = [1.0, 2.0, 3.0]
    >>> table.colnames
    ['x_centroid']

    Accessing via the deprecated name issues a warning:

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter('ignore')
    ...     col = table['xcentroid']
    >>> float(col[0])
    1.0
    """
    if _get_future_column_names():
        return QTable(**kwargs)

    table = DeprecatedColumnQTable(**kwargs)
    table.deprecation_map = deprecation_map
    table._deprecation_since = since
    table._deprecation_until = until
    return table


def create_deprecated_table_from_data(data, deprecation_map, *,
                                      since=None, until=None,
                                      use_qtable=False, **kwargs):
    """
    Create a new table from scratch with deprecated column name support.

    This function takes raw data and a deprecation map, renames the
    data keys internally, and constructs the appropriate ``Table`` or
    ``QTable`` subclass. All other keywords are passed directly to the
    underlying table constructor.

    If ``photutils.future_column_names`` is `True`, a standard
    `~astropy.table.QTable` or `~astropy.table.Table` is returned
    instead, with no deprecation behavior.

    Parameters
    ----------
    data : dict
        A dictionary of data for the table, using the OLD (soon to be
        deprecated) column names as keys.

    deprecation_map : dict
        A dictionary mapping old (deprecated) names to new names.

    since : str or int, optional
        The version in which the column names were deprecated. If
        `None`, the deprecation version is not mentioned in the
        warning message.

    until : str or int, optional
        The version in which the old column names will be removed. If
        `None`, the removal version is not mentioned in the warning
        message.

    use_qtable : bool, optional
        If ``True``, a ``DeprecatedColumnQTable`` (or
        `~astropy.table.QTable` when ``photutils.future_column_names``
        is `True`) will be created. Defaults to ``False``.

    **kwargs : dict, optional
        Any other keywords accepted by the ``astropy.table.Table``
        constructor (e.g., ``masked=True``, ``meta={...}``).

    Returns
    -------
    table : `DeprecatedColumnTable` or `DeprecatedColumnQTable`
        A new table instance with deprecation behavior. If
        ``photutils.future_column_names`` is `True`, a standard
        `~astropy.table.Table` or `~astropy.table.QTable` is returned.

    Examples
    --------
    Create a table with deprecated column names:

    >>> import warnings
    >>> from photutils.utils._deprecation import (
    ...     create_deprecated_table_from_data)
    >>> data = {'xcentroid': [1.0, 2.0], 'ycentroid': [3.0, 4.0]}
    >>> dep_map = {'xcentroid': 'x_centroid', 'ycentroid': 'y_centroid'}
    >>> table = create_deprecated_table_from_data(data, dep_map)

    The table stores data under the new column names:

    >>> table.colnames
    ['x_centroid', 'y_centroid']

    Accessing via a deprecated name issues a warning:

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter('ignore')
    ...     col = table['xcentroid']
    >>> float(col[0])
    1.0

    Use ``use_qtable=True`` to create a `~astropy.table.QTable`:

    >>> qtable = create_deprecated_table_from_data(
    ...     data, dep_map, use_qtable=True)
    >>> type(qtable).__name__
    'DeprecatedColumnQTable'
    """
    # Rename the keys in the data dictionary before creation
    renamed_data = {
        deprecation_map.get(k, k): v for k, v in data.items()
    }

    if _get_future_column_names():
        table_class = QTable if use_qtable else Table
        return table_class(renamed_data, **kwargs)

    table_class = (DeprecatedColumnQTable if use_qtable
                   else DeprecatedColumnTable)

    # Create the table instance
    table = table_class(renamed_data, **kwargs)
    table.deprecation_map = deprecation_map
    table._deprecation_since = since
    table._deprecation_until = until
    return table
