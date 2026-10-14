# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for opting in to the future table column names.

This module is imported by the top-level package, so it must not import
anything from ``photutils.utils`` or any other heavy dependency.
"""

from contextlib import contextmanager
from contextvars import ContextVar

_SENTINEL = object()
_future_column_names_var = ContextVar(
    'photutils_future_column_names', default=_SENTINEL,
)


def get_future_column_names():
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
    though ``photutils.future_column_names`` were set to ``enabled``,
    without modifying the global flag. This is safe to use in
    multithreaded and async code because the override is stored in a
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
