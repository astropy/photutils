# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _progress_bars module.
"""

import builtins
import importlib
from unittest.mock import patch

import pytest

from photutils.utils._optional_deps import HAS_TQDM
from photutils.utils._progress_bars import add_progress_bar


def test_add_progress_bar_no_tqdm():
    """
    Test that when tqdm is not available, the original iterable is
    returned.
    """
    items = range(5)
    result = add_progress_bar(items, text=False)
    if not HAS_TQDM:
        assert result is items


def test_add_progress_bar_text():
    """
    Test add_progress_bar with text=True. When tqdm is available, a tqdm
    progress bar is returned.
    """
    items = range(5)
    result = add_progress_bar(items, desc='test', text=True)
    if HAS_TQDM:
        # Should be a tqdm instance wrapping the iterable
        assert hasattr(result, '__iter__')
        assert list(result) == list(range(5))
    else:
        assert result is items


@pytest.mark.skipif(not HAS_TQDM, reason='tqdm is required')
def test_add_progress_bar_no_text():
    """
    Test add_progress_bar with text=False (default) when tqdm is
    available. This exercises the ipywidgets try/except branch.
    """
    items = range(5)
    result = add_progress_bar(items, desc='test', text=False)
    assert hasattr(result, '__iter__')
    assert list(result) == list(range(5))


@pytest.mark.skipif(not HAS_TQDM, reason='tqdm is required')
def test_add_progress_bar_no_ipywidgets():
    """
    Test add_progress_bar with text=False when ipywidgets is not
    available, exercising the ImportError fallback branch.
    """
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == 'ipywidgets':
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    items = range(5)
    with patch('builtins.__import__', side_effect=mock_import):
        result = add_progress_bar(items, desc='test', text=False)
    assert hasattr(result, '__iter__')
    assert list(result) == list(range(5))


def test_dummy_tqdm_class():
    """
    Test the dummy tqdm class that is defined when tqdm is not
    installed. We reload the module with tqdm mocked as unavailable.
    """
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if 'tqdm' in name:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    with patch('builtins.__import__', side_effect=mock_import):
        import photutils.utils._progress_bars as pb_mod
        importlib.reload(pb_mod)

    # The dummy tqdm class should now be defined
    dummy = pb_mod.tqdm(total=10, desc='test')
    assert dummy.__enter__() is dummy
    assert dummy.__exit__(None, None, None) is None
    assert dummy.update(1) is None
    assert dummy.set_postfix_str('test') is None

    # Reload again to restore the original state
    importlib.reload(pb_mod)
