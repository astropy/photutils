# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for progress bars.
"""

# pylint: disable-next=E0611
from photutils.utils._optional_deps import HAS_TQDM


def add_progress_bar(iterable=None, desc=None, total=None, text=False):
    """
    Add a progress bar for an iterable.

    Parameters
    ----------
    iterable : iterable, optional
        The iterable for which to add a progress bar. Set to `None` to
        manually manage the progress bar updates.

    desc : str, optional
        The prefix string for the progress bar.

    total : int, optional
        The number of expected iterations. If unspecified, len(iterable)
        is used if possible.

    text : bool, optional
        Whether to always use a text-based progress bar.

    Returns
    -------
    result : tqdm iterable
        A tqdm progress bar. If in a notebook and ipywidgets is
        installed, it will return a ipywidgets-based progress bar.
        Otherwise it will return a text-based progress bar.
    """
    if HAS_TQDM:
        if text:
            from tqdm import tqdm
        else:
            try:  # pragma: no cover
                # pylint: disable-next=W0611
                from ipywidgets import FloatProgress  # noqa: F401
                from tqdm.auto import tqdm
            except ImportError:  # pragma: no cover
                from tqdm import tqdm

        iterable = tqdm(iterable=iterable, desc=desc, total=total)  # pragma: no cover

    return iterable
