# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for progress bars.
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
    result : tqdm iterable, iterable, or `None`
        When tqdm is installed, a tqdm progress bar (ipywidgets-based in
        a notebook if available, otherwise text-based). When tqdm is not
        installed, the original iterable is returned unchanged (which
        may be `None`).

    Notes
    -----
    When ``iterable=None`` and ``total`` is provided, the returned tqdm
    object is a manually-managed progress bar. The caller is responsible
    for advancing it by calling ``.update()`` and closing it when
    finished. This mode is used when the caller drives the loop directly
    (e.g., with a ``while`` loop) rather than iterating over a sequence.
    When tqdm is not installed, `None` is returned in this case and the
    caller must guard against it.
    """
    if HAS_TQDM:
        if text:
            from tqdm import tqdm
        else:
            try:
                # pylint: disable-next=W0611
                from ipywidgets import FloatProgress  # noqa: F401
                from tqdm.auto import tqdm
            except ImportError:
                from tqdm import tqdm

        iterable = tqdm(iterable=iterable, desc=desc,
                        total=total)

    return iterable


# Define tqdm as a dummy class if it is not available.
# This is needed to use tqdm as a context manager with multiprocessing.
try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:  # noqa: N801
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def update(self, *args, **kwargs):
            pass

        def set_postfix_str(self, *args, **kwargs):
            pass
