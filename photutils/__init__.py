# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Photutils is an Astropy affiliated package to provide tools for
detecting and performing photometry of astronomical sources.

It also has tools for background estimation, ePSF building, PSF
matching, radial profiles, centroiding, and morphological measurements.
"""

try:
    from .version import version as __version__
except ImportError:
    __version__ = ''

from .utils._deprecation import use_future_column_names  # noqa: F401


future_column_names = False
"""
If `True`, all photutils functions return standard
`~astropy.table.QTable` (or `~astropy.table.Table`) instances with the
new column names instead of deprecated-column subclasses. Set this to
`True` after updating your code to use the new column names to verify
compatibility with the 4.0 behavior.

For a scoped override that does not affect the global flag, use
`photutils.use_future_column_names` as a context manager::

    with photutils.use_future_column_names():
        table = cat.to_table()  # returns a plain QTable
"""
