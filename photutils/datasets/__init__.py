# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Load or make datasets for examples and tests.

Some small data files are bundled in the ``photutils`` repo.

Large data files are available from a ``photutils-datasets` repo
and loaded into the Astropy cache on the user's machine on first load.
"""

from .load import *
from .make import *
