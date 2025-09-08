# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define test helpers.
"""

import pytest
from astropy.utils.introspection import minversion

PYTEST_LT_80 = not minversion(pytest, '8.0')
