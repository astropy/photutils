# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides the tools used to run the test suite.
"""

import pytest
from astropy.utils.introspection import minversion

PYTEST_LT_80 = not minversion(pytest, '8.0.dev')
