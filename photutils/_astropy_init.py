# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os

__all__ = ['__version__', 'test']

try:
    from .version import version as __version__
except ImportError:
    __version__ = ''

# Create the test function for self test
from astropy.tests.runner import TestRunner

test = TestRunner.make_test_runner_in(os.path.dirname(__file__))
