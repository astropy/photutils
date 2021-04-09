# Licensed under a 3-clause BSD style license - see LICENSE.rst

__all__ = ['__version__']

# this indicates whether or not we are in the package's setup.py
try:
    _ASTROPY_SETUP_
except NameError:
    import builtins
    builtins._ASTROPY_SETUP_ = False

try:
    from .version import version as __version__
except ImportError:
    __version__ = ''


if not _ASTROPY_SETUP_:  # noqa
    import os

    # Create the test function for self test
    from astropy.tests.runner import TestRunner
    test = TestRunner.make_test_runner_in(os.path.dirname(__file__))
    test.__test__ = False
    __all__ += ['test']
