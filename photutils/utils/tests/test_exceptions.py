# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the exceptions module.
"""

import warnings

import pytest
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.utils import PhotutilsDeprecationWarning


class TestPhotutilsDeprecationWarning:
    """
    Tests for PhotutilsDeprecationWarning.
    """

    def test_subclass(self):
        assert issubclass(PhotutilsDeprecationWarning,
                          AstropyDeprecationWarning)

    def test_not_hidden_by_default(self):
        # Like AstropyDeprecationWarning, this is deliberately not a
        # DeprecationWarning subclass. Python hides those by default
        # outside of __main__.
        assert not issubclass(PhotutilsDeprecationWarning,
                              DeprecationWarning)

    def test_module_path(self):
        # The module path is what pytest and the warnings module
        # display, so it must identify photutils.
        assert (PhotutilsDeprecationWarning.__module__
                == 'photutils.utils.exceptions')

    def test_caught_by_astropy_filter(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', AstropyDeprecationWarning)
            with pytest.raises(PhotutilsDeprecationWarning):
                warnings.warn('test', PhotutilsDeprecationWarning,
                              stacklevel=1)
