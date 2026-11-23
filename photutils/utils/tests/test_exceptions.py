# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the exceptions module.
"""

import warnings

import pytest
from astropy.utils.exceptions import (AstropyDeprecationWarning,
                                      AstropyUserWarning)

from photutils.utils import NoDetectionsWarning, PhotutilsDeprecationWarning


class TestNoDetectionsWarning:
    """
    Tests for NoDetectionsWarning.
    """

    def test_subclass(self):
        assert issubclass(NoDetectionsWarning, AstropyUserWarning)

    def test_shown_after_astropy_user_warning_ignored(self):
        # A filter added after a broader one takes precedence, so the
        # warning can be kept while other AstropyUserWarning are
        # ignored.
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('ignore', AstropyUserWarning)
            warnings.simplefilter('always', NoDetectionsWarning)
            warnings.warn('other', AstropyUserWarning, stacklevel=1)
            warnings.warn('test', NoDetectionsWarning, stacklevel=1)

        assert [record.category for record in records] == [
            NoDetectionsWarning]


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
