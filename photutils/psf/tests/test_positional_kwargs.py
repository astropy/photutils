# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for deprecation warnings when optional arguments are passed
positionally.
"""

import numpy as np
import pytest
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.psf.flags import decode_psf_flags
from photutils.psf.groupers import SourceGrouper


class TestDecodePSFFlagsPositionalKwargs:
    """
    Test decode_psf_flags warns for positional optional args.
    """

    def test_positional_warns(self):
        match = 'decode_psf_flags'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            decode_psf_flags(0, True)  # noqa: FBT003

    def test_keyword_no_warning(self):
        decode_psf_flags(0, return_bit_values=True)


class TestSourceGrouperPositionalKwargs:
    """
    Test SourceGrouper.__call__ warns for positional optional args.
    """

    def test_positional_warns(self):
        grouper = SourceGrouper(min_separation=10)
        x = np.array([0.0, 5.0, 50.0])
        y = np.array([0.0, 5.0, 50.0])
        match = '__call__'
        with pytest.warns(AstropyDeprecationWarning, match=match):
            grouper(x, y, True)  # noqa: FBT003

    def test_keyword_no_warning(self):
        grouper = SourceGrouper(min_separation=10)
        x = np.array([0.0, 5.0, 50.0])
        y = np.array([0.0, 5.0, 50.0])
        grouper(x, y, return_groups_object=True)
