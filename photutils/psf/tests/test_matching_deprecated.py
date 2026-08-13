# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the deprecated photutils.psf.matching subpackage.
"""

import pytest
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils import psf_matching
from photutils.psf import matching as old_matching

DEPRECATION_MATCH = 'photutils.psf.matching is deprecated'


@pytest.mark.parametrize('name', old_matching.__all__)
def test_deprecated_attribute_access(name):
    """
    Test that accessing each public name from the old location emits a
    deprecation warning and returns the object from the new location.
    """
    with pytest.warns(AstropyDeprecationWarning, match=DEPRECATION_MATCH):
        obj = getattr(old_matching, name)
    assert obj is getattr(psf_matching, name)


def test_deprecated_from_import():
    """
    Test that a from-import of a name public in photutils < 3.0 emits
    a deprecation warning.
    """
    with pytest.warns(AstropyDeprecationWarning, match=DEPRECATION_MATCH):
        from photutils.psf.matching import resize_psf
    assert resize_psf is psf_matching.resize_psf


def test_all_contains_old_public_names():
    """
    Test that every name public in photutils < 3.0 is still importable
    from the old location.
    """
    old_public_names = ('create_matching_kernel', 'resize_psf',
                        'CosineBellWindow', 'HanningWindow',
                        'SplitCosineBellWindow', 'TopHatWindow',
                        'TukeyWindow')
    for name in old_public_names:
        assert name in old_matching.__all__


def test_invalid_attribute():
    """
    Test that an invalid attribute raises AttributeError without a
    deprecation warning.
    """
    match = "module 'photutils.psf.matching' has no attribute 'invalid'"
    with pytest.raises(AttributeError, match=match):
        getattr(old_matching, 'invalid')  # noqa: B009


def test_dir():
    """
    Test that dir() lists all public names.
    """
    assert sorted(dir(old_matching)) == sorted(old_matching.__all__)
    assert 'resize_psf' in dir(old_matching)
