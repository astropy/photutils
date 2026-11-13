# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the deprecated ``photutils.converters`` and
``photutils.extension`` modules.
"""

import importlib

import pytest

from photutils.asdf.converters import _ASDF_ASTROPY_INSTALLED
from photutils.utils.exceptions import PhotutilsDeprecationWarning

# The PSF converter modules need asdf-astropy
_PSF_CONVERTER_MODULES = ('converters.functional_models',
                          'converters.image_models')

DEPRECATED_MODULES = [
    pytest.param(name, marks=pytest.mark.skipif(
        name in _PSF_CONVERTER_MODULES and not _ASDF_ASTROPY_INSTALLED,
        reason='asdf-astropy is required'))
    for name in ('converters', 'converters.apertures',
                 'converters.functional_models', 'converters.image_models',
                 'extension')
]


@pytest.mark.parametrize('name', DEPRECATED_MODULES)
def test_deprecated_module(name):
    """
    Test that the public names of a deprecated module warn and are the
    same objects as those in the new module.
    """
    old = importlib.import_module(f'photutils.{name}')
    new = importlib.import_module(f'photutils.asdf.{name}')
    assert old.__all__ == new.__all__
    assert old.__all__
    assert dir(old) == sorted(old.__all__)

    match = (f'{old.__name__} is deprecated.*removed in version 4.0.*'
             f'Use {new.__name__} instead')
    for attr in old.__all__:
        with pytest.warns(PhotutilsDeprecationWarning, match=match):
            assert getattr(old, attr) is getattr(new, attr)


@pytest.mark.parametrize('name', DEPRECATED_MODULES)
def test_deprecated_module_unknown_name(name):
    """
    Test that a name that is not public in a deprecated module raises
    an `AttributeError`.
    """
    old = importlib.import_module(f'photutils.{name}')
    match = 'has no attribute'
    with pytest.raises(AttributeError, match=match):
        _ = old.missing_name
