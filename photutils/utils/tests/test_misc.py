# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _misc module.
"""

import pytest

from photutils.utils._misc import _get_meta


@pytest.mark.parametrize('utc', (False, True))
def test_get_meta(utc):
    meta = _get_meta(utc)
    keys = ('date', 'version')
    for key in keys:
        assert key in meta

    versions = meta['version']
    assert isinstance(versions, dict)
    keys = ('Python', 'photutils', 'astropy', 'numpy', 'scipy', 'skimage',
            'sklearn', 'matplotlib', 'gwcs', 'bottleneck')
    for key in keys:
        assert key in versions
