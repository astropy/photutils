# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _misc module.
"""

import builtins
from unittest.mock import patch

import pytest

from photutils.utils._misc import _get_date, _get_meta, _get_version_info


def test_get_version_info_import_error():
    """
    Test that _get_version_info returns None for packages that cannot
    be imported.
    """
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == 'gwcs':
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    with patch('builtins.__import__', side_effect=mock_import):
        versions = _get_version_info()
    assert versions['gwcs'] is None


@pytest.mark.parametrize('utc', [False, True])
def test_get_date_oserror_fallback(utc):
    """
    Test that _get_date falls back when astimezone raises OSError.
    """
    with patch('photutils.utils._misc.datetime') as mock_dt:
        mock_now = mock_dt.now.return_value
        mock_now.astimezone.side_effect = OSError('no timezone')
        mock_now.strftime.return_value = '2025-01-01 00:00:00'
        mock_dt.now.return_value = mock_now
        from datetime import UTC
        mock_dt.UTC = UTC
        result = _get_date(utc=utc)
        assert isinstance(result, str)


@pytest.mark.parametrize('utc', [False, True])
def test_get_meta(utc):
    """
    Test _get_meta returns expected keys.
    """
    meta = _get_meta(utc)
    keys = ('date', 'version')
    for key in keys:
        assert key in meta

    versions = meta['version']
    assert isinstance(versions, dict)
    keys = ('Python', 'photutils', 'astropy', 'numpy', 'scipy', 'skimage',
            'matplotlib', 'gwcs', 'bottleneck')
    for key in keys:
        assert key in versions
