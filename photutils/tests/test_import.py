# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the import of the top-level package.
"""

import subprocess  # nosec B404
import sys

import pytest


@pytest.mark.parametrize('module', ['photutils.utils', 'astropy.table',
                                    'scipy'])
def test_import_is_lightweight(module):
    """
    Test that importing the top-level package does not import its
    subpackages or their heavy dependencies.
    """
    code = f'import sys; import photutils; print({module!r} in sys.modules)'
    result = subprocess.run(  # noqa: S603  # nosec B603
        [sys.executable, '-c', code], check=True, capture_output=True,
        text=True)
    assert result.stdout.strip() == 'False'
