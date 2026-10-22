# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the import of the top-level package.
"""

import os
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
    # Use the import path of this process so that the child imports the
    # same photutils (e.g., a source tree that pytest added to sys.path).
    # The -P option keeps the child from putting the current directory
    # first on its import path, where a different photutils could shadow
    # this one.
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path)}
    result = subprocess.run(  # noqa: S603  # nosec B603
        [sys.executable, '-P', '-c', code], check=True,
        capture_output=True, text=True, env=env, timeout=120)
    assert result.stdout.strip() == 'False'
