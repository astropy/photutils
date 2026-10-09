# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This plugin provides support for specifying the -P option to test both
code and docs for a specific sub-package.
"""

import os

import pytest
from packaging.version import Version

_pytest_version = Version(pytest.__version__)
PYTEST_GE_8_0 = any([_pytest_version.is_devrelease,
                     _pytest_version.is_prerelease,
                     _pytest_version >= Version('8.0')])


def pytest_addoption(parser):

    parser.addoption("--package", "-P", action="store",
                     help="The name of a specific package to test, e.g. "
                          "'io.fits' or 'utils'. Accepts comma separated "
                          "string to specify multiple packages.")


if PYTEST_GE_8_0:

    @pytest.hookimpl(tryfirst=True)
    def pytest_ignore_collect(collection_path, config):

        # NOTE: it is important that when we don't want to skip a file we return
        # None and not False - if we return False pytest will not call any other
        # pytest_ignore_collect function in other plugins, e.g. pytest-doctestplus.

        # If the --package/-P option wasn't specified, don't do anything
        if config.getvalue('package') is None:
            return None

        # If the path is a directory, never skip - just do the filtering on a file
        # by file basis.
        if collection_path.is_dir():
            return None

        # Otherwise ignore filename for remainder of checks
        path = str(collection_path.parent)

        # Split path into components
        path = path.split(os.path.sep)

        # Now cycle through and find the top level of the package - this is the
        # last one that contains an ``__init__.py`` or ``index.rst`` file. We need
        # to make sure that at least one of these files was found before escaping.
        found_prev = False
        for i in range(len(path), -1, -1):
            subpath = os.path.sep.join(path[:i])
            found = (os.path.exists(os.path.join(subpath, '__init__.py')) or
                     os.path.exists(os.path.join(subpath, 'index.rst')))
            if found_prev and not found:
                break
            found_prev = found

        subpackage_path = path[i+1:]

        # Find selected sub-packages
        selected = config.getvalue('package').strip().split(',')

        # Finally, we check if this is one of the specified ones
        for subpackage_target in selected:
            for i, target in enumerate(subpackage_target.split('.')):
                if i >= len(subpackage_path) or target != subpackage_path[i]:
                    break

            else:
                return None

        return True

else:

    @pytest.hookimpl(tryfirst=True)
    def pytest_ignore_collect(path, config):

        # NOTE: it is important that when we don't want to skip a file we return
        # None and not False - if we return False pytest will not call any other
        # pytest_ignore_collect function in other plugins, e.g. pytest-doctestplus.

        # If the --package/-P option wasn't specified, don't do anything
        if config.getvalue('package') is None:
            return None

        # If the path is a directory, never skip - just do the filtering on a file
        # by file basis.
        if os.path.isdir(path):
            return None

        # Otherwise ignore filename for remainder of checks
        path = os.path.dirname(path)

        # Split path into components
        path = path.split(os.path.sep)

        # Now cycle through and find the top level of the package - this is the
        # last one that contains an ``__init__.py`` or ``index.rst`` file. We need
        # to make sure that at least one of these files was found before escaping.
        found_prev = False
        for i in range(len(path), -1, -1):
            subpath = os.path.sep.join(path[:i])
            found = (os.path.exists(os.path.join(subpath, '__init__.py')) or
                     os.path.exists(os.path.join(subpath, 'index.rst')))
            if found_prev and not found:
                break
            found_prev = found

        subpackage_path = path[i+1:]

        # Find selected sub-packages
        selected = config.getvalue('package').strip().split(',')

        # Finally, we check if this is one of the specified ones
        for subpackage_target in selected:
            for i, target in enumerate(subpackage_target.split('.')):
                if i >= len(subpackage_path) or target != subpackage_path[i]:
                    break

            else:
                return None

        return True
