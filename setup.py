#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import builtins
import os

# Ensure that astropy-helpers is available
import ah_bootstrap  # noqa

from setuptools import setup
from setuptools.config import read_configuration

from astropy_helpers.distutils_helpers import is_distutils_display_option
from astropy_helpers.setup_helpers import register_commands, get_package_info
from astropy_helpers.version_helpers import generate_version_py

# Store the package name in a built-in variable so it's easy
# to get from other parts of the setup infrastructure
builtins._ASTROPY_PACKAGE_NAME_ = read_configuration('setup.cfg')['metadata']['name']

# Create a dictionary with setup command overrides. Note that this gets
# information about the package (name and version) from the setup.cfg file.
cmdclassd = register_commands()

# Freeze build information in version.py. Note that this gets information
# about the package (name and version) from the setup.cfg file.
version = generate_version_py()

# Get configuration information from all of the various subpackages.
# See the docstring for setup_helpers.update_package_files for more
# details.
package_info = get_package_info()

if is_distutils_display_option():
    # Avoid installing setup_requires dependencies if the user just
    # queries for information
    setup_requires = []
else:
    setup_requires = read_configuration('setup.cfg')['options']['setup_requires']
    # Make sure we have the packages needed for building photuils, but do not
    # require them when installing from an sdist as the c files are included.
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'PKG-INFO')):
        setup_requires.extend(['cython>=0.28'])

setup(setup_requires=setup_requires, version=version, cmdclass=cmdclassd,
      **package_info)
