#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Use "distribute" - the setuptools fork that supports python 3.
from distribute_setup import use_setuptools
use_setuptools()

import glob
import os
import sys
from setuptools import setup, find_packages

#A dirty hack to get around some early import/configurations ambiguities
#This is the same as setup_helpers.set_build_mode(), but does not require
#importing setup_helpers
if sys.version_info[0] >= 3:
    import builtins
else:
    import __builtin__ as builtins
builtins._PACKAGE_SETUP_ = True

import astropy
from astropy import setup_helpers
from astropy.version_helper import get_git_devstr, generate_version_py

import photutils

# Set affiliated package-specific settings
PACKAGENAME = 'photutils'
DESCRIPTION = 'Astropy affiliated package for image photometry tasks.'
LONG_DESCRIPTION = photutils.__doc__
AUTHOR = 'Photutils developers, Kyle Barbary'
AUTHOR_EMAIL = 'kylebarbary@gmail.com'
LICENSE = 'BSD'
URL = 'http://astropy.org'

#version should be PEP386 compatible (http://www.python.org/dev/peps/pep-0386)
version = '0.0.dev'

# Indicates if this version is a release version
release = 'dev' not in version

cmdclassd = setup_helpers.register_commands(PACKAGENAME, version, release)

if not release:
    version += get_git_devstr(False)
generate_version_py(PACKAGENAME, version, release,
                    setup_helpers.get_debug_option())

# Adjust the compiler in case the default on this platform is to use a
# broken one.
setup_helpers.adjust_compiler(PACKAGENAME)

# Use the find_packages tool to locate all packages and modules
packagenames = find_packages()

# Treat everything in scripts except README.rst as a script to be installed
scripts = glob.glob(os.path.join('scripts', '*'))
scripts.remove(os.path.join('scripts', 'README.rst'))

# Additional C extensions that are not Cython-based should be added here.
extensions = []

# A dictionary to keep track of all package data to install
package_data = {PACKAGENAME: ['data/*']}

# A dictionary to keep track of extra packagedir mappings
package_dirs = {}

# Update extensions, package_data, packagenames and package_dirs from
# any sub-packages that define their own extension modules and package
# data.  See the docstring for setup_helpers.update_package_files for
# more details.
setup_helpers.update_package_files(PACKAGENAME, extensions, package_data,
                                   packagenames, package_dirs)



setup(name=PACKAGENAME,
      version=version,
      description=DESCRIPTION,
      packages=packagenames,
      package_data=package_data,
      package_dir=package_dirs,
      ext_modules=extensions,
      scripts=scripts,
      requires=['astropy'],
      install_requires=['astropy'],
      provides=[PACKAGENAME],
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      url=URL,
      long_description=LONG_DESCRIPTION,
      cmdclass=cmdclassd,
      zip_safe=False,
      use_2to3=True
      )
