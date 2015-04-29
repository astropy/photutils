# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.

import os
from astropy.tests.pytest_plugins import *

# This is to figure out photutils version, rather than using Astropy's
from . import version

try:
    packagename = os.path.basename(os.path.dirname(__file__))
    TESTED_VERSIONS[packagename] = version.version
except NameError:
    pass

# Uncomment the following line to treat all DeprecationWarnings as
# exceptions
enable_deprecations_as_exceptions()

# Add scikit-image to test header information
PYTEST_HEADER_MODULES['scikit-image'] = 'skimage'
PYTEST_HEADER_MODULES['Astropy'] = 'astropy'
del PYTEST_HEADER_MODULES['h5py']
