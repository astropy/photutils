# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.

from astropy.tests.pytest_plugins import *

## Uncomment the following line to treat all DeprecationWarnings as
## exceptions
enable_deprecations_as_exceptions()

# Add scikit-image to test header information
try:
    PYTEST_HEADER_MODULES['scikit-image'] = 'skimage'
    del PYTEST_HEADER_MODULES['h5py']

except NameError:  # astropy < 1.0
    pass
