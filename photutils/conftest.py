# This file is used to configure the behavior of pytest when using the Astropy
# test infrastructure. It needs to live inside the package in order for it to
# get picked up when running the tests inside an interpreter using
# packagename.test

try:
    from pytest_astropy_header.display import (PYTEST_HEADER_MODULES,
                                               TESTED_VERSIONS)
    ASTROPY_HEADER = True
except ImportError:
    ASTROPY_HEADER = False


def pytest_configure(config):
    if ASTROPY_HEADER:
        config.option.astropy_header = True

        # Customize the following lines to add/remove entries from the
        # list of packages for which version numbers are displayed when
        # running the tests.
        PYTEST_HEADER_MODULES['Cython'] = 'Cython'
        PYTEST_HEADER_MODULES['Numpy'] = 'numpy'
        PYTEST_HEADER_MODULES['Astropy'] = 'astropy'
        PYTEST_HEADER_MODULES['Scipy'] = 'scipy'
        PYTEST_HEADER_MODULES['Matplotlib'] = 'matplotlib'
        PYTEST_HEADER_MODULES['scikit-image'] = 'skimage'
        PYTEST_HEADER_MODULES['scikit-learn'] = 'sklearn'
        PYTEST_HEADER_MODULES.pop('Pandas', None)
        PYTEST_HEADER_MODULES.pop('h5py', None)

        from photutils import __version__
        TESTED_VERSIONS['photutils'] = __version__
