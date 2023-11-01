# This file is used to configure the behavior of pytest when using the Astropy
# test infrastructure. It needs to live inside the package in order for it to
# get picked up when running the tests inside an interpreter using
# packagename.test

import numpy as np
from astropy.utils import minversion

try:
    from pytest_astropy_header.display import (PYTEST_HEADER_MODULES,
                                               TESTED_VERSIONS)
    ASTROPY_HEADER = True
except ImportError:
    ASTROPY_HEADER = False

# do not remove until we drop support for NumPy < 2.0
if minversion(np, '2.0.0.dev0+git20230726'):
    np.set_printoptions(legacy='1.25')


def pytest_configure(config):
    if ASTROPY_HEADER:
        config.option.astropy_header = True

        # Customize the following lines to add/remove entries from the
        # list of packages for which version numbers are displayed when
        # running the tests.
        PYTEST_HEADER_MODULES.clear()
        deps = ['NumPy', 'SciPy', 'Matplotlib', 'Astropy', 'scikit-image',
                'scikit-learn', 'GWCS', 'Bottleneck', 'tqdm', 'Rasterio',
                'Shapely']
        for dep in deps:
            PYTEST_HEADER_MODULES[dep] = dep.lower()

        from photutils import __version__
        TESTED_VERSIONS['photutils'] = __version__
