# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from astropy.io import fits
from astropy.tests.helper import remote_data

from ..centerer import IsophoteCenterer
from ..geometry import EllipseGeometry
from ...datasets import get_path


@remote_data
def test_centerer():
    path = get_path('isophote/M51.fits', location='photutils-datasets',
                    cache=True)
    hdu = fits.open(path)
    data = hdu[0].data
    hdu.close()

    geometry = EllipseGeometry(252, 253, 10., 0.2, np.pi/2)
    centerer = IsophoteCenterer(data, geometry, False)
    centerer.center()

    assert geometry.x0 == 257.
    assert geometry.y0 == 258.
