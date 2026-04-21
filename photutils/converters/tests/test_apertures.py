# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Tests for the photutils aperture converters.
"""
import asdf
import numpy as np
import pytest

from photutils.aperture import CircularAperture
from photutils.converters import ASDF_ASTROPY_INSTALLED

apertures = [
    CircularAperture(positions=[(1, 2), (3, 4)], r=5),
    CircularAperture(positions=(5, 6), r=7),
]


@pytest.mark.skipif(not ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
@pytest.mark.parametrize('aperture', apertures)
def test_aperture_converters(tmp_path, aperture):
    """
    Test that the aperture converters can round-trip an aperture object.
    """
    with asdf.AsdfFile() as af:
        af['aperture'] = aperture
        af.write_to(tmp_path / 'aperture.asdf')

    with asdf.open(tmp_path / 'aperture.asdf') as af:
        aperture2 = af['aperture']

        assert np.all(aperture.positions == aperture2.positions)
        assert aperture.r == aperture2.r
