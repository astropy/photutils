# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Tests for the photutils aperture converters.
"""
import numpy as np

import asdf
from photutils.aperture import CircularAperture


apertures = [
    CircularAperture(positions=[(1, 2), (3, 4)], r=5),
    CircularAperture(positions=[(5, 6)], r=7),
]


def test_aperture_converters(tmp_path):
    """
    Test that the aperture converters can round-trip an aperture object.
    """
    for aperture in apertures:
        with asdf.AsdfFile() as af:
            af["aperture"] = aperture
            af.write_to(tmp_path / "aperture.asdf")

        with asdf.open(tmp_path / "aperture.asdf") as af:
            aperture2 = af["aperture"]

        assert np.all(aperture.positions == aperture2.positions)
        assert aperture.r == aperture2.r
