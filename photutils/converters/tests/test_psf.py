# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the photutils PSF converters.
"""

import asdf
import pytest
from numpy.testing import assert_array_equal

from photutils.converters import _ASDF_ASTROPY_INSTALLED


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
def test_psf_converters(tmp_path, airy_disk_psf):
    """
    Test that the PSF converters can round-trip a PSF object.
    """
    psf, pars = airy_disk_psf
    with asdf.AsdfFile() as af:
        af['psf'] = psf
        af.write_to(tmp_path / 'psf.asdf')

        with asdf.open(tmp_path / 'psf.asdf') as af:
            psf2 = af['psf']
            for parameter in pars:
                assert_array_equal(getattr(psf, parameter),
                                   getattr(psf2, parameter))
