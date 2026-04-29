# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Tests for the photutils PSF converters.
"""
import asdf
import pytest
from numpy.testing import assert_array_equal

from photutils.converters import ASDF_ASTROPY_INSTALLED


@pytest.fixture
def psfobj(request):
    """
    A pytest fixture that returns a PSF model and the
    list of parameters to test.
    """
    return request.getfixturevalue(request.param)


psf_params = pytest.mark.parametrize('psfobj', [
    'airy_disk_units',
    'airy_disk',
], indirect=True)


@pytest.mark.skipif(not ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
@psf_params
def test_psf_converters(tmp_path, psfobj):
    """
    Test that the PSF converters can round-trip
    a PSF object.
    """
    psf, pars = psfobj
    with asdf.AsdfFile() as af:
        af['psf'] = psf
        af.write_to(tmp_path / 'psf.asdf')

        with asdf.open(tmp_path / 'psf.asdf') as af:
            psf2 = af['psf']
            for parameter in pars:
                assert_array_equal(getattr(psf, parameter),
                                   getattr(psf2, parameter))
