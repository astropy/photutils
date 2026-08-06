# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the photutils PSF converters.
"""

import asdf
import pytest
from numpy.testing import assert_array_equal

from photutils.converters import _ASDF_ASTROPY_INSTALLED


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
    'circular_gaussian_prf_units',
    'circular_gaussian_prf',
    'circular_gaussian_sigma_prf_units',
    'circular_gaussian_sigma_prf',
    'circular_gaussian_psf_units',
    'circular_gaussian_psf',
    'gaussian_prf_units',
    'gaussian_prf',
    'gaussian_psf_units',
    'gaussian_psf',
    'moffat_psf_units',
    'moffat_psf',
    'image_psf',
    'gridded_psf',
    'stdpsf_single_detector',
], indirect=True)


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
@psf_params
def test_psf_converters(tmp_path, psfobj):
    """
    Test that the PSF converters can round-trip a PSF object.
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


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
@pytest.mark.parametrize('psfobj', ['gridded_psf'], indirect=True)
def test_gridded_psf_converter_preserves_modified_oversampling(tmp_path,
                                                               psfobj):
    """Test that a modified oversampling value survives a round trip."""
    psf, _ = psfobj
    psf.oversampling = (2, 3)

    filename = tmp_path / 'psf.asdf'
    with asdf.AsdfFile() as af:
        af['psf'] = psf
        af.write_to(filename)

    with asdf.open(filename) as af:
        psf2 = af['psf']
        assert_array_equal(psf2.oversampling, (2, 3))
        assert psf2.meta['oversampling'] == (2, 3)
