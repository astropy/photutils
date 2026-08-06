# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the photutils PSF converters.
"""

import os.path as op

import asdf
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from photutils.converters import _ASDF_ASTROPY_INSTALLED
from photutils.converters.image_models import GriddedPSFModelConverter
from photutils.psf import STDPSFGrid


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

    node = GriddedPSFModelConverter().to_yaml_tree_transform(psf, None, None)
    assert 'oversampling' in node
    assert 'oversampling' not in node['meta']

    filename = tmp_path / 'psf.asdf'
    with asdf.AsdfFile() as af:
        af['psf'] = psf
        af.write_to(filename)

    with asdf.open(filename) as af:
        psf2 = af['psf']
        assert_array_equal(psf2.oversampling, (2, 3))
        assert psf2.meta['oversampling'] == (2, 3)


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
def test_stdpsf_grid_converter_preserves_oversampling(tmp_path):
    """Test that a non-default oversampling value survives a round trip."""
    data = np.ones((4, 4, 4))
    meta = {
        'grid_xypos': np.array([(0, 0), (1, 0), (0, 1), (1, 1)]),
        'grid_shape': (2, 2),
        'oversampling': 8,
        'instrument': 'test-instrument',
        'custom': {'value': 42},
    }
    psfgrid = STDPSFGrid._from_asdf(data, meta)

    filename = tmp_path / 'psf.asdf'
    with asdf.AsdfFile() as af:
        af['psf'] = psfgrid
        af.write_to(filename)

    with asdf.open(filename) as af:
        psfgrid2 = af['psf']
        assert_array_equal(psfgrid2.oversampling, (8, 8))
        assert psfgrid2.meta['instrument'] == 'test-instrument'
        assert psfgrid2.meta['custom'] == {'value': 42}
        assert 'grid_xypos' not in psfgrid2.meta
        assert 'oversampling' not in psfgrid2.meta
        assert 'grid_shape' not in psfgrid2.meta


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
def test_stdpsf_grid_converter_repeated_grid_coordinate(tmp_path):
    """
    Test a round trip for an ACS/WFC grid, whose y coordinates are
    repeated where the two detectors abut.
    """
    filename = op.join(op.dirname(op.abspath(__file__)), '..', '..', 'psf',
                       'tests', 'data', 'STDPSF_ACSWFC_F814W_mock.fits')
    psfgrid = STDPSFGrid(filename)
    assert psfgrid._grid_shape == (10, 9)

    asdf_filename = tmp_path / 'psf.asdf'
    with asdf.AsdfFile() as af:
        af['psf'] = psfgrid
        af.write_to(asdf_filename)

    with asdf.open(asdf_filename) as af:
        psfgrid2 = af['psf']
        assert psfgrid2._grid_shape == psfgrid._grid_shape
        assert_array_equal(psfgrid2.grid_xypos, psfgrid.grid_xypos)
        assert_array_equal(psfgrid2._xgrid, psfgrid._xgrid)
        assert_array_equal(psfgrid2._ygrid, psfgrid._ygrid)
        assert_array_equal(psfgrid2.data, psfgrid.data)
        assert repr(psfgrid2) == repr(psfgrid)
