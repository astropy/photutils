# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the photutils PSF converters.
"""

import asdf
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from photutils.converters import _ASDF_ASTROPY_INSTALLED
from photutils.converters.image_models import GriddedPSFModelConverter
from photutils.converters.tests import examples
from photutils.extension import PHOTUTILS_PSF_CONVERTERS
from photutils.psf import STDPSFGrid

PSF_EXAMPLES = [
    examples.airy_disk_units,
    examples.airy_disk,
    examples.circular_gaussian_prf_units,
    examples.circular_gaussian_prf,
    examples.circular_gaussian_sigma_prf_units,
    examples.circular_gaussian_sigma_prf,
    examples.circular_gaussian_psf_units,
    examples.circular_gaussian_psf,
    examples.gaussian_prf_units,
    examples.gaussian_prf,
    examples.gaussian_psf_units,
    examples.gaussian_psf,
    examples.moffat_psf_units,
    examples.moffat_psf,
    examples.image_psf,
    examples.gridded_psf,
    examples.stdpsf_single_detector,
]


def _example_id(example):
    """
    Return the test ID of an example function.
    """
    return example.__name__


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
@pytest.mark.parametrize('example', PSF_EXAMPLES, ids=_example_id)
def test_psf_converters(tmp_path, example):
    """
    Test that the PSF converters can round-trip a PSF object.
    """
    psf, params = example()

    filename = tmp_path / 'psf.asdf'
    with asdf.AsdfFile() as af:
        af['psf'] = psf
        af.write_to(filename)

    with asdf.open(filename) as af:
        psf2 = af['psf']
        for param in params:
            assert_array_equal(getattr(psf, param), getattr(psf2, param))


# Examples of every PSF model (STDPSFGrid is not a model, so it has
# no fitting state)
MODEL_EXAMPLES = [
    examples.airy_disk,
    examples.circular_gaussian_prf,
    examples.circular_gaussian_psf,
    examples.circular_gaussian_sigma_prf,
    examples.gaussian_prf,
    examples.gaussian_psf,
    examples.moffat_psf,
    examples.image_psf,
    examples.gridded_psf,
]


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
@pytest.mark.parametrize('example', MODEL_EXAMPLES, ids=_example_id)
def test_psf_converters_preserve_fitting_state(tmp_path, example):
    """
    Test that non-default fixed, bounds, and name states survive a
    round trip.

    The file stores only the fixed=True entries and the non-empty
    bounds, so parameters whose class defaults differ from that
    baseline (e.g., the shape parameters, which default to fixed=True
    with a lower bound) must be reset when reading.
    """
    psf, _ = example()
    for name in psf.param_names:
        psf.fixed[name] = not psf.fixed[name]
        psf.bounds[name] = (None, None)
    psf.bounds[psf.param_names[0]] = (0.5, 100.0)
    psf.name = 'my-psf'

    filename = tmp_path / 'psf.asdf'
    with asdf.AsdfFile() as af:
        af['psf'] = psf
        af.write_to(filename)

    with asdf.open(filename) as af:
        psf2 = af['psf']

    assert psf2.name == psf.name
    assert psf2.fixed == psf.fixed
    assert psf2.bounds == psf.bounds


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
def test_psf_examples_cover_all_converters():
    """
    Test that the round-trip test exercises every type handled by a PSF
    converter.
    """
    covered = {type(example()[0]).__name__ for example in PSF_EXAMPLES}
    handled = {name.rsplit('.', 1)[-1]
               for converter in PHOTUTILS_PSF_CONVERTERS
               for name in converter.types}
    assert covered == handled


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
def test_gridded_psf_converter_preserves_modified_oversampling(tmp_path):
    """
    Test that a modified oversampling value survives a round trip.
    """
    psf, _ = examples.gridded_psf()
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
@pytest.mark.parametrize(('example', 'converter_name', 'method'), [
    # GriddedPSFModel is an astropy model, so its converter builds the
    # tree in to_yaml_tree_transform; STDPSFGrid is not a model, so its
    # converter uses to_yaml_tree.
    (examples.gridded_psf, 'GriddedPSFModelConverter',
     'to_yaml_tree_transform'),
    (examples.stdpsf_single_detector, 'STDPSFGridConverter', 'to_yaml_tree'),
], ids=['gridded_psf_model', 'stdpsf_grid'])
def test_grid_structure_is_stored_outside_meta(example, converter_name,
                                               method):
    """
    Test that both grid converters store the grid structure as
    top-level properties rather than inside ``meta``.

    GriddedPSFModel keeps the grid structure in its meta attribute
    and STDPSFGrid keeps it in private attributes, so the converters
    normalize it to a single layout in the file.
    """
    from photutils.converters import image_models

    obj, _ = example()
    converter = getattr(image_models, converter_name)()
    node = getattr(converter, method)(obj, None, None)

    for key in ('grid_xypos', 'oversampling'):
        assert key in node
        assert key not in node['meta']
    assert 'grid_shape' not in node['meta']


@pytest.mark.skipif(not _ASDF_ASTROPY_INSTALLED,
                    reason='asdf-astropy is not installed')
def test_stdpsf_grid_converter_preserves_oversampling(tmp_path):
    """
    Test that a non-default oversampling value survives a round trip.
    """
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
    filename = examples.PSF_DATA_DIR / 'STDPSF_ACSWFC_F814W_mock.fits'
    psfgrid = STDPSFGrid(str(filename))
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
