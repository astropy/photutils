# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the gridded PSF model module.
"""

import os.path as op
from itertools import product

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.nddata import NDData
from numpy.testing import assert_allclose, assert_equal

from photutils.psf import GriddedPSFModel, STDPSFGrid
from photutils.segmentation import SourceCatalog, detect_sources
from photutils.utils._optional_deps import HAS_MATPLOTLIB, HAS_SCIPY

# the first file has a single detector, the rest have multiple detectors
STDPSF_FILENAMES = ('STDPSF_NRCA1_F150W_mock.fits',
                    'STDPSF_ACSWFC_F814W_mock.fits',
                    'STDPSF_NRCSW_F150W_mock.fits',
                    'STDPSF_WFC3UV_F814W_mock.fits',
                    'STDPSF_WFPC2_F814W_mock.fits')

WEBBPSF_FILENAMES = ('nircam_nrca1_f200w_fovp101_samp4_npsf16_mock.fits',
                     'nircam_nrca1_f200w_fovp101_samp4_npsf4_mock.fits',
                     'nircam_nrca5_f444w_fovp101_samp4_npsf4_mock.fits',
                     'nircam_nrcb4_f150w_fovp101_samp4_npsf1_mock.fits')


@pytest.fixture(name='psfmodel')
def fixture_griddedpsf_data():
    psfs = []
    y, x = np.mgrid[0:101, 0:101]
    for i in range(16):
        theta = i * 10.0 * np.pi / 180.0
        g = Gaussian2D(1, 50, 50, 10, 5, theta=theta)
        m = g(x, y)
        psfs.append(m)

    xgrid = [0, 40, 160, 200]
    ygrid = [0, 60, 140, 200]
    grid_xypos = list(product(xgrid, ygrid))

    meta = {}
    meta['grid_xypos'] = grid_xypos
    meta['oversampling'] = 4

    nddata = NDData(psfs, meta=meta)
    psfmodel = GriddedPSFModel(nddata)

    return psfmodel


class TestGriddedPSFModel:
    """
    Tests for GriddPSFModel.
    """

    def test_gridded_psf_model(self, psfmodel):
        keys = ['grid_xypos', 'oversampling']
        for key in keys:
            assert key in psfmodel.meta
        grid_xypos = psfmodel.grid_xypos
        assert len(grid_xypos) == 16
        assert_equal(psfmodel.oversampling, [4, 4])
        assert_equal(psfmodel.meta['oversampling'], psfmodel.oversampling)
        assert psfmodel.data.shape == (16, 101, 101)

        idx = np.lexsort((grid_xypos[:, 0], grid_xypos[:, 1]))
        xypos = grid_xypos[idx]
        assert_allclose(xypos, grid_xypos)

    @pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
    def test_gridded_psf_model_basic_eval(self, psfmodel):
        y, x = np.mgrid[0:100, 0:100]
        psf = psfmodel.evaluate(x=x, y=y, flux=100, x_0=40, y_0=60)
        assert psf.shape == (100, 100)

        z2, y2, x2 = np.mgrid[0:100, 0:100, 0:100]
        match = 'x and y must be 1D or 2D'
        with pytest.raises(ValueError, match=match):
            psfmodel.evaluate(x=x2, y=y2, flux=100, x_0=40, y_0=60)

    @pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
    def test_gridded_psf_model_eval_outside_grid(self, psfmodel):
        y, x = np.mgrid[-50:50, -50:50]
        psf1 = psfmodel.evaluate(x=x, y=y, flux=100, x_0=0, y_0=0)
        y, x = np.mgrid[-60:40, -60:40]
        psf2 = psfmodel.evaluate(x=x, y=y, flux=100, x_0=-10, y_0=-10)
        assert_allclose(psf1, psf2)

        y, x = np.mgrid[150:250, 150:250]
        psf3 = psfmodel.evaluate(x=x, y=y, flux=100, x_0=200, y_0=200)
        y, x = np.mgrid[170:270, 170:270]
        psf4 = psfmodel.evaluate(x=x, y=y, flux=100, x_0=220, y_0=220)
        assert_allclose(psf3, psf4)

    @pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
    def test_gridded_psf_model_interp(self, psfmodel):
        # test xyref length
        with pytest.raises(TypeError):
            psfmodel._bilinear_interp([1, 1], 1, 1, 1)

        # test if refxy points form a rectangle
        with pytest.raises(ValueError):
            xyref = [[0, 0], [0, 1], [1, 0], [2, 2]]
            zref = np.ones((4, 4, 4))
            psfmodel._bilinear_interp(xyref, zref, 1, 1)

        # test if xi and yi are outside of xyref
        xyref = [[0, 0], [0, 1], [1, 0], [1, 1]]
        zref = np.ones((4, 4, 4))
        with pytest.raises(ValueError):
            psfmodel._bilinear_interp(xyref, zref, 100, 1)
        with pytest.raises(ValueError):
            psfmodel._bilinear_interp(xyref, zref, 1, 100)

        # test non-scalar xi and yi
        idx = [0, 1, 4, 5]
        xyref = np.array(psfmodel.grid_xypos)[idx]
        psfs = psfmodel.data[idx, :, :]
        val1 = psfmodel._bilinear_interp(xyref, psfs, 10, 20)
        val2 = psfmodel._bilinear_interp(xyref, psfs, [10], [20])
        assert_allclose(val1, val2)

    def test_gridded_psf_model_invalid_inputs(self):
        data = np.ones((4, 3, 3))

        # check if NDData
        with pytest.raises(TypeError):
            GriddedPSFModel(data)

        # check PSF data dimension
        with pytest.raises(ValueError):
            GriddedPSFModel(NDData(np.ones((3, 3))))

        # check that grid_xypos is in meta
        meta = {'oversampling': 4}
        nddata = NDData(data, meta=meta)
        with pytest.raises(ValueError):
            GriddedPSFModel(nddata)

        # check grid_xypos length
        meta = {'grid_xypos': [[0, 0], [1, 0], [1, 0]], 'oversampling': 4}
        nddata = NDData(data, meta=meta)
        with pytest.raises(ValueError):
            GriddedPSFModel(nddata)

        # check if grid_xypos is a regular grid
        meta = {'grid_xypos': [[0, 0], [1, 0], [1, 0], [3, 4]],
                'oversampling': 4}
        nddata = NDData(data, meta=meta)
        with pytest.raises(ValueError):
            GriddedPSFModel(nddata)

        # check that oversampling is in meta
        meta = {'grid_xypos': [[0, 0], [0, 1], [1, 0], [1, 1]]}
        nddata = NDData(data, meta=meta)
        with pytest.raises(ValueError):
            GriddedPSFModel(nddata)

    @pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
    def test_gridded_psf_model_eval(self, psfmodel):
        """
        Create a simulated image using GriddedPSFModel and test
        the properties of the generated sources.
        """
        shape = (200, 200)
        data = np.zeros(shape)
        eval_xshape = (np.ceil(psfmodel.data.shape[2]
                               / psfmodel.oversampling[1])).astype(int)
        eval_yshape = (np.ceil(psfmodel.data.shape[1]
                               / psfmodel.oversampling[0])).astype(int)

        xx = [40, 50, 160, 160]
        yy = [60, 150, 50, 140]
        zz = [100, 100, 100, 100]
        for xxi, yyi, zzi in zip(xx, yy, zz):
            x0 = np.floor(xxi - (eval_xshape - 1) / 2.0).astype(int)
            y0 = np.floor(yyi - (eval_yshape - 1) / 2.0).astype(int)
            x1 = x0 + eval_xshape
            y1 = y0 + eval_yshape

            x0 = max(x0, 0)
            y0 = max(y0, 0)
            x1 = min(x1, shape[1])
            y1 = min(y1, shape[0])

            y, x = np.mgrid[y0:y1, x0:x1]
            data[y, x] += psfmodel.evaluate(x=x, y=y, flux=zzi, x_0=xxi,
                                            y_0=yyi)

        segm = detect_sources(data, 0.0, 5)
        cat = SourceCatalog(data, segm)
        orients = cat.orientation.value
        assert_allclose(orients[1], 50.0, rtol=1.0e-5)
        assert_allclose(orients[2], -80.0, rtol=1.0e-5)
        assert 88.3 < orients[0] < 88.4
        assert 64.0 < orients[3] < 64.2

    def test_copy(self, psfmodel):
        flux = psfmodel.flux.value
        new_model = psfmodel.copy()

        assert_equal(new_model.data, psfmodel.data)
        assert_equal(new_model.grid_xypos, psfmodel.grid_xypos)

        new_model.flux = 100
        assert new_model.flux.value != flux

        new_model.x_0.fixed = True
        new_model.y_0.fixed = True
        new_model2 = new_model.copy()
        assert new_model2.x_0.fixed
        assert new_model.fixed == new_model2.fixed

    def test_deepcopy(self, psfmodel):
        flux = psfmodel.flux.value
        new_model = psfmodel.deepcopy()
        new_model.flux = 100
        assert new_model.flux.value != flux

    @pytest.mark.skipif(not HAS_SCIPY, reason='scipy is required')
    def test_cache(self, psfmodel):
        for x, y in psfmodel.grid_xypos:
            psfmodel.x_0 = x
            psfmodel.y_0 = y
            psfmodel(0, 0)
            psfmodel(1, 1)

        assert psfmodel._cache_info().hits == 16
        assert psfmodel._cache_info().misses == 16
        assert psfmodel._cache_info().currsize == 16

        psfmodel.clear_cache()
        assert psfmodel._cache_info().hits == 0
        assert psfmodel._cache_info().misses == 0
        assert psfmodel._cache_info().currsize == 0

    def test_repr(self, psfmodel):
        model_repr = repr(psfmodel)
        assert '<GriddedPSFModel(' in model_repr
        for param in psfmodel.param_names:
            assert param in model_repr

    def test_str(self, psfmodel):
        model_str = str(psfmodel)
        keys = ('Grid_shape', 'Number of ePSFs', 'ePSF shape', 'Oversampling')
        for key in keys:
            assert key in model_str
        for param in psfmodel.param_names:
            assert param in model_str

    def test_gridded_psf_oversampling(self, psfmodel):
        nddata = NDData(psfmodel.data, meta=psfmodel.meta)
        nddata.meta['oversampling'] = [4, 4]
        psfmodel2 = GriddedPSFModel(nddata)
        assert_equal(psfmodel2.oversampling, psfmodel.oversampling)

    def test_read_stdpsf(self):
        """
        Test STDPSF read for a single detector.
        """
        filename = 'STDPSF_NRCA1_F150W_mock.fits'
        filename = op.join(op.dirname(op.abspath(__file__)), 'data', filename)
        psfmodel = GriddedPSFModel.read(filename)
        assert psfmodel.data.shape[0] == len(psfmodel.meta['grid_xypos'])
        assert_equal(psfmodel.oversampling, [4, 4])
        assert_equal(psfmodel.meta['oversampling'], psfmodel.oversampling)

    @pytest.mark.parametrize(('filename', 'detector_id'),
                             list(product(STDPSF_FILENAMES[1:], (1, 2))))
    def test_read_stdpsf_multi_detector(self, filename, detector_id):
        """
        Test STDPSF read for a multiple detectors.
        """
        filename = op.join(op.dirname(op.abspath(__file__)), 'data', filename)
        psfmodel = GriddedPSFModel.read(filename, detector_id=detector_id,
                                        format='stdpsf')
        assert psfmodel.data.shape[0] == len(psfmodel.meta['grid_xypos'])
        assert_equal(psfmodel.oversampling, [4, 4])
        assert_equal(psfmodel.meta['oversampling'], psfmodel.oversampling)

        # test format auto-detect
        filename = op.join(op.dirname(op.abspath(__file__)), 'data', filename)
        psfmodel = GriddedPSFModel.read(filename, detector_id=detector_id)
        assert psfmodel.data.shape[0] == len(psfmodel.meta['grid_xypos'])

        match = 'detector_id must be specified'
        with pytest.raises(ValueError, match=match):
            GriddedPSFModel.read(filename, detector_id=None)

        match = 'detector_id must be '
        with pytest.raises(ValueError, match=match):
            GriddedPSFModel.read(filename, detector_id=-1)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    @pytest.mark.parametrize('filename', WEBBPSF_FILENAMES)
    def test_read_webbpsf(self, filename):
        filename = op.join(op.dirname(op.abspath(__file__)), 'data', filename)
        psfmodel = GriddedPSFModel.read(filename, format='webbpsf')
        assert psfmodel.data.shape[0] == len(psfmodel.meta['grid_xypos'])
        assert_equal(psfmodel.oversampling, [4, 4])
        assert_equal(psfmodel.meta['oversampling'], psfmodel.oversampling)
        psfmodel.plot_grid()

        # test format auto-detect
        filename = op.join(op.dirname(op.abspath(__file__)), 'data', filename)
        psfmodel = GriddedPSFModel.read(filename)
        assert psfmodel.data.shape[0] == len(psfmodel.meta['grid_xypos'])

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_plot(self, psfmodel):
        psfmodel.plot_grid()
        psfmodel.plot_grid(peak_norm=True, cmap='Blues', vmax_scale=0.9)
        psfmodel.plot_grid(deltas=True)
        psfmodel.plot_grid(deltas=True, peak_norm=True)

        # simulate a grid where one or more ePSFS are blank (all zeros)
        model = psfmodel.deepcopy()
        model.data[0] = 0.0
        model.plot_grid(deltas=True)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
@pytest.mark.parametrize('filename', STDPSF_FILENAMES)
def test_stdpsfgrid(filename):
    filename = op.join(op.dirname(op.abspath(__file__)), 'data', filename)
    psfgrid = STDPSFGrid(filename)
    assert 'grid_xypos' in psfgrid.meta
    assert 'oversampling' in psfgrid.meta
    assert_equal(psfgrid.oversampling, [4, 4])
    assert psfgrid.data.shape[0] == len(psfgrid.meta['grid_xypos'])

    psfgrid.plot_grid()


def test_stdpsfgrid_repr_str():
    filename = STDPSF_FILENAMES[0]
    filename = op.join(op.dirname(op.abspath(__file__)), 'data', filename)
    psfgrid = STDPSFGrid(filename)
    assert repr(psfgrid) == str(psfgrid)
    keys = ('STDPSF', 'Grid_shape', 'Number of ePSFs', 'ePSF shape',
            'Oversampling')
    for key in keys:
        assert key in repr(psfgrid)
