# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the gridded_models module.
"""

import os.path as op
from itertools import product

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.nddata import NDData
from astropy.table import QTable
from numpy.testing import assert_allclose, assert_equal

from photutils.datasets import make_model_image
from photutils.psf import GriddedPSFModel, STDPSFGrid
from photutils.segmentation import SourceCatalog, detect_sources
from photutils.utils._optional_deps import HAS_MATPLOTLIB

# The first file has a single detector, the rest have multiple detectors
STDPSF_FILENAMES = ('STDPSF_NRCA1_F150W_mock.fits',
                    'STDPSF_ACSWFC_F814W_mock.fits',
                    'STDPSF_NRCSW_F150W_mock.fits',
                    'STDPSF_WFC3UV_F814W_mock.fits',
                    'STDPSF_WFPC2_F814W_mock.fits')

WEBBPSF_FILENAMES = ('nircam_nrca1_f200w_fovp101_samp4_npsf16_mock.fits',
                     'nircam_nrca1_f200w_fovp101_samp4_npsf4_mock.fits',
                     'nircam_nrca5_f444w_fovp101_samp4_npsf4_mock.fits',
                     'nircam_nrcb4_f150w_fovp101_samp4_npsf1_mock.fits')


def _reference_find_bounding_points(model, x, y):
    """
    Reference implementation of the bounding-point lookup using the
    pre-fast-path ``numpy.searchsorted``/``numpy.where`` algorithm.

    This is used to verify that the optimized ``_find_bounding_points``
    and ``_bounding_lookup`` produce equivalent results.
    """
    xidx = np.searchsorted(model._xgrid, x) - 1
    yidx = np.searchsorted(model._ygrid, y) - 1
    xidx = np.clip(xidx, 0, len(model._xgrid) - 2)
    yidx = np.clip(yidx, 0, len(model._ygrid) - 2)

    x0, x1 = model._xgrid[xidx], model._xgrid[xidx + 1]
    y0, y1 = model._ygrid[yidx], model._ygrid[yidx + 1]

    xcoords, ycoords = model.grid_xypos.T
    lower_left = np.where((xcoords == x0) & (ycoords == y0))[0][0]
    lower_right = np.where((xcoords == x1) & (ycoords == y0))[0][0]
    upper_left = np.where((xcoords == x0) & (ycoords == y1))[0][0]
    upper_right = np.where((xcoords == x1) & (ycoords == y1))[0][0]

    grid_idx = np.array((lower_left, lower_right, upper_left, upper_right))
    grid_xy = np.array((x0, x1, y0, y1))
    return grid_idx, grid_xy


def _reference_bilinear_weights(xi, yi, grid_xy):
    """
    Reference implementation of the bilinear weights using the
    pre-fast-path ``numpy.clip`` algorithm.
    """
    x0, x1, y0, y1 = grid_xy
    xi = np.clip(xi, x0, x1)
    yi = np.clip(yi, y0, y1)
    norm = (x1 - x0) * (y1 - y0)
    return np.array([(x1 - xi) * (y1 - yi), (xi - x0) * (y1 - yi),
                     (x1 - xi) * (yi - y0), (xi - x0) * (yi - y0)]) / norm


def _reference_calc_model_values(model, x_0, y_0, xi, yi):
    """
    Reference implementation of ``_calc_model_values`` using the
    pre-fast-path bounding-point and weight algorithms.
    """
    grid_idx, grid_xy = _reference_find_bounding_points(model, x_0, y_0)
    interpolators = np.array([model._calc_interpolator(gidx)
                              for gidx in grid_idx])
    weights = _reference_bilinear_weights(x_0, y_0, grid_xy)

    idx = np.where(weights != 0)
    interpolators = interpolators[idx]
    weights = weights[idx]

    result = 0
    for interp, weight in zip(interpolators, weights, strict=True):
        result += interp(xi, yi, grid=False) * weight
    return result


@pytest.fixture(name='psfmodel')
def fixture_griddedpsf_data():
    psfs = []
    yy, xx = np.mgrid[0:101, 0:101]
    for i in range(16):
        theta = np.deg2rad(i * 10.0)
        gmodel = Gaussian2D(1, 50, 50, 10, 5, theta=theta)
        psfs.append(gmodel(xx, yy))

    xgrid = [0, 40, 160, 200]
    ygrid = [0, 60, 140, 200]
    meta = {}
    meta['grid_xypos'] = list(product(xgrid, ygrid))
    meta['oversampling'] = 4

    nddata = NDData(psfs, meta=meta)
    return GriddedPSFModel(nddata)


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

        # Check that data and grid_xypos attributes are read-only
        match = 'object has no setter'
        with pytest.raises(AttributeError, match=match):
            psfmodel.data = np.ones((4, 5, 5))
        with pytest.raises(AttributeError, match=match):
            psfmodel.grid_xypos = [[0, 0], [1, 1]]

    def test_repr_str(self, psfmodel):
        repr_str = repr(psfmodel)
        assert 'GriddedPSFModel' in repr_str
        assert 'flux=1.' in repr_str
        assert 'x_0=0.' in repr_str
        assert 'y_0=0.' in repr_str
        assert 'oversampling=' in repr_str
        assert 'fill_value=0.0' in repr_str

        str_str = str(psfmodel)
        assert 'GriddedPSFModel' in str_str
        assert 'Number of PSFs: 16' in str_str
        assert 'PSF shape (oversampled pixels): (101, 101)' in str_str
        assert 'Oversampling: [4, 4]' in str_str
        assert 'Fill Value: 0.0' in str_str

    def test_gridded_psf_model_basic_eval(self, psfmodel):
        assert psfmodel(0, 0) == 1
        assert psfmodel(100, 100) == 0
        assert_allclose(psfmodel([0, 100], [0, 100]), [1, 0])

        y, x = np.mgrid[0:100, 0:100]
        psf = psfmodel.evaluate(x=x, y=y, flux=100, x_0=40, y_0=60)
        assert psf.shape == (100, 100)

        _, y2, x2 = np.mgrid[0:100, 0:100, 0:100]
        match = 'x and y must be 1D or 2D'
        with pytest.raises(ValueError, match=match):
            psfmodel.evaluate(x=x2, y=y2, flux=100, x_0=40, y_0=60)

    def test_gridded_psf_model_single_psf(self, psfmodel):
        psfmodel = psfmodel.copy()
        psfmodel._data = psfmodel.data[0:1, :, :]
        assert psfmodel(0, 0) == 1
        assert psfmodel(100, 100) == 0
        assert_allclose(psfmodel([0, 100], [0, 100]), [1, 0])

        y, x = np.mgrid[0:100, 0:100]
        psf = psfmodel.evaluate(x=x, y=y, flux=100, x_0=40, y_0=60)
        assert psf.shape == (100, 100)

        _, y2, x2 = np.mgrid[0:100, 0:100, 0:100]
        match = 'x and y must be 1D or 2D'
        with pytest.raises(ValueError, match=match):
            psfmodel.evaluate(x=x2, y=y2, flux=100, x_0=40, y_0=60)

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

    def test_scalar_fastpath_caches(self, psfmodel):
        """
        The scalar fast-path lookup caches should have the expected
        types and shapes.
        """
        nx = len(psfmodel._xgrid)
        ny = len(psfmodel._ygrid)

        assert isinstance(psfmodel._xgrid_list, list)
        assert isinstance(psfmodel._ygrid_list, list)
        assert all(isinstance(val, float) for val in psfmodel._xgrid_list)
        assert all(isinstance(val, float) for val in psfmodel._ygrid_list)
        assert_allclose(psfmodel._xgrid_list, psfmodel._xgrid)
        assert_allclose(psfmodel._ygrid_list, psfmodel._ygrid)

        lookup = psfmodel._bounding_lookup
        assert lookup.shape == (nx - 1, ny - 1, 4)
        assert lookup.dtype == np.int64

    def test_bounding_lookup_table(self, psfmodel):
        """
        Each entry of the precomputed lookup table should map a grid
        cell to the source indices of its four bounding ePSFs.
        """
        xcoords, ycoords = psfmodel.grid_xypos.T
        lookup = psfmodel._bounding_lookup
        for ix in range(len(psfmodel._xgrid) - 1):
            for iy in range(len(psfmodel._ygrid) - 1):
                x0 = psfmodel._xgrid[ix]
                x1 = psfmodel._xgrid[ix + 1]
                y0 = psfmodel._ygrid[iy]
                y1 = psfmodel._ygrid[iy + 1]
                expected = [
                    np.where((xcoords == x0) & (ycoords == y0))[0][0],
                    np.where((xcoords == x1) & (ycoords == y0))[0][0],
                    np.where((xcoords == x0) & (ycoords == y1))[0][0],
                    np.where((xcoords == x1) & (ycoords == y1))[0][0]]
                assert_equal(lookup[ix, iy], expected)

    def test_find_bounding_points_interior(self, psfmodel):
        """
        For interior points (not on a grid line), the optimized lookup
        should match the reference algorithm exactly.
        """
        for x_0, y_0 in ((20, 30), (100, 100), (180, 170), (45.5, 61.5)):
            grid_idx, grid_xy = psfmodel._find_bounding_points(x_0, y_0)
            ref_idx, ref_xy = _reference_find_bounding_points(
                psfmodel, x_0, y_0)
            assert_equal(grid_idx, ref_idx)
            assert_allclose(grid_xy, ref_xy)

    def test_find_bounding_points_out_of_bounds(self, psfmodel):
        """
        Out-of-grid points should clamp to the nearest grid cell.
        """
        # Below the grid -> first cell
        _, grid_xy = psfmodel._find_bounding_points(-10, -20)
        assert_allclose(grid_xy, (psfmodel._xgrid[0], psfmodel._xgrid[1],
                                  psfmodel._ygrid[0], psfmodel._ygrid[1]))

        # Above the grid -> last cell
        _, grid_xy = psfmodel._find_bounding_points(500, 500)
        assert_allclose(grid_xy, (psfmodel._xgrid[-2], psfmodel._xgrid[-1],
                                  psfmodel._ygrid[-2], psfmodel._ygrid[-1]))

    def test_bilinear_weights(self, psfmodel):
        """
        Bilinear weights should sum to one, be non-negative, and clamp
        out-of-cell coordinates to the cell bounds.
        """
        grid_xy = np.array((0.0, 40.0, 0.0, 60.0))

        # Interior point
        weights = psfmodel._calc_bilinear_weights(10.0, 15.0, grid_xy)
        assert_allclose(weights.sum(), 1.0)
        assert np.all(weights >= 0)

        # Exact lower-left corner -> one-hot on the lower-left point
        weights = psfmodel._calc_bilinear_weights(0.0, 0.0, grid_xy)
        assert_allclose(weights, (1.0, 0.0, 0.0, 0.0))

        # Exact upper-right corner -> one-hot on the upper-right point
        weights = psfmodel._calc_bilinear_weights(40.0, 60.0, grid_xy)
        assert_allclose(weights, (0.0, 0.0, 0.0, 1.0))

        # Out-of-cell coordinates are clamped to the cell bounds
        clamped = psfmodel._calc_bilinear_weights(-5.0, 80.0, grid_xy)
        edge = psfmodel._calc_bilinear_weights(0.0, 60.0, grid_xy)
        assert_allclose(clamped, edge)

    def test_evaluate_matches_reference_algorithm(self, psfmodel):
        """
        The optimized evaluation must produce the same result as the
        pre-fast-path reference algorithm for interior, on-grid, and
        out-of-bounds positions.
        """
        y, x = np.mgrid[0:50, 0:50]
        positions = ((20, 30), (100, 100), (180, 170), (45.5, 61.5),
                     (40, 60), (160, 140), (0, 0), (200, 200),
                     (-10, -20), (500, 500))
        for x_0, y_0 in positions:
            xi = psfmodel.oversampling[1] * (x.astype(float) - x_0)
            yi = psfmodel.oversampling[0] * (y.astype(float) - y_0)
            xi += psfmodel.origin[0]
            yi += psfmodel.origin[1]
            result = psfmodel._calc_model_values(x_0, y_0, xi, yi)
            expected = _reference_calc_model_values(psfmodel, x_0, y_0, xi, yi)
            assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_gridded_psf_model_invalid_inputs(self):
        data = np.ones((4, 5, 5))

        # Check if NDData
        match = 'data must be an NDData instance'
        with pytest.raises(TypeError, match=match):
            GriddedPSFModel(data)

        # Check PSF data dimension
        match = 'The NDData data attribute must be a 3D numpy ndarray'
        with pytest.raises(ValueError, match=match):
            GriddedPSFModel(NDData(np.ones((3, 3))))

        match = 'The length of the PSF x and y axes must both be at least 4'
        with pytest.raises(ValueError, match=match):
            GriddedPSFModel(NDData(np.ones((4, 3, 3))))

        match = 'The number of ePSFs must not be 2 or 3'
        meta = {'grid_xypos': [[0, 0], [1, 0], [1, 0]], 'oversampling': 4}
        nddata = NDData(np.ones((3, 4, 4)), meta=meta)
        with pytest.raises(ValueError, match=match):
            GriddedPSFModel(nddata)

        match = 'All elements of input data must be finite'
        data2 = np.ones((4, 5, 5))
        data2[0, 2, 2] = np.nan
        with pytest.raises(ValueError, match=match):
            GriddedPSFModel(NDData(data2))

        # Check that grid_xypos is in meta
        meta = {'oversampling': 4}
        nddata = NDData(data, meta=meta)
        match = "'grid_xypos' must be in the nddata meta dictionary"
        with pytest.raises(ValueError, match=match):
            GriddedPSFModel(nddata)

        # Check grid_xypos length
        meta = {'grid_xypos': [[0, 0], [1, 0], [1, 0]], 'oversampling': 4}
        nddata = NDData(data, meta=meta)
        match = 'length of grid_xypos must match the number of input ePSFs'
        with pytest.raises(ValueError, match=match):
            GriddedPSFModel(nddata)

        # Check if grid_xypos is a regular grid
        meta = {'grid_xypos': [[0, 0], [1, 0], [1, 0], [3, 4]],
                'oversampling': 4}
        nddata = NDData(data, meta=meta)
        match = 'grid_xypos must form a rectangular grid'
        with pytest.raises(ValueError, match=match):
            GriddedPSFModel(nddata)

        meta = {'grid_xypos': [[0, 0], [0, 2], [0, 4], [0, 6]],
                'oversampling': 4}
        nddata = NDData(data, meta=meta)
        match = 'grid_xypos must form a rectangular grid'
        with pytest.raises(ValueError, match=match):
            GriddedPSFModel(nddata)

        # Check that oversampling is in meta
        meta = {'grid_xypos': [[0, 0], [0, 1], [1, 0], [1, 1]]}
        nddata = NDData(data, meta=meta)
        match = "'oversampling' must be in the nddata meta dictionary"
        with pytest.raises(ValueError, match=match):
            GriddedPSFModel(nddata)

    def test_gridded_psf_model_eval(self, psfmodel):
        """
        Create a simulated image using GriddedPSFModel and test the
        properties of the generated sources.
        """
        shape = (200, 200)
        params = QTable()
        params['x_0'] = [40, 50, 160, 160]
        params['y_0'] = [60, 150, 50, 140]
        params['flux'] = [100, 100, 100, 100]
        data = make_model_image(shape, psfmodel, params)

        segm = detect_sources(data, 0.0, 5)
        cat = SourceCatalog(data, segm)
        orients = cat.orientation.value
        assert_allclose(orients[1], 50.0, rtol=1.0e-5)
        assert_allclose(orients[2], 280.0, rtol=1.0e-5)
        assert 88.3 < orients[0] < 88.4
        assert 64.0 < orients[3] < 64.2

    @pytest.mark.parametrize('deepcopy', [False, True])
    def test_copy(self, psfmodel, deepcopy):
        flux = psfmodel.flux.value
        model_copy = psfmodel.deepcopy() if deepcopy else psfmodel.copy()

        assert_equal(model_copy.data, psfmodel.data)
        assert_equal(model_copy.grid_xypos, psfmodel.grid_xypos)
        assert_equal(model_copy.oversampling, psfmodel.oversampling)
        assert_equal(model_copy.meta, psfmodel.meta)
        assert model_copy.flux.value == psfmodel.flux.value
        assert model_copy.x_0.value == psfmodel.x_0.value
        assert model_copy.y_0.value == psfmodel.y_0.value
        assert model_copy.fixed == psfmodel.fixed

        model_copy.data[0, 0, 0] = 42
        if deepcopy:
            assert model_copy.data[0, 0, 0] != psfmodel.data[0, 0, 0]
        else:
            assert model_copy.data[0, 0, 0] == psfmodel.data[0, 0, 0]

        model_copy.flux = 100
        assert model_copy.flux.value != flux

        model_copy.x_0.fixed = True
        model_copy.y_0.fixed = True
        new_model = model_copy.copy()
        assert new_model.x_0.fixed
        assert new_model.fixed == model_copy.fixed

    def test_repr(self, psfmodel):
        model_repr = repr(psfmodel)
        assert '<GriddedPSFModel(' in model_repr
        for param in psfmodel.param_names:
            assert param in model_repr

    def test_str(self, psfmodel):
        model_str = str(psfmodel)
        keys = ('Grid shape', 'Number of PSFs', 'PSF shape', 'Oversampling')
        for key in keys:
            assert key in model_str
        for param in psfmodel.param_names:
            assert param in model_str

    def test_gridded_psf_oversampling(self, psfmodel):
        nddata = NDData(psfmodel.data, meta=psfmodel.meta)
        nddata.meta['oversampling'] = [4, 4]
        psfmodel2 = GriddedPSFModel(nddata)
        assert_equal(psfmodel2.oversampling, psfmodel.oversampling)

    def test_bounding_box(self, psfmodel):
        # oversampling is 4
        bbox = psfmodel.bounding_box.bounding_box()
        assert_equal(bbox, ((-12.625, 12.625), (-12.625, 12.625)))

        model = psfmodel.copy()
        model.oversampling = 1
        bbox = model.bounding_box.bounding_box()
        assert_equal(bbox, ((-50.5, 50.5), (-50.5, 50.5)))

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

    @pytest.mark.parametrize('filename', STDPSF_FILENAMES[1:])
    @pytest.mark.parametrize('detector_id', [1, 2])
    def test_read_stdpsf_multi_detector(self, filename, detector_id):
        """
        Test STDPSF read for multiple detectors.
        """
        filename = op.join(op.dirname(op.abspath(__file__)), 'data', filename)
        psfmodel = GriddedPSFModel.read(filename, detector_id=detector_id,
                                        format='stdpsf')
        assert psfmodel.data.shape[0] == len(psfmodel.meta['grid_xypos'])
        assert_equal(psfmodel.oversampling, [4, 4])
        assert_equal(psfmodel.meta['oversampling'], psfmodel.oversampling)

        # Test format auto-detect
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

        # Test format auto-detect
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
    keys = ('STDPSF', 'Grid_shape', 'Number of PSFs', 'PSF shape',
            'Oversampling')
    for key in keys:
        assert key in repr(psfgrid)
