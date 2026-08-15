# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the interpolators module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.utils.exceptions import AstropyDeprecationWarning
from numpy.testing import assert_allclose

from photutils.background.background_2d import Background2D
from photutils.background.interpolators import (BkgIDWInterpolator,
                                                _BkgZoomInterpolator)


@pytest.fixture
def test_data():
    """
    Create test data for interpolator tests.
    """
    return np.ones((300, 300))


@pytest.fixture
def test_mesh():
    """
    Create test mesh for interpolator tests.
    """
    return np.array([[0.01, 0.01, 0.02],
                     [0.01, 0.02, 0.03],
                     [0.03, 0.03, 12.9]])


def test_zoom_interp_constant_mesh(test_data):
    """
    Test the zoom interpolator with a constant-valued mesh.

    When all mesh values are equal, the interpolator takes an early-exit
    path that fills the output with the constant value directly,
    bypassing `scipy.ndimage.zoom` entirely. This path must produce the
    correct fill value both for plain arrays and for Quantity inputs.
    """
    bkg = Background2D(test_data, 100)
    interp = _BkgZoomInterpolator()

    constant_mesh = np.full((3, 3), 7.5)
    result = interp(constant_mesh, **bkg._interp_kwargs)
    assert result.shape == bkg._interp_kwargs['shape']
    assert np.all(result == 7.5)

    # Also verify with a Quantity mesh
    unit = u.nJy
    bkg_q = Background2D(test_data << unit, 100)
    result_q = interp(constant_mesh << unit, **bkg_q._interp_kwargs)
    assert result_q.shape == bkg_q._interp_kwargs['shape']
    assert np.all(result_q == 7.5)


def test_zoom_interp(test_data, test_mesh):
    """
    Test the zoom interpolator.
    """
    bkg = Background2D(test_data, 100)

    interp = _BkgZoomInterpolator(clip=False)
    zoom = interp(test_mesh, **bkg._interp_kwargs)
    assert zoom.shape == (300, 300)

    # Test with units
    unit = u.nJy
    bkg = Background2D(test_data << unit, 100)
    interp = _BkgZoomInterpolator(clip=False)
    zoom = interp(test_mesh << unit, **bkg._interp_kwargs)
    assert zoom.shape == (300, 300)

    # Test repr
    cls_repr = repr(interp)
    assert cls_repr.startswith(f'{interp.__class__.__name__}')


def test_zoom_interp_clip(test_data, test_mesh):
    """
    Test the zoom interpolator with clipping.
    """
    bkg = Background2D(test_data, 100)

    interp1 = _BkgZoomInterpolator(clip=False)
    zoom1 = interp1(test_mesh, **bkg._interp_kwargs)

    interp2 = _BkgZoomInterpolator(clip=True)
    zoom2 = interp2(test_mesh, **bkg._interp_kwargs)

    minval = np.min(test_mesh)
    maxval = np.max(test_mesh)
    assert np.min(zoom1) < minval
    assert np.max(zoom1) > maxval
    assert np.min(zoom2) == minval
    assert np.max(zoom2) == maxval


def test_idw_interp(test_data, test_mesh):
    """
    Test the IDW interpolator.
    """
    with pytest.warns(AstropyDeprecationWarning):
        interp = BkgIDWInterpolator()
    with pytest.warns(AstropyDeprecationWarning):
        bkg = Background2D(test_data, 100, interpolator=interp)

    zoom = interp(test_mesh, **bkg._interp_kwargs)
    assert zoom.shape == (300, 300)

    # Test constant mesh data
    zoom = interp(np.ones_like(test_mesh), **bkg._interp_kwargs)
    assert np.all(zoom == 1)

    # Test with units
    unit = u.nJy
    with pytest.warns(AstropyDeprecationWarning):
        bkg = Background2D(test_data << unit, 100, interpolator=interp)
    zoom = interp(test_mesh << unit, **bkg._interp_kwargs)
    assert zoom.shape == (300, 300)

    # Test repr
    cls_repr = repr(interp)
    assert cls_repr.startswith(f'{interp.__class__.__name__}')


class TestThreadedZoom:
    """
    Tests for the multithreaded row-band zoom in _BkgZoomInterpolator.
    """

    @staticmethod
    def _make_kwargs(mesh_shape, box_size, *, crop=(0, 0), n_threads=1,
                     dtype=np.float64):
        shape = (mesh_shape[0] * box_size[0] - crop[0],
                 mesh_shape[1] * box_size[1] - crop[1])
        return {'shape': shape, 'dtype': dtype,
                'box_size': np.array(box_size), 'n_threads': n_threads}

    @pytest.mark.parametrize('order', [0, 1, 3, 5])
    @pytest.mark.parametrize('mode', ['reflect', 'mirror'])
    def test_threaded_matches_serial(self, order, mode):
        """
        Test that the multithreaded zoom matches the serial scipy zoom
        up to floating-point rounding, including with uneven bands,
        non-square meshes and boxes, and output cropping.
        """
        rng = np.random.default_rng(5)
        mesh = rng.normal(10.0, 3.0, (17, 11))
        interp = _BkgZoomInterpolator(order=order, mode=mode)

        kwargs = self._make_kwargs((17, 11), (13, 7), crop=(4, 3))
        ref = interp(mesh, **kwargs)
        for n_threads in (2, 5, 16):
            kwargs['n_threads'] = n_threads
            result = interp(mesh, **kwargs)
            assert result.shape == kwargs['shape']
            assert_allclose(result, ref, rtol=1e-10)

    def test_threaded_float32(self):
        """
        Test the multithreaded zoom with float32 mesh data.
        """
        rng = np.random.default_rng(5)
        mesh = rng.normal(10.0, 3.0, (17, 11)).astype(np.float32)
        interp = _BkgZoomInterpolator()
        kwargs = self._make_kwargs((17, 11), (13, 7), crop=(4, 3),
                                   dtype=np.float32)
        ref = interp(mesh, **kwargs)
        kwargs['n_threads'] = 8
        result = interp(mesh, **kwargs)
        assert result.dtype == np.float32
        assert_allclose(result, ref, rtol=1e-5)

    def test_unsupported_mode_falls_back(self):
        """
        Test that boundary modes with grid_mode-specific edge handling
        fall back to the serial scipy zoom (identical results).
        """
        rng = np.random.default_rng(5)
        mesh = rng.normal(10.0, 3.0, (9, 9))
        interp = _BkgZoomInterpolator(mode='nearest')
        kwargs = self._make_kwargs((9, 9), (10, 10))
        ref = interp(mesh, **kwargs)
        kwargs['n_threads'] = 8
        result = interp(mesh, **kwargs)
        assert_allclose(result, ref, rtol=0, atol=0)

    def test_more_threads_than_rows(self):
        """
        Test that n_threads larger than the number of output rows is
        clamped to the number of rows.
        """
        rng = np.random.default_rng(5)
        mesh = rng.normal(10.0, 3.0, (2, 3))
        interp = _BkgZoomInterpolator()
        kwargs = self._make_kwargs((2, 3), (2, 2))
        ref = interp(mesh, **kwargs)
        kwargs['n_threads'] = 64  # only 4 output rows
        result = interp(mesh, **kwargs)
        assert_allclose(result, ref, rtol=1e-10)
