# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the local_background module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.stats import SigmaClip
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_equal

from photutils.aperture import CircularAnnulus
from photutils.background import (BiweightLocationBackground,
                                  BiweightScaleBackgroundRMS, LocalBackground,
                                  MADStdBackgroundRMS, MeanBackground,
                                  MedianBackground, MMMBackground,
                                  ModeEstimatorBackground,
                                  SExtractorBackground, StdBackgroundRMS)


def test_local_background_invalid_radii():
    """
    Test that LocalBackground raises errors for invalid radius values.
    """
    # Test negative inner radius
    match = 'inner_radius must be positive'
    with pytest.raises(ValueError, match=match):
        LocalBackground(-5, 10)

    # Test zero inner radius
    with pytest.raises(ValueError, match=match):
        LocalBackground(0, 10)

    # Test negative outer radius
    match = 'outer_radius must be positive'
    with pytest.raises(ValueError, match=match):
        LocalBackground(5, -10)

    # Test zero outer radius
    with pytest.raises(ValueError, match=match):
        LocalBackground(5, 0)

    # Test outer_radius <= inner_radius
    match = 'outer_radius must be greater than inner_radius'
    with pytest.raises(ValueError, match=match):
        LocalBackground(10, 5)

    # Test equal radii
    with pytest.raises(ValueError, match=match):
        LocalBackground(10, 10)


def test_local_background():
    """
    Test the basic functionality of LocalBackground with a simple
    constant data array.
    """
    data = np.ones((101, 101))
    local_bkg = LocalBackground(5, 10, bkg_estimator=MedianBackground())

    x = np.arange(1, 7) * 10
    y = np.arange(1, 7) * 10
    bkg = local_bkg(data, x, y)
    assert_allclose(bkg, np.ones(len(x)))

    # Test scalar x and y
    bkg2 = local_bkg(data, x[2], y[2])
    assert not isinstance(bkg2, np.ndarray)
    assert_allclose(bkg[2], bkg2)

    bkg3 = local_bkg(data, -100, -100)
    assert np.isnan(bkg3)

    match = "'positions' must not contain any non-finite"
    with pytest.raises(ValueError, match=match):
        _ = local_bkg(data, x[2], np.inf)

    cls_repr = repr(local_bkg)
    assert cls_repr.startswith(local_bkg.__class__.__name__)

    # Test default bkg_estimator
    local_bkg2 = LocalBackground(5, 10, bkg_estimator=None)
    bkg4 = local_bkg2(data, x, y)
    assert_allclose(bkg4, bkg)


def test_local_background_units():
    """
    Test that Quantity input data returns a Quantity with the same unit.
    """
    data = np.ones((101, 101))
    local_bkg = LocalBackground(5, 10)

    bkg = local_bkg(data << u.Jy, 50, 50)
    assert isinstance(bkg, u.Quantity)
    assert bkg.unit == u.Jy
    assert_allclose(bkg.value, local_bkg(data, 50, 50))

    x = [30, 50, 70]
    y = [30, 50, 70]
    bkg2 = local_bkg(data << u.Jy, x, y)
    assert isinstance(bkg2, u.Quantity)
    assert bkg2.unit == u.Jy
    assert_allclose(bkg2.value, local_bkg(data, x, y))


def test_local_background_estimator_1d():
    """
    Test that the bkg_estimator can be a 1D function that takes an array
    and returns a scalar.
    """

    def estimator(data):
        assert data.ndim == 1
        return np.nanmedian(data)

    data = np.ones((51, 51))
    local_bkg = LocalBackground(3, 6, bkg_estimator=estimator)
    bkg = local_bkg(data, [10, 20], [10, 20])
    assert_allclose(bkg, np.ones(2))


def test_to_aperture_mismatched_shapes():
    """
    Test that an error is raised if x and y have different shapes.
    """
    local_bkg = LocalBackground(5, 10)
    match = 'x and y must have the same shape'
    with pytest.raises(ValueError, match=match):
        local_bkg.to_aperture([1, 2], [1, 2, 3])


def test_to_aperture_scalar():
    """
    Test to_aperture method with scalar x and y positions.
    """
    r_in = 5
    r_out = 10
    local_bkg = LocalBackground(r_in, r_out)

    # Test scalar positions
    x = 50.0
    y = 50.0
    aperture = local_bkg.to_aperture(x, y)

    # Check aperture type and properties
    assert isinstance(aperture, CircularAnnulus)
    assert_allclose(aperture.positions, [[x, y]])
    assert_allclose(aperture.r_in, r_in)
    assert_allclose(aperture.r_out, r_out)


def test_to_aperture_array():
    """
    Test to_aperture method with array x and y positions.
    """
    r_in = 7.5
    r_out = 15.2
    local_bkg = LocalBackground(r_in, r_out)

    # Test array positions
    x = np.array([10.0, 20.1, 35.3])
    y = np.array([14.4, 27.2, 33.4])
    xypos = list(zip(x, y, strict=False))
    aperture = local_bkg.to_aperture(x, y)

    # Check aperture type and properties
    assert isinstance(aperture, CircularAnnulus)
    assert_allclose(aperture.positions, xypos)
    assert_allclose(aperture.r_in, r_in)
    assert_allclose(aperture.r_out, r_out)

    # Test list positions
    x = list(x)
    y = list(y)
    aperture2 = local_bkg.to_aperture(x, y)
    assert aperture == aperture2


class TestFastLocalBackground:
    """
    Tests that the batched ApertureStats-based fast path gives the
    same results as the per-aperture estimator loop.
    """

    @classmethod
    def setup_class(cls):
        rng = np.random.default_rng(0)
        data = rng.normal(10.0, 2.0, (151, 151))
        data[40:60, 40:60] += 100.0  # outliers to clip
        mask = np.zeros(data.shape, dtype=bool)
        mask[100:130, 100:130] = True  # fully masks one annulus
        cls.data = data
        cls.mask = mask
        # Interior, clipped-outlier, near-mask, fully-masked,
        # edge-clipped, and off-image positions
        cls.x = np.array([30.0, 50.0, 75.0, 115.0, 3.0, -50.0])
        cls.y = np.array([30.0, 50.0, 75.0, 115.0, 3.0, -50.0])

    def _run_both_paths(self, local_bkg, monkeypatch, data=None):
        if data is None:
            data = self.data
        fast = local_bkg(data, self.x, self.y, mask=self.mask)
        assert local_bkg._fast_estimator_spec() is not None
        monkeypatch.setattr(LocalBackground, '_fast_estimator_spec',
                            lambda _self: None)
        slow = local_bkg(data, self.x, self.y, mask=self.mask)
        monkeypatch.undo()
        return fast, slow

    @pytest.mark.parametrize('estimator', [
        MeanBackground(), MedianBackground(),
        ModeEstimatorBackground(median_factor=2.5, mean_factor=1.5),
        MMMBackground(), SExtractorBackground(),
        BiweightLocationBackground(), StdBackgroundRMS(),
        MADStdBackgroundRMS(), BiweightScaleBackgroundRMS()])
    def test_matches_slow_path(self, estimator, monkeypatch):
        """
        Test that all supported estimator classes match the per-aperture
        loop, including for edge-clipped, fully-masked, and
        non-overlapping apertures (NaN).
        """
        local_bkg = LocalBackground(5, 10, bkg_estimator=estimator)
        fast, slow = self._run_both_paths(local_bkg, monkeypatch)
        assert np.isnan(fast[3])  # fully masked annulus
        assert np.isnan(fast[5])  # no overlap with the data
        assert_allclose(fast, slow, rtol=1e-12)

    @pytest.mark.parametrize('sigma_clip', [
        None,
        SigmaClip(sigma=2.0, maxiters=5),
        SigmaClip(sigma=3.0, maxiters=10, cenfunc='mean',
                  stdfunc='mad_std')])
    def test_sigma_clip_variants_match_slow_path(self, sigma_clip,
                                                 monkeypatch):
        """
        Test that estimator sigma_clip variants match the per-aperture
        loop.
        """
        estimator = MedianBackground(sigma_clip=sigma_clip)
        local_bkg = LocalBackground(5, 10, bkg_estimator=estimator)
        fast, slow = self._run_both_paths(local_bkg, monkeypatch)
        assert_allclose(fast, slow, rtol=1e-12)

    def test_nonfinite_data_matches_slow_path(self, monkeypatch):
        """
        Test that non-finite data values are excluded, matching the
        per-aperture loop (where sigma clipping removes them with a
        warning; the fast path masks them silently).
        """
        data = self.data.copy()
        data[30:34, 36:40] = np.nan  # inside the annulus at (30, 30)
        local_bkg = LocalBackground(5, 10)

        fast = local_bkg(data, self.x, self.y, mask=self.mask)
        monkeypatch.setattr(LocalBackground, '_fast_estimator_spec',
                            lambda _self: None)
        match = 'Input data contains invalid values'
        with pytest.warns(AstropyUserWarning, match=match):
            slow = local_bkg(data, self.x, self.y, mask=self.mask)
        assert_allclose(fast, slow, rtol=1e-12)

    def test_unsupported_estimators_fall_back(self):
        """
        Test that unsupported estimators fall back to the per-aperture
        loop and still produce finite results.
        """

        class MyMedianBackground(MedianBackground):
            pass

        def estimator_func(values):
            return np.nanmedian(values)

        # Note: np.float64 anchors are used because a Python float M
        # triggers an astropy biweight_location bug
        unsupported = [BiweightLocationBackground(c=5.0),
                       BiweightLocationBackground(M=np.float64(1.0)),
                       BiweightScaleBackgroundRMS(c=8.0),
                       BiweightScaleBackgroundRMS(M=np.float64(1.0)),
                       MyMedianBackground(), estimator_func]
        data = np.ones((51, 51))
        for estimator in unsupported:
            local_bkg = LocalBackground(5, 10, bkg_estimator=estimator)
            assert local_bkg._fast_estimator_spec() is None
            assert np.isfinite(local_bkg(data, 25, 25))

    def test_n_threads_identical(self):
        """
        Test that n_threads (passed through to ApertureStats) gives
        results identical to the single-threaded computation.
        """
        local_bkg1 = LocalBackground(5, 10)
        local_bkg4 = LocalBackground(5, 10, n_threads=4)
        assert local_bkg4.n_threads == 4
        bkg1 = local_bkg1(self.data, self.x, self.y, mask=self.mask)
        bkg4 = local_bkg4(self.data, self.x, self.y, mask=self.mask)
        assert_equal(bkg1, bkg4)

    def test_n_threads_repr(self):
        """
        Test that n_threads appears in the repr.
        """
        local_bkg = LocalBackground(5, 10, n_threads=4)
        assert 'n_threads=4' in repr(local_bkg)

    def test_invalid_n_threads(self):
        """
        Test that an error is raised if n_threads is not a positive
        integer.
        """
        match = 'n_threads must be a positive integer'
        for n_threads in (0, -1, 2.5):
            with pytest.raises(ValueError, match=match):
                LocalBackground(5, 10, n_threads=n_threads)
