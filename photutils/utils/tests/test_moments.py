# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _moments module.
"""

import numpy as np
import pytest
from astropy.wcs import WCS
from numpy.testing import assert_allclose, assert_equal

from photutils.utils._moments import (PIXEL_VARIANCE, centroid_from_moments,
                                      covariance_determinant,
                                      covariance_from_moments,
                                      covariance_min_eigval,
                                      eigvals_from_covariance, image_moments,
                                      inertia_tensor_from_moments,
                                      is_singular_covariance,
                                      orientation_from_covariance,
                                      pixel_to_sky_covariance,
                                      regularize_covariance,
                                      sky_orientation_from_covariance)


@pytest.fixture
def moments_central():
    """
    Central moments for four sources with a leading source axis.

    Index ``[:, i, j]`` is the sum of ``y**i * x**j``. The sources are:

    0. An axis-aligned ellipse with x variance 4 and y variance 1.
    1. A unit-variance source with covariance 0.5 (45 degree tilt).
    2. A source with zero total flux (undefined covariance).
    3. A point-like source with zero second moments.
    """
    moments = np.zeros((4, 3, 3))
    moments[0, 0, 0] = 2.0
    moments[0, 0, 2] = 8.0
    moments[0, 2, 0] = 2.0
    moments[1, 0, 0] = 1.0
    moments[1, 0, 2] = 1.0
    moments[1, 2, 0] = 1.0
    moments[1, 1, 1] = 0.5
    moments[2, 0, 2] = 1.0
    moments[3, 0, 0] = 1.0
    return moments


@pytest.fixture
def covariances():
    """
    Raw ``(N, 2, 2)`` covariance matrices covering the shape cases.

    0. Resolved axis-aligned ellipse (det 4, min eigenvalue 1).
    1. Point-like, all zeros (det 0, min eigenvalue 0).
    2. Rank-1 degenerate along x (det 0, min eigenvalue 0).
    3. Elongated with one unresolved axis (det 0.2, min eigenvalue
       0.05, which is below 1/12 while det is above (1/12)**2).
    4. Not positive semidefinite (det -3, min eigenvalue -1).
    5. Partially non-finite.
    """
    return np.array([[[4.0, 0.0], [0.0, 1.0]],
                     [[0.0, 0.0], [0.0, 0.0]],
                     [[4.0, 0.0], [0.0, 0.0]],
                     [[4.0, 0.0], [0.0, 0.05]],
                     [[1.0, 2.0], [2.0, 1.0]],
                     [[np.nan, 0.0], [0.0, 1.0]]])


@pytest.fixture
def tan_wcs():
    """
    A TAN WCS with a 1 arcsec pixel scale, North up and East left.

    The forward Jacobian at the reference pixel maps ``+x`` to ``-East``
    and ``+y`` to ``+North``.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crpix = [50.0, 50.0]
    wcs.wcs.crval = [10.0, 0.0]
    wcs.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
    return wcs


def test_moments():
    """
    Test image_moments with a simple 2x2 array (raw moments).
    """
    data = np.array([[0, 1], [0, 1]])
    moments = image_moments(data, order=2)
    result = np.array([[2, 2, 2], [1, 1, 1], [1, 1, 1]])

    assert_equal(moments, result)
    assert_allclose(moments[0, 1] / moments[0, 0], 1.0)
    assert_allclose(moments[1, 0] / moments[0, 0], 0.5)


def test_moments_central():
    """
    Test image_moments with center=None (central moments).
    """
    data = np.array([[0, 1], [0, 1]])
    moments = image_moments(data, center=None, order=2)
    result = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert_allclose(moments, result)


def test_moments_central_nonsquare():
    """
    Test image_moments with center=None and a non-square array.
    """
    data = np.array([[0, 1], [0, 1], [0, 1]])
    moments = image_moments(data, center=None, order=2)
    result = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert_allclose(moments, result)


def test_moments_central_invalid_dim():
    """
    Test that image_moments with non-2D data raises ValueError.
    """
    data = np.arange(27).reshape(3, 3, 3)
    match = 'data must be a 2D array'
    with pytest.raises(ValueError, match=match):
        image_moments(data, order=3)


def test_moments_central_negative_order():
    """
    Test that image_moments with negative order raises ValueError.
    """
    data = np.array([[0, 1], [0, 1]])
    match = 'order must be non-negative'
    with pytest.raises(ValueError, match=match):
        image_moments(data, order=-1)


def test_pixel_variance():
    """
    Test the single-pixel variance constant.
    """
    assert PIXEL_VARIANCE == 1.0 / 12.0


class TestCentroidFromMoments:
    """
    Tests for centroid_from_moments.
    """

    def test_values(self):
        """
        Test the (x, y) order against raw image moments.
        """
        data = np.zeros((5, 7))
        data[1, 4] = 1.0
        data[3, 4] = 3.0
        moments = np.array([image_moments(data, order=1)])
        centroid = centroid_from_moments(moments)
        assert centroid.shape == (1, 2)
        assert_allclose(centroid, [[4.0, 2.5]])

    def test_zero_flux(self):
        """
        Test that zero total flux gives NaN without a warning.
        """
        centroid = centroid_from_moments(np.zeros((2, 2, 2)))
        assert centroid.shape == (2, 2)
        assert np.all(np.isnan(centroid))


class TestInertiaTensorFromMoments:
    """
    Tests for inertia_tensor_from_moments.
    """

    def test_values(self, moments_central):
        """
        Test the element layout and the sign of the cross term.
        """
        tensor = inertia_tensor_from_moments(moments_central)
        assert tensor.shape == (4, 2, 2)
        assert_allclose(tensor[0], [[8.0, 0.0], [0.0, 2.0]])
        assert_allclose(tensor[1], [[1.0, -0.5], [-0.5, 1.0]])
        assert_allclose(tensor[2], [[1.0, 0.0], [0.0, 0.0]])
        assert_allclose(tensor[3], [[0.0, 0.0], [0.0, 0.0]])


class TestCovarianceFromMoments:
    """
    Tests for covariance_from_moments.
    """

    def test_values(self, moments_central):
        """
        Test the element layout and the flux normalization.
        """
        covar = covariance_from_moments(moments_central)
        assert covar.shape == (4, 2, 2)
        assert_allclose(covar[0], [[4.0, 0.0], [0.0, 1.0]])
        assert_allclose(covar[1], [[1.0, 0.5], [0.5, 1.0]])
        assert_allclose(covar[3], [[0.0, 0.0], [0.0, 0.0]])
        assert_equal(covar[:, 0, 1], covar[:, 1, 0])

    def test_zero_flux(self, moments_central):
        """
        Test that zero total flux is non-finite without a warning.
        """
        covar = covariance_from_moments(moments_central)
        assert not np.any(np.isfinite(covar[2]))

    def test_higher_order_moments_ignored(self, moments_central):
        """
        Test that third-order and higher moments are not used.
        """
        moments = np.full((4, 5, 5), 99.0)
        moments[:, :3, :3] = moments_central
        assert_equal(covariance_from_moments(moments),
                     covariance_from_moments(moments_central))


def test_covariance_determinant(covariances):
    """
    Test the determinants, including a NaN matrix without a warning.
    """
    det = covariance_determinant(covariances)
    assert det.shape == (6,)
    assert_allclose(det[:5], [4.0, 0.0, 0.0, 0.2, -3.0])
    assert np.isnan(det[5])


class TestCovarianceMinEigval:
    """
    Tests for covariance_min_eigval.
    """

    def test_values(self, covariances):
        """
        Test the closed-form smaller eigenvalue.
        """
        min_eig = covariance_min_eigval(covariances)
        assert min_eig.shape == (6,)
        assert_allclose(min_eig[:5], [1.0, 0.0, 0.0, 0.05, -1.0],
                        atol=1e-15)
        assert np.isnan(min_eig[5])

    def test_matches_eigvalsh(self):
        """
        Test the closed form against numpy for random matrices.
        """
        rng = np.random.default_rng(0)
        arr = rng.normal(size=(20, 2, 2))
        covar = arr @ arr.swapaxes(1, 2)  # symmetric
        expected = np.linalg.eigvalsh(covar)[:, 0]
        assert_allclose(covariance_min_eigval(covar), expected)

    def test_precomputed_determinant(self, covariances):
        """
        Test that a precomputed determinant gives the same result.
        """
        det = covariance_determinant(covariances)
        assert_equal(covariance_min_eigval(covariances, determinant=det),
                     covariance_min_eigval(covariances))


class TestIsSingularCovariance:
    """
    Tests for is_singular_covariance.
    """

    def test_determinant_only(self, covariances):
        """
        Test the determinant-only form.

        The elongated source 3 is not flagged. The source 4 with a
        negative determinant is flagged.
        """
        mask = is_singular_covariance(covariances,
                                      include_degenerate=False)
        assert mask.dtype == bool
        assert_equal(mask, [False, True, True, False, True, False])

    def test_include_degenerate(self, covariances):
        """
        Test that the elongated source 3 is also flagged.
        """
        mask = is_singular_covariance(covariances,
                                      include_degenerate=True)
        assert mask.dtype == bool
        assert_equal(mask, [False, True, True, True, True, False])

    @pytest.mark.parametrize('include_degenerate', [False, True])
    def test_threshold(self, include_degenerate):
        """
        Test that the threshold is the single-pixel variance.
        """
        covar = np.array([(PIXEL_VARIANCE - 1e-9) * np.eye(2),
                          (PIXEL_VARIANCE + 1e-9) * np.eye(2)])
        mask = is_singular_covariance(
            covar, include_degenerate=include_degenerate)
        assert_equal(mask, [True, False])


class TestRegularizeCovariance:
    """
    Tests for regularize_covariance.
    """

    def test_values(self, covariances):
        """
        Test each shape case.

        The elongated source 3 has a determinant above the threshold,
        so it is not bumped even though its minor-axis variance is below
        the single-pixel variance.
        """
        reg = regularize_covariance(covariances)
        assert_equal(reg[0], covariances[0])
        assert_allclose(reg[1], PIXEL_VARIANCE * np.eye(2))
        assert_allclose(reg[2], [[4.0 + PIXEL_VARIANCE, 0.0],
                                 [0.0, PIXEL_VARIANCE]])
        assert_equal(reg[3], covariances[3])
        assert np.all(np.isnan(reg[4]))
        assert_equal(reg[5], covariances[5])

    def test_negative_trace(self):
        """
        Test that a positive determinant with a negative trace is NaN.
        """
        covar = np.array([[[-1.0, 0.0], [0.0, -1.0]]])
        assert np.all(np.isnan(regularize_covariance(covar)))

    def test_input_not_modified(self, covariances):
        """
        Test that the input array is left untouched.
        """
        original = covariances.copy()
        reg = regularize_covariance(covariances)
        assert reg is not covariances
        assert_equal(covariances, original)

    def test_single_bump_clears_threshold(self):
        """
        Test that one bump lifts the determinant to the threshold.

        Diagonal matrices are used so the determinant is an exact
        non-negative product with no rounding below zero.
        """
        rng = np.random.default_rng(1)
        var_x = rng.uniform(0.1, 10.0, size=50)
        var_y = rng.uniform(0.0, 1.0, size=50) * PIXEL_VARIANCE**2 / var_x
        covar = np.zeros((50, 2, 2))
        covar[:, 0, 0] = var_x
        covar[:, 1, 1] = var_y
        assert np.all(covariance_determinant(covar) < PIXEL_VARIANCE**2)
        det = covariance_determinant(regularize_covariance(covar))
        assert np.all(det >= PIXEL_VARIANCE**2 * (1 - 1e-12))


class TestCovarianceEigvals:
    """
    Tests for eigvals_from_covariance.
    """

    def test_values(self, covariances):
        """
        Test the descending order and the NaN cases.

        Source 4 has eigenvalues 3 and -1. Source 5 is partially
        non-finite.
        """
        eigvals = eigvals_from_covariance(covariances)
        assert eigvals.shape == (6, 2)
        assert_allclose(eigvals[0], [4.0, 1.0])
        assert_allclose(eigvals[3], [4.0, 0.05])
        assert np.all(np.isnan(eigvals[4:]))

    def test_tilted(self):
        """
        Test a matrix with a nonzero off-diagonal element.
        """
        covar = np.array([[[1.0, 0.5], [0.5, 1.0]]])
        assert_allclose(eigvals_from_covariance(covar), [[1.5, 0.5]])

    def test_no_finite_matrices(self):
        """
        Test all-NaN and empty inputs.
        """
        eigvals = eigvals_from_covariance(np.full((2, 2, 2), np.nan))
        assert eigvals.shape == (2, 2)
        assert np.all(np.isnan(eigvals))
        assert eigvals_from_covariance(np.empty((0, 2, 2))).shape == (0, 2)


def test_orientation_from_covariance(covariances):
    """
    Test the orientation in degrees, including the (-90, 90] range.
    """
    theta = orientation_from_covariance(covariances)
    assert theta.shape == (6,)
    assert_allclose(theta[0], 0.0)
    assert np.isnan(theta[5])

    covar = np.array([[[1.0, 0.5], [0.5, 1.0]],
                      [[1.0, -0.5], [-0.5, 1.0]],
                      [[1.0, 0.0], [0.0, 4.0]]])
    assert_allclose(orientation_from_covariance(covar), [45.0, -45.0, 90.0])


class TestPixelToSkyCovariance:
    """
    Tests for pixel_to_sky_covariance.
    """

    def test_values(self, tan_wcs):
        """
        Test the transport and the NaN cases.

        The x-axis flip changes the sign of the off-diagonal element.
        A non-finite covariance or position gives a NaN matrix.
        """
        pix_cov = np.array([[[4.0, 1.0], [1.0, 2.0]],
                            [[np.nan, 1.0], [1.0, 2.0]],
                            [[4.0, 1.0], [1.0, 2.0]]])
        xycen = np.array([[49.0, 49.0], [49.0, 49.0], [np.nan, 49.0]])
        sky_cov = pixel_to_sky_covariance(tan_wcs, pix_cov, xycen)
        assert sky_cov.shape == (3, 2, 2)
        assert_allclose(sky_cov[0], [[4.0, -1.0], [-1.0, 2.0]], rtol=1e-6)
        assert np.all(np.isnan(sky_cov[1:]))

    def test_no_finite_sources(self, tan_wcs):
        """
        Test that the WCS is not evaluated when no source is finite.
        """
        pix_cov = np.full((2, 2, 2), np.nan)
        xycen = np.full((2, 2), 49.0)
        sky_cov = pixel_to_sky_covariance(tan_wcs, pix_cov, xycen)
        assert sky_cov.shape == (2, 2, 2)
        assert np.all(np.isnan(sky_cov))


def test_sky_orientation_from_covariance():
    """
    Test the position angle, measured from North toward East.

    The matrices are ordered (East, North).
    """
    sky_cov = np.array([[[4.0, 0.0], [0.0, 1.0]],
                        [[1.0, 0.0], [0.0, 4.0]],
                        [[1.0, 0.5], [0.5, 1.0]],
                        [[1.0, -0.5], [-0.5, 1.0]],
                        [[np.nan, 0.0], [0.0, 1.0]]])
    angle = sky_orientation_from_covariance(sky_cov)
    assert angle.shape == (5,)
    assert_allclose(angle[:4], [90.0, 0.0, 45.0, -45.0])
    assert np.isnan(angle[4])
