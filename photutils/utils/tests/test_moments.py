# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _moments module.
"""

import warnings

import numpy as np
import pytest
from astropy.wcs import WCS
from numpy.testing import assert_allclose, assert_equal

from photutils.utils._moments import (PIXEL_VARIANCE, PSD_RTOL,
                                      centroid_from_moments,
                                      covariance_determinant,
                                      covariance_from_moments,
                                      covariance_max_eigval,
                                      covariance_min_eigval,
                                      eigvals_from_covariance,
                                      floor_covariance_eigvals, image_moments,
                                      inertia_tensor_from_moments,
                                      is_invalid_covariance,
                                      is_singular_covariance, major_axis_angle,
                                      orientation_from_covariance,
                                      pixel_to_sky_covariance,
                                      regularize_covariance,
                                      sky_orientation_from_covariance)


@pytest.fixture(autouse=True)
def warnings_as_errors():
    """
    Turn every warning into an error for each test in this module.

    The helpers must not emit a warning for zero-flux or non-finite
    inputs. This fixture checks that without relying on the pytest
    warnings plugin, which some CI jobs disable.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        yield


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


class TestImageMoments:
    """
    Tests for image_moments.
    """

    def test_raw(self):
        """
        Test the raw moments of a simple 2x2 array.
        """
        data = np.array([[0, 1], [0, 1]])
        moments = image_moments(data, order=2)
        result = np.array([[2, 2, 2], [1, 1, 1], [1, 1, 1]])

        assert_equal(moments, result)
        assert_allclose(moments[0, 1] / moments[0, 0], 1.0)
        assert_allclose(moments[1, 0] / moments[0, 0], 0.5)

    def test_central(self):
        """
        Test the central moments about the centroid.
        """
        data = np.array([[0, 1], [0, 1]])
        moments = image_moments(data, center=(1.0, 0.5), order=2)
        result = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                           [0.5, 0.0, 0.0]])
        assert_allclose(moments, result)

    def test_central_nonsquare(self):
        """
        Test the central moments of a non-square array.
        """
        data = np.array([[0, 1], [0, 1], [0, 1]])
        moments = image_moments(data, center=(1.0, 1.0), order=2)
        result = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                           [2.0, 0.0, 0.0]])
        assert_allclose(moments, result)

    def test_invalid_dim(self):
        """
        Test that non-2D data raises ValueError.
        """
        data = np.arange(27).reshape(3, 3, 3)
        match = 'data must be a 2D array'
        with pytest.raises(ValueError, match=match):
            image_moments(data, order=3)

    def test_negative_order(self):
        """
        Test that a negative order raises ValueError.
        """
        data = np.array([[0, 1], [0, 1]])
        match = 'order must be non-negative'
        with pytest.raises(ValueError, match=match):
            image_moments(data, order=-1)


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


@pytest.mark.parametrize('index', [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_covariance_determinant_nan(index):
    """
    Test that a NaN in any element gives a NaN determinant.

    The zero elements make the matrix singular if the NaN is ignored,
    which is how a pivoted LU determinant can return zero.
    """
    covar = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    covar[0][index] = np.nan
    assert np.isnan(covariance_determinant(covar)[0])
    covar = np.full((1, 2, 2), np.nan)
    assert np.isnan(covariance_determinant(covar)[0])


class TestCovarianceMinEigval:
    """
    Tests for covariance_min_eigval.
    """

    def test_values(self, covariances):
        """
        Test the closed-form smaller eigenvalue.
        """
        det = covariance_determinant(covariances)
        min_eig = covariance_min_eigval(covariances, determinant=det)
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
        det = covariance_determinant(covar)
        min_eig = covariance_min_eigval(covar, determinant=det)
        assert_allclose(min_eig, expected)

    def test_nearly_isotropic(self):
        """
        Test nearly isotropic matrices, whose two eigenvalues differ by
        about the rounding error of the matrix elements.
        """
        rng = np.random.default_rng(2)
        var_x = rng.uniform(0.01, 100.0, size=1000)
        var_y = var_x * (1.0 + rng.uniform(-1e-9, 1e-9, size=1000))
        covar = np.zeros((1000, 2, 2))
        covar[:, 0, 0] = var_x
        covar[:, 1, 1] = var_y
        det = covariance_determinant(covar)
        min_eig = covariance_min_eigval(covar, determinant=det)
        assert_allclose(min_eig, np.minimum(var_x, var_y), rtol=1e-7)

    def test_non_positive_trace(self):
        """
        Test matrices with a zero or negative trace.
        """
        covar = np.array([[[0.0, 0.0], [0.0, 0.0]],
                          [[-1.0, 0.0], [0.0, -2.0]],
                          [[-2.0, 1.0], [1.0, -2.0]],
                          [[1.0, 0.0], [0.0, -1.0]]])
        det = covariance_determinant(covar)
        min_eig = covariance_min_eigval(covar, determinant=det)
        assert_allclose(min_eig, [0.0, -2.0, -3.0, -1.0], atol=1e-15)

    @pytest.mark.parametrize('major', [1e6, 1e10, 1e14, 1e16])
    def test_highly_elongated(self, major):
        """
        Test that a large major-axis variance does not degrade the
        minor-axis variance through cancellation.
        """
        covar = np.array([[[major, 0.0], [0.0, 0.1]]])
        det = covariance_determinant(covar)
        min_eig = covariance_min_eigval(covar, determinant=det)
        assert_allclose(min_eig, [0.1], rtol=1e-12)

    def test_no_overflow(self):
        """
        Test that a variance whose square overflows gives the correct
        minor-axis variance.
        """
        covar = np.array([[[1e200, 0.0], [0.0, 0.01]]])
        det = covariance_determinant(covar)
        min_eig = covariance_min_eigval(covar, determinant=det)
        assert_allclose(min_eig, [0.01], rtol=1e-12)


class TestCovarianceMaxEigval:
    """
    Tests for covariance_max_eigval.
    """

    def test_values(self, covariances):
        """
        Test the closed-form larger eigenvalue, including NaN
        propagation.
        """
        eig_max = covariance_max_eigval(covariances)
        assert eig_max.shape == (6,)
        assert_allclose(eig_max[:5], [4.0, 0.0, 4.0, 4.0, 3.0])
        assert np.isnan(eig_max[5])

    def test_matches_eigvalsh(self):
        """
        Test random symmetric matrices against np.linalg.eigvalsh.
        """
        rng = np.random.default_rng(0)
        arr = rng.normal(size=(200, 2, 2))
        covar = arr @ arr.transpose(0, 2, 1)
        expected = np.linalg.eigvalsh(covar)[:, 1]
        assert_allclose(covariance_max_eigval(covar), expected, rtol=1e-13)

    def test_no_overflow(self):
        """
        Test that huge variances do not overflow.
        """
        covar = np.array([[[1e300, 0.0], [0.0, 1e290]]])
        assert_allclose(covariance_max_eigval(covar), 1e300)


class TestIsSingularCovariance:
    """
    Tests for is_singular_covariance.
    """

    def test_values(self, covariances):
        """
        Test each shape case.

        The elongated source 3 has a determinant above ``(1/12)**2`` and
        is flagged only by its minor-axis variance. The source 4 is not
        positive semidefinite, so it is invalid instead of singular. The
        partially non-finite source 5 is not flagged.
        """
        det = covariance_determinant(covariances)
        mask = is_singular_covariance(covariances, determinant=det)
        assert mask.dtype == bool
        assert_equal(mask, [False, True, True, True, False, False])

    def test_threshold(self):
        """
        Test that the threshold is the single-pixel variance.
        """
        covar = np.array([[[4.0, 0.0], [0.0, PIXEL_VARIANCE - 1e-9]],
                          [[4.0, 0.0], [0.0, PIXEL_VARIANCE + 1e-9]]])
        det = covariance_determinant(covar)
        mask = is_singular_covariance(covar, determinant=det)
        assert_equal(mask, [True, False])

    def test_includes_determinant_test(self):
        """
        Test that every symmetric matrix with a determinant below
        ``PIXEL_VARIANCE**2`` is flagged as singular or as invalid, and
        never as both.
        """
        rng = np.random.default_rng(3)
        covar = rng.normal(scale=0.3, size=(5000, 2, 2))
        covar[:, 1, 0] = covar[:, 0, 1]
        det = covariance_determinant(covar)
        small_det = det < PIXEL_VARIANCE**2
        assert 0 < np.count_nonzero(small_det) < len(det)
        singular = is_singular_covariance(covar, determinant=det)
        invalid = is_invalid_covariance(covar, determinant=det)
        assert np.any(singular)
        assert np.any(invalid)
        assert not np.any(singular & invalid)
        assert np.all((singular | invalid)[small_det])


class TestIsInvalidCovariance:
    """
    Tests for is_invalid_covariance.
    """

    def test_values(self, covariances):
        """
        Test that only the source 4, which is not positive semidefinite,
        is flagged.

        The singular sources 1 to 3 are valid, and the partially
        non-finite source 5 is not flagged.
        """
        det = covariance_determinant(covariances)
        mask = is_invalid_covariance(covariances, determinant=det)
        assert mask.dtype == bool
        assert_equal(mask, [False, False, False, False, True, False])

    def test_negative_trace(self):
        """
        Test that a negative definite matrix, whose determinant is
        positive, is flagged by its trace.
        """
        covar = np.array([[[-1.0, 0.0], [0.0, -2.0]]])
        det = covariance_determinant(covar)
        assert det[0] > 0
        assert_equal(is_invalid_covariance(covar, determinant=det), [True])

    def test_infinite_determinant(self):
        """
        Test that a matrix that is not positive semidefinite is flagged
        as invalid, and not as singular, when its determinant overflows
        to negative infinity.
        """
        covar = np.array([[[1e200, 1e200], [1e200, -1e200]]])
        det = covariance_determinant(covar)
        assert det[0] == -np.inf
        assert_equal(is_invalid_covariance(covar, determinant=det), [True])
        assert_equal(is_singular_covariance(covar, determinant=det), [False])

    def test_rank_one_rounding(self):
        """
        Test that an exactly thin tilted matrix is valid even when its
        computed determinant is slightly negative.
        """
        rng = np.random.default_rng(5)
        vec = rng.normal(size=(2000, 2)) * rng.uniform(0.1, 1e3, (2000, 1))
        covar = vec[:, :, np.newaxis] * vec[:, np.newaxis, :]
        det = covariance_determinant(covar)
        assert np.any(det < 0)
        assert not np.any(is_invalid_covariance(covar, determinant=det))
        assert np.all(is_singular_covariance(covar, determinant=det))

    def test_matches_regularized_nan(self):
        """
        Test that the invalid matrices are exactly the finite ones that
        regularize_covariance sets to NaN.
        """
        rng = np.random.default_rng(6)
        covar = rng.normal(scale=0.5, size=(5000, 2, 2))
        covar[:, 1, 0] = covar[:, 0, 1]
        det = covariance_determinant(covar)
        invalid = is_invalid_covariance(covar, determinant=det)
        reg = regularize_covariance(covar, determinant=det)
        assert 0 < np.count_nonzero(invalid) < len(det)
        assert_equal(np.isnan(reg).all(axis=(1, 2)), invalid)


class TestFloorCovarianceEigvals:
    """
    Tests for floor_covariance_eigvals.
    """

    def test_scalar_minimum(self):
        """
        Test that only an eigenvalue below the minimum is raised.
        """
        covar = np.array([[[4.0, 0.0], [0.0, 0.05]],
                          [[0.0, 0.0], [0.0, 0.0]]])
        floored = floor_covariance_eigvals(covar, minimum=0.1)
        assert_allclose(floored[0], [[4.0, 0.0], [0.0, 0.1]])
        assert_allclose(floored[1], 0.1 * np.eye(2))

    def test_array_minimum(self):
        """
        Test a separate minimum for each matrix.
        """
        covar = np.array([[[4.0, 0.0], [0.0, 0.05]],
                          [[4.0, 0.0], [0.0, 0.05]]])
        floored = floor_covariance_eigvals(covar,
                                           minimum=np.array([0.01, 0.2]))
        assert_equal(floored[0], covar[0])
        assert_allclose(floored[1], [[4.0, 0.0], [0.0, 0.2]])

    def test_tilted(self):
        """
        Test that the eigenvectors of a tilted matrix are kept.
        """
        covar = np.array([[[2.0, 1.95], [1.95, 2.0]]])
        floored = floor_covariance_eigvals(covar, minimum=0.5)
        assert_allclose(eigvals_from_covariance(floored), [[3.95, 0.5]])
        assert_allclose(orientation_from_covariance(floored), [45.0])
        assert_equal(floored[:, 0, 1], floored[:, 1, 0])

    @pytest.mark.parametrize('angle', [10.0, 30.0, 45.0, 77.0])
    def test_both_below_minimum(self, angle):
        """
        Test that a tilted matrix with both eigenvalues below the
        minimum becomes exactly isotropic, with no spurious orientation
        from rounding.
        """
        theta = np.deg2rad(angle)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        covar = (rot @ np.diag([0.05, 0.02]) @ rot.T)[np.newaxis]
        floored = floor_covariance_eigvals(covar, minimum=0.1)
        assert_equal(floored[0], 0.1 * np.eye(2))
        assert_equal(orientation_from_covariance(floored), [0.0])

    def test_floor_precision(self):
        """
        Test that a floored eigenvalue equals the minimum to within the
        rounding error of the matrix elements, which scales with the
        larger eigenvalue.
        """
        rng = np.random.default_rng(2)
        n_matrices = 1000
        theta = rng.uniform(0.0, np.pi, n_matrices)
        eig_max = 10.0**rng.uniform(-0.5, 4.0, n_matrices)
        eig_min = rng.uniform(0.0, PIXEL_VARIANCE, n_matrices)
        cos, sin = np.cos(theta), np.sin(theta)
        rot = np.stack([np.stack([cos, -sin], axis=-1),
                        np.stack([sin, cos], axis=-1)], axis=-2)
        eigvals = np.stack([eig_max, eig_min], axis=-1)
        covar = np.einsum('nij,nj,nkj->nik', rot, eigvals, rot)
        covar[:, 1, 0] = covar[:, 0, 1]

        floored = floor_covariance_eigvals(covar, minimum=PIXEL_VARIANCE)
        result = eigvals_from_covariance(floored)
        tol = 10.0 * np.finfo(float).eps * eig_max
        assert np.all(np.abs(result[:, 1] - PIXEL_VARIANCE) <= tol)
        assert_allclose(result[:, 0], eig_max, rtol=1e-14)

    @pytest.mark.parametrize('covar', [
        [[1e200, 0.0], [0.0, 1e200]],
        [[1e308, 0.0], [0.0, 1e308]],
        [[1e308, 9e307], [9e307, 1e308]]])
    def test_overflow_unchanged(self, covar):
        """
        Test that a resolved matrix is returned unchanged, without a
        warning, when its determinant or its trace overflows.
        """
        covar = np.array([covar])
        floored = floor_covariance_eigvals(covar, minimum=PIXEL_VARIANCE)
        assert_equal(floored, covar)

    @pytest.mark.parametrize('covar', [
        [[np.nan, 0.0], [0.0, 1.0]],
        [[np.inf, 0.0], [0.0, 0.01]],
        [[0.01, np.inf], [np.inf, 0.01]],
        [[0.01, 0.0], [0.0, np.nan]]])
    def test_non_finite_unchanged(self, covar):
        """
        Test that a matrix with a non-finite element is returned
        unchanged, without a warning.
        """
        covar = np.array([covar, [[1.0, 0.0], [0.0, 0.01]]])
        floored = floor_covariance_eigvals(covar, minimum=PIXEL_VARIANCE)
        assert_equal(floored[0], covar[0])
        assert_allclose(floored[1], [[1.0, 0.0], [0.0, PIXEL_VARIANCE]])

    @pytest.mark.parametrize('covar', [
        [[1e308, 0.0], [0.0, -1e308]],
        [[0.0, 1e308], [1e308, 0.0]]])
    def test_overflow_indefinite(self, covar):
        """
        Test that a huge matrix that is not positive semidefinite, whose
        eigenvalue difference overflows, does not emit a warning and is
        set to NaN by the regularization.
        """
        covar = np.array([covar])
        floored = floor_covariance_eigvals(covar, minimum=PIXEL_VARIANCE)
        assert floored.shape == covar.shape
        det = covariance_determinant(covar)
        reg = regularize_covariance(covar, determinant=det)
        assert np.all(np.isnan(reg))

    def test_overflow_thin(self):
        """
        Test that only the unresolved axis is raised when the square of
        the other variance overflows.
        """
        covar = np.array([[[1e200, 0.0], [0.0, 0.01]]])
        floored = floor_covariance_eigvals(covar, minimum=PIXEL_VARIANCE)
        assert_allclose(floored, [[[1e200, 0.0], [0.0, PIXEL_VARIANCE]]],
                        rtol=1e-12)

    def test_unchanged_and_not_modified(self):
        """
        Test that a matrix above the minimum is returned bit for bit
        and that the input array is left untouched.
        """
        covar = np.array([[[2.0, 0.3], [0.3, 1.0]],
                          [[2.0, 0.3], [0.3, 0.04]]])
        original = covar.copy()
        floored = floor_covariance_eigvals(covar, minimum=0.1)
        assert floored is not covar
        assert_equal(covar, original)
        assert_equal(floored[0], covar[0])
        assert not np.array_equal(floored[1], covar[1])

    def test_empty(self):
        """
        Test an input with no matrices.
        """
        floored = floor_covariance_eigvals(np.empty((0, 2, 2)), minimum=0.1)
        assert floored.shape == (0, 2, 2)


def regularize(covariance):
    """
    Regularize covariance matrices with the determinant computed from
    the same matrices.
    """
    det = covariance_determinant(covariance)
    return regularize_covariance(covariance, determinant=det)


class TestRegularizeCovariance:
    """
    Tests for regularize_covariance.
    """

    def test_values(self, covariances):
        """
        Test each shape case.

        Only an eigenvalue below the single-pixel variance is raised.
        The resolved major axis of sources 2 and 3 is unchanged.
        """
        reg = regularize(covariances)
        assert_equal(reg[0], covariances[0])
        assert_allclose(reg[1], PIXEL_VARIANCE * np.eye(2))
        assert_allclose(reg[2], [[4.0, 0.0], [0.0, PIXEL_VARIANCE]])
        assert_allclose(reg[3], [[4.0, 0.0], [0.0, PIXEL_VARIANCE]])
        assert np.all(np.isnan(reg[4]))
        assert_equal(reg[5], covariances[5])

    def test_tilted(self):
        """
        Test that the orientation and the resolved eigenvalue of a
        tilted thin source are preserved.
        """
        covar = np.array([[[2.0, 1.95], [1.95, 2.0]]])
        reg = regularize(covar)
        assert_allclose(eigvals_from_covariance(reg),
                        [[3.95, PIXEL_VARIANCE]])
        assert_allclose(orientation_from_covariance(reg), [45.0])
        assert_equal(reg[:, 0, 1], reg[:, 1, 0])

    def test_continuous_at_threshold(self):
        """
        Test that the regularized matrix is continuous where the
        minor-axis variance crosses the single-pixel variance.
        """
        covar = np.array([[[4.0, 0.0], [0.0, PIXEL_VARIANCE - 1e-9]],
                          [[4.0, 0.0], [0.0, PIXEL_VARIANCE + 1e-9]]])
        reg = regularize(covar)
        assert_allclose(reg[0], reg[1], atol=2e-9)

    def test_rank_one_rounding(self):
        """
        Test that an exactly thin tilted source is regularized when
        rounding makes its computed determinant slightly negative.

        The determinant of a rank-1 matrix is zero. A slightly negative
        value is passed explicitly because whether rounding produces one
        depends on the matrix elements.
        """
        covar = np.array([[[0.25, -0.25], [-0.25, 0.25]]])
        det = np.array([-1.0e-17])
        mask = is_singular_covariance(covar, determinant=det)
        assert_equal(mask, [True])
        reg = regularize_covariance(covar, determinant=det)
        assert_allclose(eigvals_from_covariance(reg),
                        [[0.5, PIXEL_VARIANCE]])
        assert_allclose(orientation_from_covariance(reg), [-45.0])

    def test_negative_determinant(self):
        """
        Test that a determinant that is negative beyond rounding still
        gives NaN.
        """
        covar = np.array([[[0.25, -0.25], [-0.25, 0.25]]])
        det = np.array([-1.0e-9])
        reg = regularize_covariance(covar, determinant=det)
        assert np.all(np.isnan(reg))

    @pytest.mark.parametrize('scale', [1.0, 1.0e-6, 1.0e6])
    def test_negative_determinant_tolerance(self, scale):
        """
        Test that the tolerance on a negative determinant is
        ``PSD_RTOL`` times the squared trace, for any overall scale of
        the matrix.
        """
        covar = scale * np.array([[[0.25, -0.25], [-0.25, 0.25]],
                                  [[0.25, -0.25], [-0.25, 0.25]]])
        trace = 0.5 * scale
        det = -PSD_RTOL * trace**2 * np.array([0.5, 2.0])
        reg = regularize_covariance(covar, determinant=det)
        assert np.all(np.isfinite(reg[0]))
        assert np.all(np.isnan(reg[1]))

    def test_negative_trace(self):
        """
        Test that a positive determinant with a negative trace is NaN.
        """
        covar = np.array([[[-1.0, 0.0], [0.0, -1.0]]])
        assert covariance_determinant(covar)[0] > 0
        assert np.all(np.isnan(regularize(covar)))

    def test_input_not_modified(self, covariances):
        """
        Test that the input array is left untouched.
        """
        original = covariances.copy()
        reg = regularize(covariances)
        assert reg is not covariances
        assert_equal(covariances, original)

    def test_no_singular_sources(self):
        """
        Test that resolved sources are returned unchanged.
        """
        covar = np.array([[[4.0, 0.5], [0.5, 1.0]]])
        assert_equal(regularize(covar), covar)

    def test_floor_is_minimum_eigenvalue(self):
        """
        Test that every regularized eigenvalue reaches the single-pixel
        variance for random positive semidefinite matrices.
        """
        rng = np.random.default_rng(1)
        arr = rng.normal(scale=0.3, size=(50, 2, 2))
        covar = arr @ arr.swapaxes(1, 2)  # symmetric
        eigvals = eigvals_from_covariance(regularize(covar))
        assert np.all(eigvals >= PIXEL_VARIANCE * (1 - 1e-12))

    def test_modified_matches_singular_mask(self):
        """
        Test that the regularization modifies exactly the matrices
        selected by is_singular_covariance and is_invalid_covariance.

        The masks set the ``'singular_covariance'`` and
        ``'undefined_shape'`` flags but are not passed to
        regularize_covariance, so this pins that they use the same
        criteria. The singular matrices stay finite. The random
        symmetric matrices include ones that are not positive
        semidefinite, which are invalid and set to NaN. Most have a
        minor-axis variance close to ``PIXEL_VARIANCE``.
        """
        rng = np.random.default_rng(4)
        n_matrices = 20000
        theta = rng.uniform(0.0, np.pi, n_matrices)
        eig_max = 10.0**rng.uniform(-2.0, 3.0, n_matrices)
        eig_min = PIXEL_VARIANCE * rng.uniform(-0.5, 2.0, n_matrices)
        eig_min = np.minimum(eig_min, eig_max)
        cos, sin = np.cos(theta), np.sin(theta)
        covar = np.empty((n_matrices, 2, 2))
        covar[:, 0, 0] = eig_max * cos**2 + eig_min * sin**2
        covar[:, 1, 1] = eig_max * sin**2 + eig_min * cos**2
        covar[:, 0, 1] = (eig_max - eig_min) * cos * sin
        covar[:, 1, 0] = covar[:, 0, 1]

        det = covariance_determinant(covar)
        singular = is_singular_covariance(covar, determinant=det)
        invalid = is_invalid_covariance(covar, determinant=det)
        reg = regularize_covariance(covar, determinant=det)
        modified = np.any(reg != covar, axis=(1, 2))
        assert 0 < np.count_nonzero(singular) < n_matrices
        assert np.any(invalid)
        assert_equal(modified, singular | invalid)
        assert np.all(np.isfinite(reg[singular]))
        assert np.all(np.isnan(reg[invalid]))

    def test_idempotent(self, covariances):
        """
        Test that regularizing a regularized matrix does not change it
        beyond rounding error.

        A floored eigenvalue equals ``PIXEL_VARIANCE`` only to within
        rounding, so it can be floored again by a tiny amount.
        """
        reg = regularize(covariances)
        assert_allclose(regularize(reg), reg, rtol=1e-12)


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
        Test an all-NaN input.
        """
        eigvals = eigvals_from_covariance(np.full((2, 2, 2), np.nan))
        assert eigvals.shape == (2, 2)
        assert np.all(np.isnan(eigvals))

    def test_matches_eigvalsh(self):
        """
        Test random positive definite matrices, including highly
        elongated ones, against np.linalg.eigvalsh.
        """
        rng = np.random.default_rng(0)
        arr = rng.normal(size=(200, 2, 2))
        arr[::2, :, 1] *= 1e-6
        covar = arr @ arr.transpose(0, 2, 1) + 1e-13 * np.eye(2)
        expected = np.fliplr(np.linalg.eigvalsh(covar))
        eigvals = eigvals_from_covariance(covar)
        assert_allclose(eigvals[:, 0], expected[:, 0], rtol=1e-13)
        # The closed form is limited only by the determinant rounding
        atol = 10.0 * np.finfo(float).eps * expected[:, 0]
        assert np.all(np.abs(eigvals[:, 1] - expected[:, 1]) <= atol)

    def test_infinite_element(self):
        """
        Test that an infinite element gives NaN eigenvalues, without a
        warning.
        """
        covar = np.array([[[np.inf, 0.0], [0.0, 1.0]],
                          [[1.0, np.inf], [np.inf, 1.0]],
                          [[np.inf, 0.0], [0.0, np.inf]]])
        assert np.all(np.isnan(eigvals_from_covariance(covar)))

    def test_determinant_overflow(self):
        """
        Test that both eigenvalues are accurate when the determinant
        overflows.
        """
        covar = np.array([[[1e200, 0.0], [0.0, 1e190]],
                          [[2e200, 1e200], [1e200, 2e200]]])
        assert_allclose(eigvals_from_covariance(covar),
                        [[1e200, 1e190], [3e200, 1e200]], rtol=1e-14)

    def test_zero_matrix(self):
        """
        Test that an all-zero matrix has zero eigenvalues.
        """
        eigvals = eigvals_from_covariance(np.zeros((1, 2, 2)))
        assert_equal(eigvals, [[0.0, 0.0]])


def test_major_axis_angle():
    """
    Test the angle convention, the isotropic case, and NaN propagation.
    """
    var_a = np.array([4.0, 1.0, 1.0, 1.0, 1.0, np.inf])
    var_b = np.array([1.0, 4.0, 1.0, 1.0, 1.0, np.inf])
    covar_ab = np.array([0.0, 0.0, 0.5, -0.5, 0.0, 0.0])
    angle = major_axis_angle(var_a, var_b, covar_ab)
    assert_allclose(angle[:5], [0.0, 90.0, 45.0, -45.0, 0.0])
    assert np.isnan(angle[5])


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


def test_orientation_from_covariance_infinite():
    """
    Test that two infinite variances give NaN without a warning.
    """
    covar = np.array([[[np.inf, 0.0], [0.0, np.inf]]])
    assert np.isnan(orientation_from_covariance(covar)[0])


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


def test_sky_orientation_from_covariance_infinite():
    """
    Test that two infinite variances give NaN without a warning.
    """
    sky_cov = np.array([[[np.inf, 0.0], [0.0, np.inf]]])
    assert np.isnan(sky_orientation_from_covariance(sky_cov)[0])


def test_empty_inputs(tan_wcs):
    """
    Test that every helper accepts inputs with no sources.
    """
    moments = np.empty((0, 3, 3))
    covar = np.empty((0, 2, 2))
    det = covariance_determinant(covar)
    assert det.shape == (0,)
    assert centroid_from_moments(moments).shape == (0, 2)
    assert inertia_tensor_from_moments(moments).shape == (0, 2, 2)
    assert covariance_from_moments(moments).shape == (0, 2, 2)
    assert covariance_min_eigval(covar, determinant=det).shape == (0,)
    mask = is_singular_covariance(covar, determinant=det)
    assert mask.shape == (0,)
    assert mask.dtype == bool
    assert regularize(covar).shape == (0, 2, 2)
    assert eigvals_from_covariance(covar).shape == (0, 2)
    assert orientation_from_covariance(covar).shape == (0,)
    sky_cov = pixel_to_sky_covariance(tan_wcs, covar, np.empty((0, 2)))
    assert sky_cov.shape == (0, 2, 2)
    assert sky_orientation_from_covariance(sky_cov).shape == (0,)
