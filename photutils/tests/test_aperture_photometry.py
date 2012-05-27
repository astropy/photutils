# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal_nulp

from ..aperture import annulus_elliptical
from ..aperture import aperture_elliptical


class TestArrayShapes(object):

    def setup_class(self):
        self.data = np.ones((10, 10), dtype=np.float)

    def test_single_object_single_aperture(self):
        # Input single object and single aperture
        flux = annulus_elliptical(self.data, 5., 5., 2., 5., 5., np.pi / 4.)
        assert np.isscalar(flux) == True

    def test_single_object_multiple_apertures(self):
        # One object, multiple apertures
        flux = annulus_elliptical(self.data, 2., 5.,
                           [[2.], [5.]], [[5.], [8.]], [[5.], [8.]], np.pi / 2.)
        assert flux.shape == (2, 1)

    def test_multiple_objects_single_aperture(self):
        # Multiple objects, single aperture
        flux = annulus_elliptical(self.data, [2., 3., 4.], [5., 6., 7.],
                           2., 5., 5., np.pi / 2.)
        assert flux.shape == (3, )

    def test_multiple_objects_multiple_apertures(self):
        # Multiple objects, multiple apertures
        flux = annulus_elliptical(self.data, [2., 3., 4.], [5., 6., 7.],
                           [2., 3., 4.], [5., 6., 7.], [5., 6., 7.], np.pi / 2.)
        assert flux.shape == (3, )

    def test_multiple_objects_multiple_aperture_per_object(self):
        # Multiple objects, multiple apertures per object
        flux = annulus_elliptical(self.data, [2., 3., 4.], [5., 6., 7.],
                           [[2.], [5.]], [[5.], [8.]], [[5.], [8.]], np.pi / 2.)
        assert flux.shape == (2, 3)

    def test_mismatch_object_apertures(self):
        # Mismatched number of objects and apertures
        with pytest.raises(ValueError) as exc:
            annulus_elliptical(self.data, [2., 3., 4.], [5., 6., 7.],
                               [2., 3.], [5., 6.], [5., 6.], np.pi / 2.)
        assert exc.value.args[0] == "trailing dimension of 'apertures' must be 1 or match length of xc, yc"


def test_in_and_out_of_array():
    from ..aperture import annulus_elliptical
    import numpy as np

    # An annulus fully in the array
    data = np.ones((40, 40), dtype=np.float)
    a_in = 5.
    a_out = 10.
    b_out = 5.
    theta = np.pi / 4.
    flux = annulus_elliptical(data, 20., 20., a_in, a_out, b_out, theta)
    true_flux = np.pi * (a_out * b_out - a_in ** 2 * b_out / a_out)
    assert abs((flux - true_flux) / true_flux) < 0.01

    # An annulus fully out of the array should return 0
    a_in = 5.
    a_out = 10.
    b_out = 5.
    theta = np.pi / 4.
    flux = annulus_elliptical(data, 60., 60., a_in, a_out, b_out, theta)
    true_flux = np.pi * (a_out * b_out - a_in ** 2 * b_out / a_out)
    assert abs(flux) < 0.001


class TestErrorGain(object):

    def setup_class(self):
        self.data = np.ones((40, 40), dtype=np.float)
        self.xc = 20.
        self.yc = 20.
        self.a = 10.
        self.b = 5.
        self.theta = -np.pi / 4.
        self.area = np.pi * self.a * self.b
        self.true_flux = self.area

    def test_scalar_error_no_gain(self):

        # Scalar error, no gain.
        error = 1.
        flux, fluxerr = aperture_elliptical(self.data, self.xc, self.yc,
                                            self.a, self.b, self.theta,
                                            error=error)
        assert abs((flux - self.true_flux) / self.true_flux) < 0.01

        # Error should be exact to machine precision for apertures
        # with defined area.
        true_error = error * np.sqrt(self.area)
        assert_array_almost_equal_nulp(fluxerr, true_error, 1)

    def test_scalar_error_scalar_gain(self):

        # Scalar error, scalar gain.
        error = 1.
        gain = 1.
        flux, fluxerr = aperture_elliptical(self.data, self.xc, self.yc,
                                            self.a, self.b, self.theta,
                                            error=error, gain=gain)

        assert abs((flux - self.true_flux) / self.true_flux) < 0.01
        true_error = np.sqrt(error ** 2 * self.area + flux)
        assert_array_almost_equal_nulp(fluxerr, true_error, 1)

    def test_scalar_error_array_gain(self):

        # Scalar error, Array gain.
        error = 1.
        gain = np.ones(self.data.shape, dtype=np.float)
        flux, fluxerr = aperture_elliptical(self.data, self.xc, self.yc,
                                            self.a, self.b, self.theta,
                                            error=error, gain=gain)
        assert abs((flux - self.true_flux) / self.true_flux) < 0.01
        true_error = np.sqrt(error ** 2 * self.area + flux)
        assert abs((fluxerr - true_error) / true_error) < 0.01

    def test_array_error_no_gain(self):

        # Array error, no gain.
        error = np.ones(self.data.shape, dtype=np.float)
        flux, fluxerr = aperture_elliptical(self.data, self.xc, self.yc,
                                            self.a, self.b, self.theta,
                                            error=error)
        assert abs((flux - self.true_flux) / self.true_flux) < 0.01
        true_error = np.sqrt(self.area)
        assert abs((fluxerr - true_error) / true_error) < 0.01

    def test_array_error_scalar_gain(self):

        # Array error, scalar gain.
        error = np.ones(self.data.shape, dtype=np.float)
        gain = 1.
        flux, fluxerr = aperture_elliptical(self.data, self.xc, self.yc,
                                            self.a, self.b, self.theta,
                                            error=error, gain=gain)
        assert abs((flux - self.true_flux) / self.true_flux) < 0.01
        true_error = np.sqrt(self.area + flux)
        assert abs((fluxerr - true_error) / true_error) < 0.01

    def test_array_error_array_gain(self):

        # Array error, Array gain.
        error = np.ones(self.data.shape, dtype=np.float)
        gain = np.ones(self.data.shape, dtype=np.float)
        flux, fluxerr = aperture_elliptical(self.data, self.xc, self.yc,
                                            self.a, self.b, self.theta,
                                            error=error, gain=gain)
        assert abs((flux - self.true_flux) / self.true_flux) < 0.01
        true_error = np.sqrt(self.area + flux)
        assert abs((fluxerr - true_error) / true_error) < 0.01
