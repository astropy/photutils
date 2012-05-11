# Licensed under a 3-clause BSD style license - see LICENSE.rst

def test_array_shapes():
    from pytest import raises
    import numpy as np
    from ..aperture import annulus_elliptical

    data = np.ones((10, 10), dtype=np.float)

    # Input single object and single aperture
    flux = annulus_elliptical(data, 5., 5., 2., 5., 5., np.pi / 4.)
    assert np.isscalar(flux) == True

    # One object, multiple apertures
    flux = annulus_elliptical(data, 2., 5.,
                       [[2.],[5.]], [[5.],[8.]], [[5.],[8.]], np.pi/2.)
    assert flux.shape == (2, 1)

    # Multiple objects, single aperture
    flux = annulus_elliptical(data, [2., 3., 4.], [5., 6., 7.],
                       2., 5., 5., np.pi/2.)
    assert flux.shape == (3,)

    # Multiple objects, multiple apertures
    flux = annulus_elliptical(data, [2., 3., 4.], [5., 6., 7.],
                       [2.,3.,4.], [5.,6.,7.], [5.,6.,7.], np.pi/2.)
    assert flux.shape == (3,)

    # Multiple objects, multiple apertures per object
    flux = annulus_elliptical(data, [2., 3., 4.], [5., 6., 7.],
                       [[2.], [5.]], [[5.], [8.]], [[5.], [8.]], np.pi/2.)
    assert flux.shape == (2, 3)

    # Mismatched number of objects and apertures
    with raises(ValueError):
        flux = annulus_elliptical(data, [2., 3., 4.], [5., 6., 7.],
                           [2.,3.], [5.,6.], [5.,6.], np.pi/2.)


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


def test_error_and_gain():

    from ..aperture import aperture_elliptical
    import numpy as np

    data = np.ones((40, 40), dtype=np.float)
    xc = 20.
    yc = 20.
    a = 10.
    b = 5.
    theta = -np.pi / 4.
    area = np.pi * a * b
    true_flux = area

    # Scalar error, no gain.
    error = 1.
    flux, fluxerr = aperture_elliptical(data, xc, yc, a, b, theta,
                                        error=error)
    assert abs((flux - true_flux) / true_flux) < 0.01
    true_error = error * np.sqrt(area)
    # Error should be exact to machine precision for apertures 
    # with defined area.
    assert abs((fluxerr - true_error) / true_error) < 0.01

    # Scalar error, scalar gain.
    error = 1.
    gain = 1.
    flux, fluxerr = aperture_elliptical(data, xc, yc, a, b, theta,
                                        error=error, gain=gain)
    assert abs((flux - true_flux) / true_flux) < 0.01
    true_error = np.sqrt(error ** 2 * area + flux)
    assert abs((fluxerr - true_error) / true_error) < 0.01

    # Scalar error, Array gain.
    error = 1.
    gain = np.ones(data.shape, dtype=np.float)
    flux, fluxerr = aperture_elliptical(data, xc, yc, a, b, theta,
                                        error=error, gain=gain)
    assert abs((flux - true_flux) / true_flux) < 0.01
    true_error = np.sqrt(error ** 2 * area + flux)
    assert abs((fluxerr - true_error) / true_error) < 0.01

    # Array error, no gain.
    error = np.ones(data.shape, dtype=np.float)
    flux, fluxerr = aperture_elliptical(data, xc, yc, a, b, theta,
                                        error=error)
    assert abs((flux - true_flux) / true_flux) < 0.01
    true_error = np.sqrt(area)
    assert abs((fluxerr - true_error) / true_error) < 0.01

    # Array error, scalar gain.
    error = np.ones(data.shape, dtype=np.float)
    gain = 1.
    flux, fluxerr = aperture_elliptical(data, xc, yc, a, b, theta,
                                        error=error, gain=gain)
    assert abs((flux - true_flux) / true_flux) < 0.01
    true_error = np.sqrt(area + flux)
    assert abs((fluxerr - true_error) / true_error) < 0.01

    # Array error, Array gain.
    error = np.ones(data.shape, dtype=np.float)
    gain = np.ones(data.shape, dtype=np.float)
    flux, fluxerr = aperture_elliptical(data, xc, yc, a, b, theta,
                                        error=error, gain=gain)
    assert abs((flux - true_flux) / true_flux) < 0.01
    true_error = np.sqrt(area + flux)
    assert abs((fluxerr - true_error) / true_error) < 0.01
