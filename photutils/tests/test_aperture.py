# Licensed under a 3-clause BSD style license - see LICENSE.rst

def test_annulus_elliptical_basic():
    from ..aperture import annulus_elliptical
    import numpy as np
    from pytest import raises

    arr = np.ones((10, 10), dtype=np.float)

    # Input single object and single aperture
    flux = annulus_elliptical(arr, 5., 5., 2., 5., 5., np.pi / 4.)
    assert np.isscalar(flux) == True

    # One object, multiple apertures
    flux = annulus_elliptical(arr, 2., 5.,
                       [[2.],[5.]], [[5.],[8.]], [[5.],[8.]], np.pi/2.)
    assert flux.shape == (2, 1)

    # Multiple objects, single aperture
    flux = annulus_elliptical(arr, [2., 3., 4.], [5., 6., 7.],
                       2., 5., 5., np.pi/2.)
    assert flux.shape == (3,)

    # Multiple objects, multiple apertures
    flux = annulus_elliptical(arr, [2., 3., 4.], [5., 6., 7.],
                       [2.,3.,4.], [5.,6.,7.], [5.,6.,7.], np.pi/2.)
    assert flux.shape == (3,)

    # Multiple objects, multiple apertures per object
    flux = annulus_elliptical(arr, [2., 3., 4.], [5., 6., 7.],
                       [[2.], [5.]], [[5.], [8.]], [[5.], [8.]], np.pi/2.)
    assert flux.shape == (2, 3)

    # Mismatched number of objects and apertures
    with raises(ValueError):
        flux = annulus_elliptical(arr, [2., 3., 4.], [5., 6., 7.],
                           [2.,3.], [5.,6.], [5.,6.], np.pi/2.)


def test_annulus_elliptical_ones():
    from ..aperture import annulus_elliptical
    import numpy as np

    # An annulus fully in the array
    arr = np.ones((40, 40), dtype=np.float)
    a_in = 5.
    a_out = 10.
    b_out = 5.
    theta = np.pi / 4.
    flux = annulus_elliptical(arr, 20., 20., a_in, a_out, b_out, theta)
    true_flux = np.pi * (a_out * b_out - a_in ** 2 * b_out / a_out)
    assert abs((flux - true_flux) / true_flux) < 0.01

    # An annulus fully out of the array
    a_in = 5.
    a_out = 10.
    b_out = 5.
    theta = np.pi / 4.
    flux = annulus_elliptical(arr, 60., 60., a_in, a_out, b_out, theta)
    true_flux = np.pi * (a_out * b_out - a_in ** 2 * b_out / a_out)
    assert abs(flux) < 0.001
