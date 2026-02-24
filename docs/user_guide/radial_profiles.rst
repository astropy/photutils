.. _radial_profiles:

Radial Profiles (`photutils.profiles`)
======================================

Introduction
------------

`photutils.profiles` provides tools to calculate radial profiles
(mean flux in concentric circular annular bins) and curves of growth
(cumulative flux within concentric circular, square, or elliptical
apertures). This page covers radial profiles. See
:ref:`curves_of_growth` for curves of growth.


Preliminaries
-------------

Let's start by making a synthetic image of a single source. Note that
there is no background in this image. One should background-subtract the
data before creating a radial profile or curve of growth.

    >>> import numpy as np
    >>> from astropy.modeling.models import Gaussian2D
    >>> from photutils.datasets import make_noise_image
    >>> gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    >>> yy, xx = np.mgrid[0:100, 0:100]
    >>> data = gmodel(xx, yy)
    >>> bkg_sig = 2.1
    >>> noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    >>> data += noise
    >>> error = np.zeros_like(data) + bkg_sig

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from astropy.visualization import simple_norm
    from photutils.datasets import make_noise_image

    # Create an artificial single source
    gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    bkg_sig = 2.1
    noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    data += noise
    error = np.zeros_like(data) + bkg_sig

    fig, ax = plt.subplots(figsize=(5, 5))
    norm = simple_norm(data, 'sqrt')
    ax.imshow(data, norm=norm, origin='lower')


Creating a Radial Profile
-------------------------

First, we'll use the `~photutils.centroids.centroid_2dg` function
to find the source centroid from the simulated image defined above::

    >>> from photutils.centroids import centroid_2dg
    >>> xycen = centroid_2dg(data)
    >>> print(xycen)  # doctest: +FLOAT_CMP
    [47.76934534 52.3884076 ]

We'll use this centroid position as the center of our radial profile.

We create a radial profile using the `~photutils.profiles.RadialProfile`
class. The radial bins are defined by inputting a 1D array of radii that
represent the radial *edges* of circular annulus apertures. The radial
spacing does not need to be constant. The input ``error`` array is the
uncertainty in the data values. The input ``mask`` array is a boolean
mask with the same shape as the data, where a `True` value indicates a
masked pixel::

    >>> from photutils.profiles import RadialProfile
    >>> edge_radii = np.arange(25)
    >>> rp = RadialProfile(data, xycen, edge_radii, error=error)

The output `~photutils.profiles.RadialProfile.radius` attribute values
are defined as the arithmetic means of the input radial-bins edges
(``radii``). Note that this is different from the input ``radii``, which
are the radial bin edges rather than centers::

    >>> print(rp.radii)  # doctest: +FLOAT_CMP
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
     24]

    >>> print(rp.radius)  # doctest: +FLOAT_CMP
    [ 0.5  1.5  2.5  3.5  4.5  5.5  6.5  7.5  8.5  9.5 10.5 11.5 12.5 13.5
     14.5 15.5 16.5 17.5 18.5 19.5 20.5 21.5 22.5 23.5]

The `~photutils.profiles.RadialProfile.profile` and
`~photutils.profiles.RadialProfile.profile_error` attributes contain the
output 1D `~numpy.ndarray` objects containing the radial profile and
propagated errors::

    >>> print(rp.profile)  # doctest: +FLOAT_CMP
    [ 4.30187860e+01  4.02502046e+01  3.57758011e+01  3.16071235e+01
      2.61511082e+01  2.10539746e+01  1.63701300e+01  1.16674718e+01
      8.12828014e+00  5.78962699e+00  3.59342666e+00  2.35353336e+00
      1.20355937e+00  7.67093923e-01  4.24650784e-01  8.67989701e-02
      5.11484374e-02 -9.82041768e-02  2.37482124e-02 -3.66602855e-02
      6.84802299e-02  1.72239596e-01 -3.86056497e-02  7.30423743e-02]

    >>> print(rp.profile_error)  # doctest: +FLOAT_CMP
    [1.18479813 0.68404352 0.52985783 0.4478116  0.39493271 0.35723008
     0.32860388 0.30591356 0.28735575 0.27181133 0.25854415 0.24704749
     0.23695963 0.22801451 0.22001149 0.21279603 0.20624688 0.20026744
     0.19477961 0.18971954 0.18503438 0.18068002 0.17661928 0.17282057]


Raw Data Profile
^^^^^^^^^^^^^^^^

The `~photutils.profiles.RadialProfile` class also includes
:attr:`~photutils.profiles.RadialProfile.data_radius` and
:attr:`~photutils.profiles.RadialProfile.data_profile` attributes that
can be used to plot the raw data profile. These attributes provide the
radii and values of the unmasked data points within the maximum radius
defined by the input radii.

Let's plot the raw data profile along with the radial profile and its
error bars:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D

    from photutils.centroids import centroid_2dg
    from photutils.datasets import make_noise_image
    from photutils.profiles import RadialProfile

    # Create an artificial single source
    gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    bkg_sig = 2.1
    noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    data += noise
    error = np.zeros_like(data) + bkg_sig

    # Find the source centroid
    xycen = centroid_2dg(data)

    # Create the radial profile
    edge_radii = np.arange(26)
    rp = RadialProfile(data, xycen, edge_radii, error=error)

    # Plot the radial profile
    fig, ax = plt.subplots(figsize=(8, 6))
    rp.plot(ax=ax, color='C0')
    rp.plot_error(ax=ax)
    ax.scatter(rp.data_radius, rp.data_profile, s=1, color='C1')

Normalization
^^^^^^^^^^^^^

If desired, the radial profile can be normalized using the
:meth:`~photutils.profiles.RadialProfile.normalize` method. By default
(``method='max'``), the profile is normalized such that its maximum
value is 1. Setting ``method='sum'`` can be used to normalize the
profile such that its sum (integral) is 1::

    >>> rp.normalize(method='max')

There is also a method to "unnormalize" the radial profile
back to the original values prior to running any calls to the
:meth:`~photutils.profiles.RadialProfile.normalize` method::

    >>> rp.unnormalize()

Plotting
^^^^^^^^

There are also convenience methods to plot the radial profile and
its error. These methods plot ``rp.radius`` versus ``rp.profile`` (with
``rp.profile_error`` as error bars). The ``label`` keyword can be used
to set the plot label.

.. doctest-skip::

    >>> rp.plot(label='Radial Profile')
    >>> rp.plot_error()

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from photutils.centroids import centroid_2dg
    from photutils.datasets import make_noise_image
    from photutils.profiles import RadialProfile

    # Create an artificial single source
    gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    bkg_sig = 2.1
    noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    data += noise
    error = np.zeros_like(data) + bkg_sig

    # Find the source centroid
    xycen = centroid_2dg(data)

    # Create the radial profile
    edge_radii = np.arange(26)
    rp = RadialProfile(data, xycen, edge_radii, error=error)

    # Plot the radial profile
    fig, ax = plt.subplots(figsize=(8, 6))
    rp.plot(ax=ax, label='Radial Profile')
    rp.plot_error(ax=ax)
    ax.legend()

The `~photutils.profiles.RadialProfile.apertures` attribute contains a
list of the apertures. Let's plot a few of the annulus apertures (the
6th, 11th, and 16th) for the `~photutils.profiles.RadialProfile`
instance on the data:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from astropy.visualization import simple_norm
    from photutils.centroids import centroid_2dg
    from photutils.datasets import make_noise_image
    from photutils.profiles import RadialProfile

    # Create an artificial single source
    gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    bkg_sig = 2.1
    noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    data += noise
    error = np.zeros_like(data) + bkg_sig

    # Find the source centroid
    xycen = centroid_2dg(data)

    # Create the radial profile
    edge_radii = np.arange(26)
    rp = RadialProfile(data, xycen, edge_radii, error=error)

    norm = simple_norm(data, 'sqrt')
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(data, norm=norm, origin='lower')
    rp.apertures[5].plot(ax=ax, color='C0', lw=2)
    rp.apertures[10].plot(ax=ax, color='C1', lw=2)
    rp.apertures[15].plot(ax=ax, color='C3', lw=2)

Fitting the profile with a 1D Gaussian or Moffat Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The radial profile can be fitted with either a 1D Gaussian
(`~astropy.modeling.functional_models.Gaussian1D`) or a 1D Moffat
(`~astropy.modeling.functional_models.Moffat1D`) model. The fitted
models are accessible via the
`~photutils.profiles.RadialProfile.gaussian_fit` and
`~photutils.profiles.RadialProfile.moffat_fit` attributes,
respectively::

    >>> rp.gaussian_fit  # doctest: +FLOAT_CMP
    <Gaussian1D(amplitude=42.25782121, mean=0., stddev=4.67512787)>

    >>> rp.moffat_fit  # doctest: +ELLIPSIS
    <Moffat1D(amplitude=..., x_0=0., gamma=..., alpha=...)>

The FWHM of each fitted model is stored in the
`~photutils.profiles.RadialProfile.gaussian_fwhm` and
`~photutils.profiles.RadialProfile.moffat_fwhm` attributes::

    >>> print(rp.gaussian_fwhm)  # doctest: +FLOAT_CMP
    11.009084813327846

    >>> print(rp.moffat_fwhm)  # doctest: +FLOAT_CMP
    10.868426520785151

The fitted models evaluated at the profile radius values are stored in
the `~photutils.profiles.RadialProfile.gaussian_profile` and
`~photutils.profiles.RadialProfile.moffat_profile` attributes.

Moffat profiles have broader wings than Gaussians and are often a
better representation of astronomical point-spread functions. Let's
plot both fitted models on the radial profile:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from photutils.centroids import centroid_2dg
    from photutils.datasets import make_noise_image
    from photutils.profiles import RadialProfile

    # Create an artificial single source
    gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    bkg_sig = 2.1
    noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    data += noise
    error = np.zeros_like(data) + bkg_sig

    # Find the source centroid
    xycen = centroid_2dg(data)

    # Create the radial profile
    edge_radii = np.arange(26)
    rp = RadialProfile(data, xycen, edge_radii, error=error)

    # Plot the radial profile with Gaussian and Moffat fits
    fig, ax = plt.subplots(figsize=(8, 6))
    rp.plot(ax=ax, label='Radial Profile')
    rp.plot_error(ax=ax)
    ax.plot(rp.radius, rp.gaussian_profile, label='Gaussian Fit')
    ax.plot(rp.radius, rp.moffat_profile, label='Moffat Fit')
    ax.legend()


API Reference
-------------

:doc:`../reference/profiles_api`
