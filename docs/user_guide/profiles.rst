.. _profiles:

Radial Profiles (`photutils.profiles`)
======================================

Introduction
------------

`photutils.profiles` provides tools to calculate radial profiles and
curves of growth using concentric circular apertures.


Preliminaries
-------------

Let’s start by making a synthetic image of a single source. Note that
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

    # create an artificial single source
    gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    bkg_sig = 2.1
    noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    data += noise
    error = np.zeros_like(data) + bkg_sig

    fig, ax = plt.subplots(figsize=(5, 5))
    norm = simple_norm(data, 'sqrt')
    ax.imshow(data, norm=norm)


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
class. The radial bins are defined by inputing a 1D array of radii that
represent the radial *edges* of circular annulus apertures. The radial
spacing does not need to be constant. The input ``error`` array is the
uncertainty in the data values. The input ``mask`` array is a boolean
mask with the same shape as the data, where a `True` value indicates a
masked pixel::

    >>> from photutils.profiles import RadialProfile
    >>> edge_radii = np.arange(25)
    >>> rp = RadialProfile(data, xycen, edge_radii, error=error, mask=None)

The output `~photutils.profiles.RadialProfile.radius` attribute values
are defined as the arithmetic means of the input radial-bins edges
(``radii``). Note this is different from the input ``radii``, which
represents the radial bin edges::

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
that can be used to plot the raw data profile. These methods return the
radii and values of the data points within the maximum radius defined by
the input radii.

Let's plot the raw data profile along with the radial profile and its
error bars:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D

    from photutils.centroids import centroid_2dg
    from photutils.datasets import make_noise_image
    from photutils.profiles import RadialProfile

    # create an artificial single source
    gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    bkg_sig = 2.1
    noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    data += noise
    error = np.zeros_like(data) + bkg_sig

    # find the source centroid
    xycen = centroid_2dg(data)

    # create the radial profile
    edge_radii = np.arange(26)
    rp = RadialProfile(data, xycen, edge_radii, error=error, mask=None)

    # plot the radial profile
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

    >> rp.normalize(method='max')

There is also a method to "unnormalize" the radial profile
back to the original values prior to running any calls to the
:meth:`~photutils.profiles.RadialProfile.normalize` method::

    >> rp.unnormalize()

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

    # create an artificial single source
    gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    bkg_sig = 2.1
    noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    data += noise
    error = np.zeros_like(data) + bkg_sig

    # find the source centroid
    xycen = centroid_2dg(data)

    # create the radial profile
    edge_radii = np.arange(26)
    rp = RadialProfile(data, xycen, edge_radii, error=error, mask=None)

    # plot the radial profile
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

    # create an artificial single source
    gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    bkg_sig = 2.1
    noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    data += noise
    error = np.zeros_like(data) + bkg_sig

    # find the source centroid
    xycen = centroid_2dg(data)

    # create the radial profile
    edge_radii = np.arange(26)
    rp = RadialProfile(data, xycen, edge_radii, error=error, mask=None)

    norm = simple_norm(data, 'sqrt')
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(data, norm=norm)
    rp.apertures[5].plot(ax=ax, color='C0', lw=2)
    rp.apertures[10].plot(ax=ax, color='C1', lw=2)
    rp.apertures[15].plot(ax=ax, color='C3', lw=2)

Fitting the profile with a 1D Gaussian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now let's fit a 1D Gaussian to the radial profile and return the
`~astropy.modeling.functional_models.Gaussian1D` model using the
`~photutils.profiles.RadialProfile.gaussian_fit` attribute. The returned
value is a 1D Gaussian model fit to the radial profile::

    >>> rp.gaussian_fit  # doctest: +FLOAT_CMP
    <Gaussian1D(amplitude=42.25782121, mean=0., stddev=4.67512787)>

The FWHM of the fitted 1D Gaussian model is stored in the
`~photutils.profiles.RadialProfile.gaussian_fwhm` attribute::

    >>> print(rp.gaussian_fwhm)  # doctest: +FLOAT_CMP
    11.009084813327846

The 1D Gaussian model evaluated at the profile radius values is stored
in the `~photutils.profiles.RadialProfile.gaussian_profile` attribute::

    >>> print(rp.gaussian_profile)  # doctest: +FLOAT_CMP
    [4.20168369e+01 4.01377829e+01 3.66280188e+01 3.19303371e+01
     2.65903224e+01 2.11530856e+01 1.60751071e+01 1.16698171e+01
     8.09290139e+00 5.36135385e+00 3.39292860e+00 2.05118576e+00
     1.18458244e+00 6.53515081e-01 3.44410166e-01 1.73390913e-01
     8.33886095e-02 3.83104412e-02 1.68134796e-02 7.04900913e-03
     2.82311500e-03 1.08008788e-03 3.94747727e-04 1.37819354e-04]

Finally, let's plot the fitted 1D Gaussian model for the
:class:`~photutils.profiles.RadialProfile` radial profile:

.. plot::
   :include-source:

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from photutils.centroids import centroid_2dg
    from photutils.datasets import make_noise_image
    from photutils.profiles import RadialProfile

    # create an artificial single source
    gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    bkg_sig = 2.1
    noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    data += noise
    error = np.zeros_like(data) + bkg_sig

    # find the source centroid
    xycen = centroid_2dg(data)

    # create the radial profile
    edge_radii = np.arange(26)
    rp = RadialProfile(data, xycen, edge_radii, error=error, mask=None)

    # plot the radial profile
    fig, ax = plt.subplots(figsize=(8, 6))
    rp.plot(ax=ax, label='Radial Profile')
    rp.plot_error(ax=ax)
    ax.plot(rp.radius, rp.gaussian_profile, label='Gaussian Fit')
    ax.legend()


Creating a Curve of Growth
--------------------------

Now let's create a curve of growth using the
`~photutils.profiles.CurveOfGrowth` class. We use the simulated image
defined above and the same source centroid.

The curve of growth will be centered at our centroid position. It will
be computed over the radial range given by the input ``radii`` array::

    >>> from photutils.profiles import CurveOfGrowth
    >>> radii = np.arange(1, 26)
    >>> cog = CurveOfGrowth(data, xycen, radii, error=error, mask=None)

Here, the `~photutils.profiles.CurveOfGrowth.radius` attribute values
are identical to the input ``radii``. Because these values are the radii
of the circular apertures used to measure the profile, they can be used
directly to measure the encircled energy/flux at a given radius. In
other words, they are the radial values that enclose the given flux::

    >>> print(cog.radius)  # doctest: +FLOAT_CMP
    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25]

The `~photutils.profiles.CurveOfGrowth.profile` and
`~photutils.profiles.CurveOfGrowth.profile_error` attributes contain
output 1D `~numpy.ndarray` objects containing the curve-of-growth
profile and propagated errors::

    >>> print(cog.profile)  # doctest: +FLOAT_CMP
    [ 135.14750208  514.49674293 1076.4617132  1771.53866121 2510.94382666
     3238.51695898 3907.08459943 4456.90125492 4891.00892262 5236.59326527
     5473.66400376 5643.72239573 5738.24972738 5803.31693644 5842.00525018
     5850.45854739 5855.76123671 5844.9631235  5847.72359025 5843.23189459
     5852.05251106 5875.32009699 5869.86235184 5880.64741302 5872.16333953]

    >>> print(cog.profile_error)  # doctest: +FLOAT_CMP
    [ 3.72215309  7.44430617 11.16645926 14.88861235 18.61076543 22.33291852
     26.05507161 29.7772247  33.49937778 37.22153087 40.94368396 44.66583704
     48.38799013 52.11014322 55.8322963  59.55444939 63.27660248 66.99875556
     70.72090865 74.44306174 78.16521482 81.88736791 85.609521   89.33167409
     93.05382717]

Normalization
^^^^^^^^^^^^^

If desired, the curve-of-growth profile can be normalized using the
:meth:`~photutils.profiles.CurveOfGrowth.normalize` method. By default
(``method='max'``), the profile is normalized such that its maximum
value is 1. Setting ``method='sum'`` can also be used to normalize the
profile such that its sum (integral) is 1::

    >> cog.normalize(method='max')

There is also a method to "unnormalize" the radial profile
back to the original values prior to running any calls to the
:meth:`~photutils.profiles.CurveOfGrowth.normalize` method::

    >> cog.unnormalize()

Plotting
^^^^^^^^

There are also convenience methods to plot the curve of growth and its
error. These methods plot ``cog.radius`` versus ``cog.profile`` (with
``cog.profile_error`` as error bars). The ``label`` keyword can be used
to set the plot label.

.. doctest-skip::

    >>> rp.plot(label='Curve of Growth')
    >>> rp.plot_error()

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from photutils.centroids import centroid_2dg
    from photutils.datasets import make_noise_image
    from photutils.profiles import CurveOfGrowth

    # create an artificial single source
    gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    bkg_sig = 2.1
    noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    data += noise
    error = np.zeros_like(data) + bkg_sig

    # find the source centroid
    xycen = centroid_2dg(data)

    # create the radial profile
    radii = np.arange(1, 26)
    cog = CurveOfGrowth(data, xycen, radii, error=error, mask=None)

    # plot the radial profile
    fig, ax = plt.subplots(figsize=(8, 6))
    cog.plot(ax=ax, label='Curve of Growth')
    cog.plot_error(ax=ax)
    ax.legend()

The `~photutils.profiles.CurveOfGrowth.apertures` attribute contains a
list of the apertures. Let's plot a few of the circular apertures (the
6th, 11th, and 16th) on the data:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from astropy.visualization import simple_norm
    from photutils.centroids import centroid_2dg
    from photutils.datasets import make_noise_image
    from photutils.profiles import CurveOfGrowth

    # create an artificial single source
    gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    bkg_sig = 2.1
    noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    data += noise
    error = np.zeros_like(data) + bkg_sig

    # find the source centroid
    xycen = centroid_2dg(data)

    # create the radial profile
    radii = np.arange(1, 26)
    cog = CurveOfGrowth(data, xycen, radii, error=error, mask=None)

    norm = simple_norm(data, 'sqrt')
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(data, norm=norm)
    cog.apertures[5].plot(ax=ax, color='C0', lw=2)
    cog.apertures[10].plot(ax=ax, color='C1', lw=2)
    cog.apertures[15].plot(ax=ax, color='C3', lw=2)


Encircled Energy
^^^^^^^^^^^^^^^^

Often, one is interested in the encircled energy (or flux) within
a given radius, where the encircled energy is generally expressed
as a normalized value between 0 and 1. If the curve of growth is
monotonically increasing and normalized such that its maximum value is
1 for an infinitely large radius, then the encircled energy is simply
the value of the curve of growth at a given radius. To achieve this, one
can input a normalized version of the ``data`` array (e.g., a normalized
PSF) to the `~photutils.profiles.CurveOfGrowth` class. One can also
use the :meth:`~photutils.profiles.CurveOfGrowth.normalize` method to
normalize the curve of growth profile to be 1 at the largest input
``radii`` value.

If the curve of growth is normalized, the encircled energy at
a given radius is simply the value of the curve of growth at
that radius. The `~photutils.profiles.CurveOfGrowth` class
provides two convenience methods to calculate the encircled
energy at a given radius (or radii) and the radius corresponding
to the given encircled energy (or energies). These methods are
:meth:`~photutils.profiles.CurveOfGrowth.calc_ee_at_radius` and
:meth:`~photutils.profiles.CurveOfGrowth.calc_radius_at_ee`,
respectively. They are implemented as interpolation functions using
the calculated curve-of-growth profile. The accuracy of these methods
is dependent on the quality of the curve-of-growth profile (e.g., it's
better to have a curve-of-growth profile with high signal to noise
and more radial bins). Also, if the curve-of-growth profile is not
monotonically increasing, the interpolation may fail.

Let's compute the encircled energy values at a few radii for the curve
of growth we created above::

    >>> cog.normalize(method='max')
    >>> ee_rads = np.array([5, 10, 15])
    >>> ee_vals = cog.calc_ee_at_radius(ee_rads)  # doctest: +FLOAT_CMP
    >>> ee_vals
    array([0.42698425, 0.89047904, 0.99342893])

    >>> cog.calc_radius_at_ee(ee_vals)  # doctest: +FLOAT_CMP
    array([ 5., 10., 15.])

Here we plot the encircled energy values.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from photutils.centroids import centroid_2dg
    from photutils.datasets import make_noise_image
    from photutils.profiles import CurveOfGrowth

    # create an artificial single source
    gmodel = Gaussian2D(42.1, 47.8, 52.4, 4.7, 4.7, 0)
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    bkg_sig = 2.1
    noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    data += noise
    error = np.zeros_like(data) + bkg_sig

    # find the source centroid
    xycen = centroid_2dg(data)

    # create the radial profile
    radii = np.arange(1, 26)
    cog = CurveOfGrowth(data, xycen, radii, error=error, mask=None)
    cog.normalize(method='max')
    ee_rads = np.array([5, 10, 15])
    ee_vals = cog.calc_ee_at_radius(ee_rads)

    # plot the radial profile
    fig, ax = plt.subplots(figsize=(8, 6))
    cog.plot(ax=ax, label='Curve of Growth')
    cog.plot_error(ax=ax)
    ax.legend()

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.vlines(ee_rads, ymin, ee_vals, colors='C1', linestyles='dashed')
    ax.hlines(ee_vals, xmin, ee_rads, colors='C1', linestyles='dashed')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    for ee_rad, ee_val in zip(ee_rads, ee_vals):
        ax.text(ee_rad/2, ee_val, f'{ee_val:.2f}', ha='center', va='bottom')


API Reference
-------------

:doc:`../reference/profiles_api`
