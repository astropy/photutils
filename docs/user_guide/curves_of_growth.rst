.. _curves_of_growth:

Curves of Growth (`photutils.profiles`)
========================================

Introduction
------------

`photutils.profiles` provides tools to calculate radial profiles
(mean flux in concentric circular annular bins) and curves of growth
(cumulative flux within concentric circular, square, or elliptical
apertures). This page covers curves of growth. See
:ref:`radial_profiles` for radial profiles.


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


Creating a Curve of Growth
--------------------------

Now let's create a curve of growth using the
`~photutils.profiles.CurveOfGrowth` class. We use the simulated image
defined above.

First, we'll find the source centroid using the
`~photutils.centroids.centroid_2dg` function::

    >>> from photutils.centroids import centroid_2dg
    >>> xycen = centroid_2dg(data)
    >>> print(xycen)
    [47.76934534 52.3884076 ]

The curve of growth will be centered at our centroid position. It will
be computed over the radial range given by the input ``radii`` array::

    >>> from photutils.profiles import CurveOfGrowth
    >>> radii = np.arange(1, 26)
    >>> cog = CurveOfGrowth(data, xycen, radii, error=error)

Here, the `~photutils.profiles.CurveOfGrowth.radius` attribute values
are identical to the input ``radii``. Because these values are the radii
of the circular apertures used to measure the profile, they can be used
directly to measure the encircled energy/flux at a given radius. In
other words, they are the radial values that enclose the given flux::

    >>> print(cog.radius)
    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25]

The `~photutils.profiles.CurveOfGrowth.profile` and
`~photutils.profiles.CurveOfGrowth.profile_error` attributes contain
output 1D `~numpy.ndarray` objects containing the curve-of-growth
profile and propagated errors::

    >>> print(cog.profile)
    [ 135.14750208  514.49674293 1076.4617132  1771.53866121 2510.94382666
     3238.51695898 3907.08459943 4456.90125492 4891.00892262 5236.59326527
     5473.66400376 5643.72239573 5738.24972738 5803.31693644 5842.00525018
     5850.45854739 5855.76123671 5844.9631235  5847.72359025 5843.23189459
     5852.05251106 5875.32009699 5869.86235184 5880.64741302 5872.16333953]

    >>> print(cog.profile_error)
    [ 3.11062819  6.83127792 10.55630757 14.27852358 18.00330858 21.72296997
     25.44442261 29.16767174 32.88949473 36.61224389 40.33235754 44.0555786
     47.77789677 51.49947445 55.22141896 58.94275538 62.66630873 66.38766913
     70.10917057 73.83146673 77.55360137 81.27674225 84.99747377 88.71963388
     92.44198203]

Normalization
^^^^^^^^^^^^^

Typically, the normalized curve of growth is of interest, where
the profile is scaled so that its maximum value is 1 at the
largest input "radii" value. This normalization is commonly
used to calculate the encircled energy fraction at a specific
radius. The curve-of-growth profile can be normalized using the
:meth:`~photutils.profiles.CurveOfGrowth.normalize` method. By default
(``method='max'``), the profile is normalized such that its maximum
value is 1. Setting ``method='sum'`` can also be used to normalize the
profile such that the sum of its values is 1::

    >>> cog.normalize(method='max')

There is also a method to "unnormalize" the curve-of-growth profile
back to the original values prior to running any calls to the
:meth:`~photutils.profiles.CurveOfGrowth.normalize` method::

    >>> cog.unnormalize()

Plotting
^^^^^^^^

There are also convenience methods to plot the curve of growth and its
error. These methods plot ``cog.radius`` versus ``cog.profile`` (with
``cog.profile_error`` as error bars). The ``label`` keyword can be used
to set the plot label. Here, we plot the normalized curve of growth and
its error:

.. doctest-skip::

    >>> cog.normalize(method='max')
    >>> cog.plot(label='Curve of Growth')
    >>> cog.plot_error()

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from photutils.centroids import centroid_2dg
    from photutils.datasets import make_noise_image
    from photutils.profiles import CurveOfGrowth

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

    # Create the curve of growth
    radii = np.arange(1, 26)
    cog = CurveOfGrowth(data, xycen, radii, error=error)

    # Plot the curve of growth
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

    # Create the curve of growth
    radii = np.arange(1, 26)
    cog = CurveOfGrowth(data, xycen, radii, error=error)

    norm = simple_norm(data, 'sqrt')
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(data, norm=norm, origin='lower')
    cog.apertures[5].plot(ax=ax, color='C0', lw=2)
    cog.apertures[10].plot(ax=ax, color='C1', lw=2)
    cog.apertures[15].plot(ax=ax, color='C3', lw=2)


.. _encircled_energy:

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
better to have a curve-of-growth profile with high signal-to-noise
and more radial bins). Also, if the curve-of-growth profile is not
monotonically increasing, the interpolation may fail.

Let's compute the encircled energy values at a few radii for the curve
of growth we created above::

    >>> cog.normalize(method='max')
    >>> ee_rads = np.array([5, 7, 10, 15])
    >>> ee_vals = cog.calc_ee_at_radius(ee_rads)
    >>> ee_vals
    array([0.42698425, 0.66439702, 0.89047904, 0.99342893])

    >>> cog.calc_radius_at_ee(ee_vals)
    array([ 5., 7., 10., 15.])

Here we plot the encircled energy values.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from photutils.centroids import centroid_2dg
    from photutils.datasets import make_noise_image
    from photutils.profiles import CurveOfGrowth

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

    # Create the curve of growth
    radii = np.arange(1, 26)
    cog = CurveOfGrowth(data, xycen, radii, error=error)
    cog.normalize(method='max')
    ee_rads = np.array([5, 7, 10, 15])
    ee_vals = cog.calc_ee_at_radius(ee_rads)

    # Plot the curve of growth
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

    for ee_rad, ee_val in zip(ee_rads, ee_vals, strict=True):
        ax.text(ee_rad/2, ee_val, f'{ee_val:.2f}', ha='center', va='bottom')


Creating an Ensquared Curve of Growth
-------------------------------------

In addition to the encircled (circular) curve of growth, one can also
compute an ensquared curve of growth using concentric square apertures.
This is done using the `~photutils.profiles.EnsquaredCurveOfGrowth`
class, which accepts ``half_sizes`` (the half side lengths of the square
apertures) instead of ``radii``. The full side length of each square
aperture is ``2 * half_sizes``.

Let's create an ensquared curve of growth for the same source we created
above::

    >>> from photutils.profiles import EnsquaredCurveOfGrowth
    >>> half_sizes = np.arange(1, 26)
    >>> ecog = EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=error)

The ensquared curve of growth profile represents the total flux within
the square aperture as a function of the square half-size::

    >>> print(ecog.half_size)
    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25]

The `~photutils.profiles.EnsquaredCurveOfGrowth.profile` and
`~photutils.profiles.EnsquaredCurveOfGrowth.profile_error` attributes
contain output 1D `~numpy.ndarray` objects containing the ensquared
curve-of-growth profile and propagated errors::

    >>> print(ecog.profile)
    [ 171.51682931  640.93965938 1328.92447125 2143.49339746 2954.67817356
     3718.43574763 4357.67409109 4845.38866928 5199.93124392 5481.4874771
     5641.60852855 5752.43870109 5791.01446371 5819.46039533 5832.44691816
     5825.39056549 5834.14890108 5833.42003487 5852.25143102 5856.59671815
     5869.46227464 5872.94734439 5868.92606685 5849.96903786 5839.55875017]

    >>> print(ecog.profile_error)
    [  3.57264883   7.77582626  11.97677454  16.17723038  20.3774983
      24.57767465  28.77779953  32.97789259  37.17796463  41.37802205
      45.57806888  49.77810781  53.97814068  58.17816881  62.37819315
      66.57821441  70.77823316  74.9782498   79.17826468  83.37827806
      87.57829015  91.77830114  95.97831117 100.17832036 104.3783288 ]

Here, we plot the normalized ensquared curve of growth.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from photutils.centroids import centroid_2dg
    from photutils.datasets import make_noise_image
    from photutils.profiles import EnsquaredCurveOfGrowth

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

    # Create the ensquared curve of growth
    half_sizes = np.arange(1, 26)
    ecog = EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=error)
    ecog.normalize(method='max')

    # Plot the ensquared curve of growth
    fig, ax = plt.subplots(figsize=(8, 6))
    ecog.plot(ax=ax)
    ecog.plot_error(ax=ax)

We can also plot a few of the square apertures on the data.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from astropy.visualization import simple_norm
    from photutils.centroids import centroid_2dg
    from photutils.datasets import make_noise_image
    from photutils.profiles import EnsquaredCurveOfGrowth

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

    # Create the ensquared curve of growth
    half_sizes = np.arange(1, 26)
    ecog = EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=error)

    norm = simple_norm(data, 'sqrt')
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(data, norm=norm, origin='lower')
    ecog.apertures[5].plot(ax=ax, color='C0', lw=2)
    ecog.apertures[10].plot(ax=ax, color='C1', lw=2)
    ecog.apertures[15].plot(ax=ax, color='C3', lw=2)


Ensquared Energy
^^^^^^^^^^^^^^^^

Similar to the encircled energy, the ensquared energy is defined as the
fraction of total energy enclosed within a square aperture of a given
half-side length, and is commonly used to describe PSF characteristics.

The `~photutils.profiles.EnsquaredCurveOfGrowth` class provides
two convenience methods to calculate the ensquared energy at a
given half-size (or half-sizes) and the half-size corresponding to
the given ensquared energy (or energies). These methods are
:meth:`~photutils.profiles.EnsquaredCurveOfGrowth.calc_ee_at_half_size`
and
:meth:`~photutils.profiles.EnsquaredCurveOfGrowth.calc_half_size_at_ee`,
respectively.

Let's compute the ensquared energy values at a few half-sizes for the
ensquared curve of growth we created above::

    >>> ecog.normalize(method='max')
    >>> ee_half_sizes = np.array([3, 6, 9])
    >>> ee_vals = ecog.calc_ee_at_half_size(ee_half_sizes)
    >>> ee_vals
    array([0.22627897, 0.63314645, 0.88540403])

    >>> ecog.calc_half_size_at_ee(ee_vals)
    array([3., 6., 9.])

Here, we plot the ensquared energy values.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from photutils.centroids import centroid_2dg
    from photutils.datasets import make_noise_image
    from photutils.profiles import EnsquaredCurveOfGrowth

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

    # Create the ensquared curve of growth
    half_sizes = np.arange(1, 26)
    ecog = EnsquaredCurveOfGrowth(data, xycen, half_sizes, error=error)
    ecog.normalize(method='max')
    ee_half_sizes = np.array([3, 6, 9])
    ee_vals = ecog.calc_ee_at_half_size(ee_half_sizes)

    # Plot the ensquared curve of growth
    fig, ax = plt.subplots(figsize=(8, 6))
    ecog.plot(ax=ax, label='Ensquared Curve of Growth')
    ecog.plot_error(ax=ax)
    ax.legend()

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.vlines(ee_half_sizes, ymin, ee_vals, colors='C1', linestyles='dashed')
    ax.hlines(ee_vals, xmin, ee_half_sizes, colors='C1', linestyles='dashed')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    for ee_hs, ee_val in zip(ee_half_sizes, ee_vals, strict=True):
        ax.text(ee_hs/2, ee_val, f'{ee_val:.2f}', ha='center',
                va='bottom')


Creating an Elliptical Curve of Growth
--------------------------------------

One can also compute a curve of growth using concentric elliptical
apertures with a fixed axis ratio and orientation. This is done using
the `~photutils.profiles.EllipticalCurveOfGrowth` class, which accepts
``radii`` (the semimajor-axis lengths of the elliptical apertures),
``axis_ratio`` (the ratio of the semiminor axis to the semimajor axis,
``b / a``), and ``theta`` (the rotation angle from the positive ``x``
axis).

Let's create an elliptical curve of growth for an elliptical source
with an axis ratio of 0.5 and a rotation angle of 42 degrees::

    >>> import numpy as np
    >>> from astropy.modeling.models import Gaussian2D
    >>> from photutils.centroids import centroid_2dg
    >>> from photutils.datasets import make_noise_image
    >>> from photutils.profiles import EllipticalCurveOfGrowth
    >>> gmodel = Gaussian2D(42.1, 47.8, 52.4, 9.4, 4.7, np.deg2rad(42))
    >>> yy, xx = np.mgrid[0:100, 0:100]
    >>> data = gmodel(xx, yy)
    >>> bkg_sig = 2.1
    >>> noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    >>> data += noise
    >>> error = np.zeros_like(data) + bkg_sig

    >>> xycen = centroid_2dg(data)
    >>> radii = np.arange(1, 40)
    >>> ecog = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
    ...                                theta=np.deg2rad(42), error=error)

The elliptical curve of growth profile represents the total flux within
the elliptical aperture as a function of semimajor-axis length::

    >>> print(ecog.radius)
    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]

The `~photutils.profiles.EllipticalCurveOfGrowth.profile` and
`~photutils.profiles.EllipticalCurveOfGrowth.profile_error` attributes
contain output 1D `~numpy.ndarray` objects containing the elliptical
curve-of-growth profile and propagated errors::

    >>> print(ecog.profile)
    [   67.39762867   267.711181     588.47524874  1021.31994307
      1546.53867489  2152.12698084  2824.97954482  3541.64650208
      4284.363828    5040.93586551  5777.06397177  6488.33779084
      7179.10371288  7826.17773764  8418.08027957  8948.7310004
      9418.56480323  9810.0373925  10163.11467352 10477.42357537
     10731.89184641 10920.11723061 11092.34059512 11235.12552706
     11347.43721424 11454.03577845 11520.64656354 11555.89668261
     11571.27935302 11583.89774142 11605.79810845 11639.93073462
     11648.27293403 11660.34772581 11662.89065496 11643.07787619
     11630.36674411 11636.61537567 11636.60448497]

    >>> print(ecog.profile_error)
    [  1.82558544   4.4811361    7.21875141   9.88960965  12.44181885
      15.0767893   17.74361162  20.3959057   23.02040853  25.60886742
      28.25910457  30.93376937  33.54721095  36.14973269  38.78350009
      41.44751786  44.08722476  46.67911616  49.30783947  51.9760285
      54.61090354  57.21854088  59.84454217  62.48828381  65.14297876
      67.7591664   70.36843954  73.01750591  75.66115523  78.29244888
      80.90838345  83.53056346  86.19047387  88.82463241  91.43141788
      94.06294498  96.71101089  99.35484622 101.96990083]

Here, we plot the normalized elliptical curve of growth.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from photutils.centroids import centroid_2dg
    from photutils.datasets import make_noise_image
    from photutils.profiles import EllipticalCurveOfGrowth

    # Create an artificial elliptical source
    gmodel = Gaussian2D(42.1, 47.8, 52.4, 9.4, 4.7, np.deg2rad(42))
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    bkg_sig = 2.1
    noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    data += noise
    error = np.zeros_like(data) + bkg_sig

    # Find the source centroid
    xycen = centroid_2dg(data)

    # Create the elliptical curve of growth
    radii = np.arange(1, 40)
    ecog = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
                                   theta=np.deg2rad(42), error=error)
    ecog.normalize(method='max')

    # Plot the elliptical curve of growth
    fig, ax = plt.subplots(figsize=(8, 6))
    ecog.plot(ax=ax)
    ecog.plot_error(ax=ax)

We can also plot a few of the elliptical apertures on the data.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from astropy.visualization import simple_norm
    from photutils.centroids import centroid_2dg
    from photutils.datasets import make_noise_image
    from photutils.profiles import EllipticalCurveOfGrowth

    # Create an artificial elliptical source
    gmodel = Gaussian2D(42.1, 47.8, 52.4, 9.4, 4.7, np.deg2rad(42))
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    bkg_sig = 2.1
    noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    data += noise
    error = np.zeros_like(data) + bkg_sig

    # Find the source centroid
    xycen = centroid_2dg(data)

    # Create the elliptical curve of growth
    radii = np.arange(1, 40)
    ecog = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
                                   theta=np.deg2rad(42), error=error)

    norm = simple_norm(data, 'sqrt')
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(data, norm=norm, origin='lower')
    ecog.apertures[5].plot(ax=ax, color='C0', lw=2)
    ecog.apertures[10].plot(ax=ax, color='C1', lw=2)
    ecog.apertures[15].plot(ax=ax, color='C3', lw=2)


Elliptical Encircled Energy
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similar to the encircled energy for circular apertures, one can compute
the enclosed energy fraction within elliptical apertures. See the
`Encircled Energy`_ section above for details on normalization.

The `~photutils.profiles.EllipticalCurveOfGrowth` class
provides two convenience methods to calculate the enclosed
energy at a given semimajor-axis length (or lengths)
and the semimajor-axis length corresponding to the
given enclosed energy (or energies). These methods are
:meth:`~photutils.profiles.EllipticalCurveOfGrowth.calc_ee_at_radius`
and
:meth:`~photutils.profiles.EllipticalCurveOfGrowth.calc_radius_at_ee`,
respectively.

Let's compute the enclosed energy values at a few semimajor-axis lengths
for the elliptical curve of growth we created above::

    >>> ecog.normalize(method='max')
    >>> ee_rads = np.array([5, 10, 15, 20])
    >>> ee_vals = ecog.calc_ee_at_radius(ee_rads)
    >>> ee_vals
    array([0.13260338, 0.43222011, 0.72178335, 0.89835564])

    >>> ecog.calc_radius_at_ee(ee_vals)
    array([ 5., 10., 15., 20.])

Here we plot the enclosed energy values.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from photutils.centroids import centroid_2dg
    from photutils.datasets import make_noise_image
    from photutils.profiles import EllipticalCurveOfGrowth

    # Create an artificial elliptical source
    gmodel = Gaussian2D(42.1, 47.8, 52.4, 9.4, 4.7, np.deg2rad(42))
    yy, xx = np.mgrid[0:100, 0:100]
    data = gmodel(xx, yy)
    bkg_sig = 2.1
    noise = make_noise_image(data.shape, mean=0., stddev=bkg_sig, seed=0)
    data += noise
    error = np.zeros_like(data) + bkg_sig

    # Find the source centroid
    xycen = centroid_2dg(data)

    # Create the elliptical curve of growth
    radii = np.arange(1, 40)
    ecog = EllipticalCurveOfGrowth(data, xycen, radii, axis_ratio=0.5,
                                   theta=np.deg2rad(42), error=error)
    ecog.normalize(method='max')
    ee_rads = np.array([5, 10, 15, 20])
    ee_vals = ecog.calc_ee_at_radius(ee_rads)

    # Plot the elliptical curve of growth
    fig, ax = plt.subplots(figsize=(8, 6))
    ecog.plot(ax=ax, label='Elliptical Curve of Growth')
    ecog.plot_error(ax=ax)
    ax.legend()

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.vlines(ee_rads, ymin, ee_vals, colors='C1', linestyles='dashed')
    ax.hlines(ee_vals, xmin, ee_rads, colors='C1', linestyles='dashed')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    for ee_rad, ee_val in zip(ee_rads, ee_vals, strict=True):
        ax.text(ee_rad/2, ee_val, f'{ee_val:.2f}', ha='center',
                va='bottom')


API Reference
-------------

:doc:`../reference/profiles_api`
