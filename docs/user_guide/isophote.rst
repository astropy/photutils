Elliptical Isophote Analysis (`photutils.isophote`)
===================================================

Introduction
------------

The `~photutils.isophote` package provides tools to fit elliptical
isophotes to a galaxy image. The isophotes are measured using the
iterative method described by `Jedrzejewski (1987, MNRAS 226, 747)
<https://ui.adsabs.harvard.edu/abs/1987MNRAS.226..747J/abstract>`_.
See the documentation of the `~photutils.isophote.Ellipse` class and
:ref:`isophote-algorithm` for details about the algorithm. The fit
gives the intensity, ellipticity, orientation, and center of
the isophotes as a function of the semimajor axis length, along with
the deviations of the isophotes from perfect ellipses.

The method was designed for elliptical galaxies. It needs an intensity
that decreases outward over the fitted region (i.e., a well-defined
negative radial intensity gradient). The fits may not converge for
other types of objects (e.g., planetary nebulae).

The main classes are:

* `~photutils.isophote.EllipseGeometry` defines the ellipse where the
  fit starts.

* `~photutils.isophote.Ellipse` fits the isophotes of an image.

* `~photutils.isophote.IsophoteList` holds the results. It is a list
  of `~photutils.isophote.Isophote` objects, one for each semimajor
  axis length.

The shape of an ellipse is described by its ellipticity, defined as

.. math::

    \epsilon = 1 - \frac{b}{a}

where :math:`a` and :math:`b` are the semimajor and semiminor axis
lengths. The ellipticity is used instead of the eccentricity because
it relates better to the visual flattening of an ellipse. A slightly
flattened circle is easily seen to have an ellipticity of about 0.1.
The same ellipse has an eccentricity of 0.44, which is not obvious
from visual inspection. The two quantities are related by

.. math::

    e = \sqrt{1 - (1 - \epsilon)^2}

The orientation of an ellipse is the angle of its semimajor axis,
measured counterclockwise from the positive x axis of the image.


Getting Started
---------------

For this example, let's simulate an elliptical galaxy with a Sersic
profile. The galaxy has an ellipticity of 0.4 and an orientation of 40
degrees, and it is not at the center of the image::

    >>> import numpy as np
    >>> from astropy.modeling.models import Sersic2D
    >>> from photutils.datasets import make_noise_image

    >>> galaxy = Sersic2D(amplitude=10.0, r_eff=30.0, n=2.0, x_0=160.0,
    ...                   y_0=140.0, ellip=0.4, theta=np.deg2rad(40.0))
    >>> shape = (300, 300)
    >>> yy, xx = np.mgrid[0:300, 0:300]
    >>> noise = make_noise_image(shape, distribution='gaussian', mean=0.0,
    ...                          stddev=0.5, seed=0)
    >>> data = galaxy(xx, yy) + noise

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Sersic2D
    from astropy.visualization import simple_norm
    from photutils.datasets import make_noise_image

    galaxy = Sersic2D(amplitude=10.0, r_eff=30.0, n=2.0, x_0=160.0,
                      y_0=140.0, ellip=0.4, theta=np.deg2rad(40.0))
    shape = (300, 300)
    yy, xx = np.mgrid[0:300, 0:300]
    noise = make_noise_image(shape, distribution='gaussian', mean=0.0,
                             stddev=0.5, seed=0)
    data = galaxy(xx, yy) + noise

    norm = simple_norm(data, 'sqrt', percent=99.0)
    fig, ax = plt.subplots()
    ax.imshow(data, origin='lower', norm=norm)

The fit needs an initial ellipse, which is defined with the
`~photutils.isophote.EllipseGeometry` class. The orientation must be
an angular `~astropy.units.Quantity`. Here the initial ellipse has the
center of the galaxy, but its ellipticity and orientation differ from
those of the galaxy::

    >>> import astropy.units as u
    >>> from photutils.isophote import EllipseGeometry
    >>> geometry = EllipseGeometry(x0=160, y0=140, sma=20, ellipticity=0.3,
    ...                            orientation=20.0 * u.deg)

Let's show this initial ellipse. The
:meth:`~photutils.isophote.EllipseGeometry.to_aperture` method returns
the ellipse as an `~photutils.aperture.EllipticalAperture`, which can
be plotted:

.. doctest-skip::

    >>> import matplotlib.pyplot as plt
    >>> from astropy.visualization import simple_norm
    >>> aper = geometry.to_aperture()
    >>> norm = simple_norm(data, 'sqrt', percent=99.0)
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(data, origin='lower', norm=norm)
    >>> aper.plot(color='white')

.. plot::

    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Sersic2D
    from astropy.visualization import simple_norm
    from photutils.datasets import make_noise_image
    from photutils.isophote import EllipseGeometry

    galaxy = Sersic2D(amplitude=10.0, r_eff=30.0, n=2.0, x_0=160.0,
                      y_0=140.0, ellip=0.4, theta=np.deg2rad(40.0))
    shape = (300, 300)
    yy, xx = np.mgrid[0:300, 0:300]
    noise = make_noise_image(shape, distribution='gaussian', mean=0.0,
                             stddev=0.5, seed=0)
    data = galaxy(xx, yy) + noise

    geometry = EllipseGeometry(x0=160, y0=140, sma=20, ellipticity=0.3,
                               orientation=20.0 * u.deg)
    aper = geometry.to_aperture()
    norm = simple_norm(data, 'sqrt', percent=99.0)
    fig, ax = plt.subplots()
    ax.imshow(data, origin='lower', norm=norm)
    aper.plot(color='white')

Next, we create an instance of the `~photutils.isophote.Ellipse`
class, inputting the image and the initial ellipse geometry::

    >>> from photutils.isophote import Ellipse
    >>> ellipse = Ellipse(data, geometry=geometry)

To fit the isophotes, we run the
:meth:`~photutils.isophote.Ellipse.fit_image` method. The fit starts
at the semimajor axis length of the initial ellipse. It grows outward
until the intensity gradient can no longer be measured, and then
inward to the center of the galaxy::

    >>> isolist = ellipse.fit_image()

The result is an `~photutils.isophote.IsophoteList` object, sorted by
the semimajor axis length. Its attributes are arrays that hold the fit
values of the isophotes. Let's print the median ellipticity and
orientation of the isophotes with semimajor axis lengths from 10 to 90
pixels. They match the values of the simulated galaxy::

    >>> idx = (isolist.sma > 10) & (isolist.sma < 90)
    >>> print(np.round(np.median(isolist.ellipticity[idx]), 2))
    0.4
    >>> print(np.round(np.median(isolist.orientation[idx]), 0))
    40.0 deg

The :meth:`~photutils.isophote.IsophoteList.to_table` method returns
the isophote values as a table::

    >>> print(isolist.to_table())  # doctest: +SKIP
           sma              intensity      ... n_iter stop_code
                                           ...
    ------------------ ------------------- ... ------ ---------
                   0.0  392.40301224103547 ...      0         0
    0.5346972612827552   268.1139667720711 ...     12         0
    0.5881669874110307  257.98012406008917 ...     10         0
    0.6469836861521338  247.13824214053332 ...     10         0
    0.7116820547673471  235.60801120562274 ...     10         0
                   ...                 ... ...    ...       ...
    101.08940569985889  0.4622304207494331 ...     14         0
    111.19834626984479 0.29106071850491716 ...     10         0
    122.31818089682928 0.19270314093489685 ...     26         0
     134.5499989865122 0.17506574466746572 ...     17         0
    148.00499888516345 0.08092386426151292 ...     12         0
    Length = 61 rows

Let's plot the ellipticity, orientation, and the center x and y
position as a function of the semimajor axis length:

.. plot::

    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Sersic2D
    from photutils.datasets import make_noise_image
    from photutils.isophote import Ellipse, EllipseGeometry

    galaxy = Sersic2D(amplitude=10.0, r_eff=30.0, n=2.0, x_0=160.0,
                      y_0=140.0, ellip=0.4, theta=np.deg2rad(40.0))
    shape = (300, 300)
    yy, xx = np.mgrid[0:300, 0:300]
    noise = make_noise_image(shape, distribution='gaussian', mean=0.0,
                             stddev=0.5, seed=0)
    data = galaxy(xx, yy) + noise

    geometry = EllipseGeometry(x0=160, y0=140, sma=20, ellipticity=0.3,
                               orientation=20.0 * u.deg)
    ellipse = Ellipse(data, geometry=geometry)
    isolist = ellipse.fit_image()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
                                                 figsize=(8, 8))
    fig.subplots_adjust(hspace=0.35, wspace=0.35)

    ax1.errorbar(isolist.sma, isolist.ellipticity,
                 yerr=isolist.ellipticity_err, fmt='o', markersize=4)
    ax1.set_xlabel('Semimajor Axis Length (pix)')
    ax1.set_ylabel('Ellipticity')

    ax2.errorbar(isolist.sma, isolist.orientation.to_value(u.deg),
                 yerr=isolist.orientation_err.to_value(u.deg), fmt='o',
                 markersize=4)
    ax2.set_xlabel('Semimajor Axis Length (pix)')
    ax2.set_ylabel('Orientation (deg)')

    ax3.errorbar(isolist.sma, isolist.x0, yerr=isolist.x0_err, fmt='o',
                 markersize=4)
    ax3.set_xlabel('Semimajor Axis Length (pix)')
    ax3.set_ylabel('x0')

    ax4.errorbar(isolist.sma, isolist.y0, yerr=isolist.y0_err, fmt='o',
                 markersize=4)
    ax4.set_xlabel('Semimajor Axis Length (pix)')
    ax4.set_ylabel('y0')

The geometry is poorly constrained for the innermost isophotes, which
cover only a few pixels, and for the outermost isophotes, where the
noise dominates.

The ``intensity`` attribute is the mean intensity along each isophote.
Let's plot it to show the surface brightness profile of the galaxy:

.. plot::

    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Sersic2D
    from photutils.datasets import make_noise_image
    from photutils.isophote import Ellipse, EllipseGeometry

    galaxy = Sersic2D(amplitude=10.0, r_eff=30.0, n=2.0, x_0=160.0,
                      y_0=140.0, ellip=0.4, theta=np.deg2rad(40.0))
    shape = (300, 300)
    yy, xx = np.mgrid[0:300, 0:300]
    noise = make_noise_image(shape, distribution='gaussian', mean=0.0,
                             stddev=0.5, seed=0)
    data = galaxy(xx, yy) + noise

    geometry = EllipseGeometry(x0=160, y0=140, sma=20, ellipticity=0.3,
                               orientation=20.0 * u.deg)
    ellipse = Ellipse(data, geometry=geometry)
    isolist = ellipse.fit_image()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(isolist.sma, isolist.intensity, yerr=isolist.intensity_err,
                fmt='o', markersize=4)
    ax.set_yscale('log')
    ax.set_xlabel('Semimajor Axis Length (pix)')
    ax.set_ylabel('Intensity')


.. _isophote-initial-ellipse:

Defining the Initial Ellipse
----------------------------

The fitting algorithm cannot find where the galaxy is located in the
image. The center of the initial ellipse must be close to the center
of the galaxy for the fit to work. The initial ellipticity and
orientation should be reasonable estimates, but they do not need to be
accurate, as shown above.

If no geometry is input, `~photutils.isophote.Ellipse` uses a default
`~photutils.isophote.EllipseGeometry` at the center of the image, with
a semimajor axis length of 10 pixels, an ellipticity of 0.2, and an
orientation of 0 degrees. This works only for a galaxy that is close
to the center of the image. The galaxy in this example is 14 pixels
from the image center. The fit with the default geometry emits a
"No meaningful fit was possible" warning and returns an empty list:

.. doctest-skip::

    >>> isolist_default = Ellipse(data).fit_image()
    >>> len(isolist_default)
    0

A poor initial ellipticity or orientation can occasionally prevent
the fits from converging (see :ref:`isophote-geometry-corrections`).
If most of the isophotes have a stop code of 2 (see
:ref:`isophote-results`), run the fit again with a different initial
ellipticity or orientation. For images with a low signal-to-noise
ratio, see :ref:`isophote-low-snr`.

Finding the Galaxy Center
^^^^^^^^^^^^^^^^^^^^^^^^^

The :meth:`~photutils.isophote.EllipseGeometry.find_center` method
runs an "object centerer" that refines an approximate center. If it
finds the galaxy, it modifies the center of the
`~photutils.isophote.EllipseGeometry` object in place. Here the input
center is about 7 pixels from the galaxy center::

    >>> geometry2 = EllipseGeometry(x0=165, y0=135, sma=20, ellipticity=0.3,
    ...                             orientation=20.0 * u.deg)
    >>> geometry2.find_center(data, verbose=False)
    >>> print(geometry2.x0, geometry2.y0)
    160.0 140.0

The object centerer scans an 11x11 pixel window centered on the
input coordinates. At each position, it extracts two concentric,
roughly circular samples, an inner disk with a radius of about 3.5
pixels and an outer ring with a radius of about 8 pixels. It then
computes a signal-to-noise-like criterion from the mean intensities
(:math:`f_1` and :math:`f_2`) and the standard deviations
(:math:`\sigma_1` and :math:`\sigma_2`) of the two samples:

.. math::

    c = \frac{f_{1} - f_{2}}{{\sqrt{\sigma_{1}^{2} + \sigma_{2}^{2}}}}

The position where this criterion is a maximum is accepted as the new
center if the criterion exceeds the ``threshold`` parameter.
Otherwise, the center is not changed. The new center is a whole-pixel
position.

The default threshold value is 0.1. This value and the sizes of the
samples and the window were found by trial and error using a number of
both artificial and real galaxy images. Very flattened galaxies (an
ellipticity of about 0.7) cannot be detected by such a simple
algorithm. Increasing the threshold makes the object centerer
stricter, in the sense that it will not detect faint objects. The
criterion has no upper limit, so only a threshold value of `numpy.inf`
guarantees that the center coordinates are not modified.

The `~photutils.isophote.Ellipse` class does not run the object
centerer. To use it, call
:meth:`~photutils.isophote.EllipseGeometry.find_center` before the
`~photutils.isophote.EllipseGeometry` object is input to
`~photutils.isophote.Ellipse`.


.. _isophote-results:

Working with the Results
------------------------

Isophotes and Isophote Lists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An `~photutils.isophote.IsophoteList` is a list of
`~photutils.isophote.Isophote` objects, sorted by the semimajor axis
length. Each attribute of an `~photutils.isophote.Isophote` is a
single value. The same attribute of an
`~photutils.isophote.IsophoteList` is an array with one value for each
isophote. With the default ``min_sma`` of 0, the first isophote in a
list returned by :meth:`~photutils.isophote.Ellipse.fit_image` is the
central pixel of the galaxy, with a semimajor axis length of zero.

An `~photutils.isophote.IsophoteList` can be indexed and sliced like a
Python list. An index returns an `~photutils.isophote.Isophote` and a
slice returns an `~photutils.isophote.IsophoteList`. The
:meth:`~photutils.isophote.IsophoteList.get_closest` method returns
the isophote whose semimajor axis length is closest to the input
value::

    >>> isophote = isolist.get_closest(40.0)
    >>> print(np.round(isophote.sma, 1))
    39.0

Each isophote has a ``stop_code`` that tells how its fit ended:

* 0: The fit converged.

* 1: Fewer than the ``min_valid_fraction`` fraction of the sampled
  points are valid.

* 2: The maximum number of iterations was reached. The isophote has
  the best geometry that was found.

* 3: The harmonic fit failed or there are too few valid points to fit.
  Such isophotes are not included in the results of
  :meth:`~photutils.isophote.Ellipse.fit_image`.

* 4: The isophote was extracted with a fixed geometry, without
  fitting.

* 5: No meaningful gradient could be measured or the ellipse diverged.
  The isophote was extracted with the geometry of the adjacent
  isophote that was fitted just before it.

See the `~photutils.isophote.Isophote` documentation for the complete
descriptions.

Tables
^^^^^^

The ``properties`` attribute of an `~photutils.isophote.IsophoteList`
lists the names of all the isophote properties. The
``default_columns`` attribute lists the ones that
:meth:`~photutils.isophote.IsophoteList.to_table` returns by default.
Use the ``columns`` keyword to select the columns of the table::

    >>> tbl = isolist.to_table(columns=['sma', 'intensity', 'ellipticity'])
    >>> print(tbl.colnames)
    ['sma', 'intensity', 'ellipticity']

The table can be saved to a file and loaded back into an
`~photutils.isophote.IsophoteList` with the
:meth:`~photutils.isophote.IsophoteList.from_table` method. The
``sma``, ``x0``, ``y0``, ``ellipticity``, and ``orientation`` columns
are required. Use ``columns=isolist.properties`` to save every
property::

    >>> from astropy.table import QTable
    >>> from photutils.isophote import IsophoteList
    >>> tbl = isolist.to_table(columns=isolist.properties)
    >>> tbl.write('isophotes.ecsv')  # doctest: +SKIP
    >>> tbl = QTable.read('isophotes.ecsv')  # doctest: +SKIP
    >>> loaded = IsophoteList.from_table(tbl)

The ``orientation`` and ``orientation_err`` columns must have an
angular unit. The ECSV and FITS formats keep the units. Plain text
formats such as CSV do not, so set the unit after reading the table
(e.g., ``tbl['orientation'].unit = 'deg'``). Any other column that
has a unit must be dimensionless, or in pixels for the ``sma``,
``x0``, ``x0_err``, ``y0``, and ``y0_err`` columns.
``from_table`` raises an error, naming the column and the first
invalid row, if a ``sma`` value is negative, if an ``ellipticity``
value is not in the range from 0 up to (but not including) 1, or if a
required geometry value is not finite.

The loaded isophotes have no sample, so ``loaded.sample`` holds `None`
for each isophote. They can still be used to make a model image with
:meth:`~photutils.isophote.IsophoteList.make_model_image`. To measure
an image along them, pass the list to
:meth:`~photutils.isophote.Ellipse.measure_isophotes`::

    >>> measured = ellipse.measure_isophotes(loaded)

The ``step`` and ``linear_growth`` keywords of ``from_table`` set the
sampling annulus of each isophote for this measurement.

Sampled Data
^^^^^^^^^^^^

Each fitted `~photutils.isophote.Isophote` keeps the data that were
sampled along its ellipse in its ``sample`` attribute, which is an
`~photutils.isophote.EllipseSample` object. The ``angles``, ``radii``,
and ``intensities`` attributes of the sample are the polar angles (in
radians, measured from the semimajor axis), the polar radii, and the
intensities of the sampled points. The
:meth:`~photutils.isophote.Isophote.sampled_coordinates` method
returns the image coordinates of the sampled points.

Let's plot a few isophotes over the image, along with their sampled
intensities as a function of the angle. Each set of intensities is
divided by the mean intensity of its isophote. For a good fit, the
intensities show no systematic variation with the angle:

.. plot::

    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Sersic2D
    from astropy.visualization import simple_norm
    from photutils.datasets import make_noise_image
    from photutils.isophote import Ellipse, EllipseGeometry

    galaxy = Sersic2D(amplitude=10.0, r_eff=30.0, n=2.0, x_0=160.0,
                      y_0=140.0, ellip=0.4, theta=np.deg2rad(40.0))
    shape = (300, 300)
    yy, xx = np.mgrid[0:300, 0:300]
    noise = make_noise_image(shape, distribution='gaussian', mean=0.0,
                             stddev=0.5, seed=0)
    data = galaxy(xx, yy) + noise

    geometry = EllipseGeometry(x0=160, y0=140, sma=20, ellipticity=0.3,
                               orientation=20.0 * u.deg)
    ellipse = Ellipse(data, geometry=geometry)
    isolist = ellipse.fit_image()

    norm = simple_norm(data, 'sqrt', percent=99.0)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    ax1.imshow(data, origin='lower', norm=norm)
    for sma in (10, 20, 40, 80):
        isophote = isolist.get_closest(sma)
        x, y = isophote.sampled_coordinates()
        line, = ax1.plot(x, y)
        sample = isophote.sample
        ax2.plot(np.rad2deg(sample.angles),
                 sample.intensities / isophote.intensity, '.',
                 color=line.get_color(), markersize=3,
                 label=f'sma = {isophote.sma:.0f} pix')
    ax1.set_title('Isophotes')
    ax2.set_xlabel('Angle from the Semimajor Axis (deg)')
    ax2.set_ylabel('Intensity / Mean Intensity')
    ax2.legend()

Fitting a Single Isophote
^^^^^^^^^^^^^^^^^^^^^^^^^

The :meth:`~photutils.isophote.Ellipse.fit_isophote` method fits one
isophote at a given semimajor axis length and returns an
`~photutils.isophote.Isophote`::

    >>> isophote = ellipse.fit_isophote(30.0)
    >>> print(np.round(isophote.ellipticity, 2))
    0.4

Calling this method in a loop over a set of semimajor axis lengths
does not reproduce :meth:`~photutils.isophote.Ellipse.fit_image`.
:meth:`~photutils.isophote.Ellipse.fit_image` starts each fit from the
geometry of the closest isophote that was already fitted, and it
decides where to stop growing from the fitted isophotes. A loop of
independent fits is more likely to stop early or to give poor fits.


.. _isophote-rejecting-pixels:

Rejecting Contaminated Pixels
-----------------------------

Foreground stars, companion galaxies, cosmic rays, and defective
pixels add light that does not follow the elliptical isophotes of the
galaxy. They can distort the fitted ellipses and stop the fit early.
Such pixels can be excluded from the fits with a mask, with sigma
clipping, or with both.

To demonstrate, let's add a bright star and a companion galaxy to the
image::

    >>> from astropy.modeling.models import Gaussian2D
    >>> star = Gaussian2D(amplitude=60.0, x_mean=195.0, y_mean=150.0,
    ...                   x_stddev=2.5, y_stddev=2.5)
    >>> companion = Gaussian2D(amplitude=25.0, x_mean=110.0, y_mean=190.0,
    ...                        x_stddev=7.0, y_stddev=7.0)
    >>> contaminated = data + star(xx, yy) + companion(xx, yy)

First, we fit this image without rejecting any pixels::

    >>> isolist1 = Ellipse(contaminated, geometry=geometry).fit_image()

Masked pixels are excluded from the fits when the image is a
`~numpy.ma.MaskedArray`. Nonfinite (NaN or infinity) image values are
excluded in the same way, with a warning if they are not masked. A
mask is the best choice for a large contaminated region whose location
is known. Let's mask a circular region around the companion::

    >>> mask = np.hypot(xx - 110.0, yy - 190.0) < 25.0
    >>> masked_data = np.ma.masked_array(contaminated, mask=mask)

Sigma clipping rejects the points sampled along an ellipse whose
intensities are outliers. It is set with the ``sigma_clip`` keyword,
which takes an `~astropy.stats.SigmaClip` object that defines the
clipping threshold, the number of iterations, and the center and
standard deviation functions. The default is `None`, which means that
no sigma clipping is performed. Sigma clipping is well suited to
small features such as stars, and their positions do not need to be
known. Let's fit the masked image with three iterations of 3-sigma
clipping::

    >>> from astropy.stats import SigmaClip
    >>> sigma_clip = SigmaClip(sigma=3.0, maxiters=3)
    >>> ellipse2 = Ellipse(masked_data, geometry=geometry)
    >>> isolist2 = ellipse2.fit_image(sigma_clip=sigma_clip)

Now let's compare the two fits using the largest distance between the
fitted centers and the galaxy center, for the isophotes with semimajor
axis lengths from 10 to 90 pixels::

    >>> max_shifts = []
    >>> for iso in (isolist1, isolist2):
    ...     idx = (iso.sma > 10) & (iso.sma < 90)
    ...     shift = np.hypot(iso.x0[idx] - 160.0, iso.y0[idx] - 140.0)
    ...     max_shifts.append(shift.max())
    >>> print(np.round(max_shifts, 1))  # doctest: +SKIP
    [6.1 1.2]
    >>> print(max_shifts[0] > 5.0, max_shifts[1] < 1.5)
    True True

Without rejection, the star and the companion pull the fitted centers
by up to 6 pixels, the ellipticity reaches 0.55, and the fit stops at
a semimajor axis length of about 92 pixels. Over this range of
semimajor axis lengths, the mask and sigma clipping keep the centers
within about 1 pixel of the galaxy center and the ellipticity between
0.39 and 0.41. The mask alone is not enough for this image, because the
star still disturbs the fit. Sigma clipping alone removes most of the
distortion and keeps the centers within 2 pixels. The mask removes
the rest.

Let's plot the image with the masked region, and the residuals of the
model images of the two fits (see :ref:`isophote-model-image`):

.. plot::

    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D, Sersic2D
    from astropy.stats import SigmaClip
    from astropy.visualization import simple_norm
    from matplotlib.patches import Circle
    from photutils.datasets import make_noise_image
    from photutils.isophote import Ellipse, EllipseGeometry

    galaxy = Sersic2D(amplitude=10.0, r_eff=30.0, n=2.0, x_0=160.0,
                      y_0=140.0, ellip=0.4, theta=np.deg2rad(40.0))
    shape = (300, 300)
    yy, xx = np.mgrid[0:300, 0:300]
    noise = make_noise_image(shape, distribution='gaussian', mean=0.0,
                             stddev=0.5, seed=0)
    data = galaxy(xx, yy) + noise

    star = Gaussian2D(amplitude=60.0, x_mean=195.0, y_mean=150.0,
                      x_stddev=2.5, y_stddev=2.5)
    companion = Gaussian2D(amplitude=25.0, x_mean=110.0, y_mean=190.0,
                           x_stddev=7.0, y_stddev=7.0)
    contaminated = data + star(xx, yy) + companion(xx, yy)

    geometry = EllipseGeometry(x0=160, y0=140, sma=20, ellipticity=0.3,
                               orientation=20.0 * u.deg)
    isolist1 = Ellipse(contaminated, geometry=geometry).fit_image()

    mask = np.hypot(xx - 110.0, yy - 190.0) < 25.0
    masked_data = np.ma.masked_array(contaminated, mask=mask)
    ellipse2 = Ellipse(masked_data, geometry=geometry)
    isolist2 = ellipse2.fit_image(sigma_clip=SigmaClip(sigma=3.0,
                                                       maxiters=3))

    residual1 = contaminated - isolist1.make_model_image(shape)
    residual2 = contaminated - isolist2.make_model_image(shape)

    norm = simple_norm(contaminated, 'sqrt', percent=99.0)
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(14, 5))
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)
    ax1.imshow(contaminated, origin='lower', norm=norm)
    ax1.add_patch(Circle((110.0, 190.0), 25.0, edgecolor='white',
                         facecolor='none', linestyle='dashed'))
    ax1.set_title('Data and Masked Region')

    ax2.imshow(residual1, origin='lower', vmin=-2, vmax=2)
    ax2.set_title('Residual without Rejection')

    ax3.imshow(residual2, origin='lower', vmin=-2, vmax=2)
    ax3.set_title('Residual with Mask and Sigma Clipping')

The star and the companion remain in both residual images, because
they are not part of the elliptical model. Without rejection, the
model is distorted around them, and it ends where the fit stopped.

The ``n_flagged`` attribute gives the number of sampled points of
each isophote that were flagged, i.e., points that are outside of the
image, masked, nonfinite, or rejected by sigma clipping. If the
fraction of valid points of an isophote is smaller than the
``min_valid_fraction`` keyword (the default is 0.7), its fit stops
with a stop code of 1. A smaller value lets the fit continue through
isophotes that cross large masked regions, or that extend past the
edge of the image.

The ``fflag`` parameter of the STSDAS ``ellipse`` task is instead the
largest acceptable fraction of flagged points. The equivalent value
here is ``min_valid_fraction = 1 - fflag``. The STSDAS default of
``fflag = 0.5`` corresponds to ``min_valid_fraction=0.5``, which
accepts more flagged points than the default used here.

Tests with artificial images showed that cosmic rays and defective
pixels are removed very effectively by combining sigma clipping with
the ``'median'`` integration mode (see
:ref:`isophote-integration-modes`).


.. _isophote-controlling-fit:

Controlling the Fit
-------------------

The keywords of :meth:`~photutils.isophote.Ellipse.fit_image` control
where the isophotes are fitted, how the image is sampled, and which
ellipse parameters are fitted. The main keywords are described here.
See :meth:`~photutils.isophote.Ellipse.fit_image` for the complete
descriptions.

Semimajor Axis Range and Step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The fit starts at the semimajor axis length given by ``sma_start``,
which defaults to the ``sma`` value of the input geometry. The
starting isophote should have a good signal-to-noise ratio and a
clearly defined shape. It should not be at the center of the galaxy
or at its faint outskirts.

By default, the semimajor axis length grows and shrinks by a factor of
(1 + ``step``). With ``linear_growth=True``, ``step`` is instead an
increment in pixels. If these keywords are not set, the ``step`` and
``linear_growth`` attributes of the input
`~photutils.isophote.EllipseGeometry` are used, which are 0.1 and
`False` by default. The ``min_sma`` and ``max_sma`` keywords limit the range of the
semimajor axis lengths. Both limits are inclusive, so an isophote
whose semimajor axis length equals a limit is included::

    >>> isolist_linear = ellipse.fit_image(step=5.0, linear_growth=True,
    ...                                    max_sma=60.0)
    >>> print(isolist_linear.sma)
    [ 0.  5. 10. 15. 20. 25. 30. 35. 40. 45. 50. 55. 60.]

If ``max_sma`` is `None` (the default), the outward growth stops when
the relative error of the local intensity gradient exceeds
``max_gradient_rel_err`` (the default is 0.5) for two consecutive
isophotes. This is the main control that keeps the ellipses from
growing into regions where the signal-to-noise ratio is too low. If
``max_sma`` is set, the isophotes beyond that point are instead
extracted up to ``max_sma`` with a fixed geometry, without fitting.

The ``max_iterative_sma`` keyword sets where this noniterative mode
starts. Isophotes with larger semimajor axis lengths are extracted
with the geometry of the last fitted isophote and have a stop code of
4. This is useful for regions of very low surface brightness, where
the fits are unstable::

    >>> isolist_fixed = ellipse.fit_image(max_iterative_sma=50.0,
    ...                                   max_sma=100.0)
    >>> print(isolist_fixed.stop_code[-3:])
    [4 4 4]

.. _isophote-integration-modes:

Integration Modes
^^^^^^^^^^^^^^^^^

The ``integration_mode`` keyword sets how the image is sampled at
each point along an ellipse:

* ``'bilinear'`` (the default): bilinear interpolation of the pixels
  around the point.

* ``'nearest_neighbor'``: the value of the pixel nearest to the point.

* ``'mean'``: the mean of the pixels in an elliptical sector around
  the point.

* ``'median'``: the median of the pixels in an elliptical sector
  around the point.

The elliptical sectors of the ``'mean'`` and ``'median'`` modes are
pieces of an elliptical annulus whose width is set by ``step``. These
modes average more pixels for each sampled point, especially with a
larger ``step``, which increases the signal-to-noise ratio in the
outer regions of a galaxy.

The algorithm described in `Jedrzejewski (1987, MNRAS 226, 747)
<https://ui.adsabs.harvard.edu/abs/1987MNRAS.226..747J/abstract>`_
uses an elaborate, high-precision scheme to take into account the
partial pixels that lie along the sector boundaries. That scheme is
not implemented here. Instead, pixels at the sector boundaries are
either fully included or discarded, depending on the positions of
their centers relative to the sector. This design decision is based
on two arguments. It would be difficult to include partial pixels in a
median, and the simpler scheme is faster.

In the ``'mean'`` and ``'median'`` modes, a sector with 6 or fewer
valid pixels is sampled with bilinear interpolation instead, and the
whole ellipse is sampled with bilinear interpolation if its sectors
are smaller than one pixel. Bilinear interpolation was found to give
smoother samples in those cases.

.. _isophote-center-limit:

Fixing or Limiting the Geometry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the center, ellipticity, and orientation of each isophote
are fitted freely. The ``fix_center``, ``fix_ellipticity``, and
``fix_orientation`` keywords hold a parameter of every isophote at
the value of the input `~photutils.isophote.EllipseGeometry`.

In the outer, low surface-brightness regions of a galaxy, a nearby
star or companion galaxy can pull the fitted center away from the
galaxy nucleus. There are two ways to control this. Setting
``fix_center=True`` holds the center of every isophote at the center
of the input geometry. Setting ``max_center_shift`` to a distance in
pixels lets the center move freely within a circle of that radius
around the center of the input geometry. If the fit of an isophote
would move the center outside this circle, the center of that
isophote is placed on the circle and held fixed for the rest of its
fit, while the ellipticity and orientation continue to be fitted. The
next isophote starts from that position and may move its center back
inside the circle.

For the contaminated image above, where the fitted centers move by up
to 6 pixels without rejection, a limit of 2 pixels gives::

    >>> ellipse3 = Ellipse(contaminated, geometry=geometry)
    >>> isolist3 = ellipse3.fit_image(max_center_shift=2.0)
    >>> shift = np.hypot(isolist3.x0 - 160.0, isolist3.y0 - 140.0)
    >>> print(np.round(shift.max(), 2))
    2.0

The ``wander`` parameter of the STSDAS ``ellipse`` task is different.
It limits the x and y sizes of each individual center correction
during the iterations, not the total distance from the input center.
When a single correction exceeds ``wander``, STSDAS stops iterating
and extracts the following isophotes in the same direction (outward or
inward) without fitting them.

Convergence
^^^^^^^^^^^

The fit of an isophote converges when the largest harmonic amplitude
is smaller than ``convergence`` (the default is 0.05) times the rms
of the residuals of the harmonic fit, scaled by the square root of
the mean sector area. At least ``miniters`` iterations (the default is
10) are performed before this criterion is applied. If the fit has
not converged after ``maxiters`` iterations (the default is 50), the
isophote has the best geometry that was found and a stop code of 2.

Combining Fits
^^^^^^^^^^^^^^

A single set of keywords does not always suit the whole galaxy. The
inner and outer regions can be fitted separately and the two lists
combined. Here the outer region is fitted with a larger step and the
``'median'`` integration mode, starting from its own initial
ellipse::

    >>> inner = ellipse.fit_image(max_sma=40.0)
    >>> geometry_outer = EllipseGeometry(x0=160, y0=140, sma=50,
    ...                                  ellipticity=0.4,
    ...                                  orientation=40.0 * u.deg)
    >>> ellipse_outer = Ellipse(data, geometry=geometry_outer)
    >>> outer = ellipse_outer.fit_image(min_sma=40.0, step=0.2,
    ...                                 integration_mode='median')
    >>> combined = inner + outer
    >>> combined.sort()

The ``+`` operator concatenates two lists, and the
:meth:`~photutils.isophote.IsophoteList.sort` method sorts the
isophotes by their semimajor axis lengths.

The same approach helps when a fit stalls at some semimajor axis
length, e.g., where the isophotes of the galaxy change shape or twist
abruptly. The fit can be restarted beyond that length with a new
initial ellipse.


.. _isophote-harmonics:

Deviations from Ellipses
------------------------

The isophotes of real galaxies are not perfect ellipses. After the
ellipse of an isophote is fitted, the third and fourth harmonics of
its sampled intensities are measured. As in the STSDAS ``ellipse``
task, each harmonic is fit to the residual intensities after the
first and second harmonics, and then the third harmonic, are
subtracted. The ``a3`` and ``b3`` attributes are the amplitudes of
the :math:`\sin(3\phi)` and :math:`\cos(3\phi)` terms, and ``a4`` and
``b4`` are the amplitudes of the :math:`\sin(4\phi)` and
:math:`\cos(4\phi)` terms, where :math:`\phi` is the polar angle
measured counterclockwise from the semimajor axis. The amplitudes are divided by the semimajor axis
length and by the absolute value of the local radial intensity
gradient, so they can be compared with each other and between
isophotes. Their errors are in the ``a3_err``, ``b3_err``,
``a4_err``, and ``b4_err`` attributes.

The ``b4`` amplitude is the most used. It is positive for disky
(kite-like) isophotes and negative for boxy isophotes. To demonstrate,
let's add a faint, thin disk along the major axis of the simulated
galaxy and fit the image::

    >>> disk = Gaussian2D(amplitude=10.0, x_mean=160.0, y_mean=140.0,
    ...                   x_stddev=30.0, y_stddev=4.0,
    ...                   theta=np.deg2rad(40.0))
    >>> disky = data + disk(xx, yy)
    >>> isolist_disky = Ellipse(disky, geometry=geometry).fit_image()
    >>> idx = (isolist_disky.sma > 15) & (isolist_disky.sma < 50)
    >>> print(np.round(np.median(isolist_disky.b4[idx]), 2))
    0.03

Let's plot the four amplitudes of the disky galaxy, and those of the
original galaxy for comparison:

.. plot::

    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D, Sersic2D
    from photutils.datasets import make_noise_image
    from photutils.isophote import Ellipse, EllipseGeometry

    galaxy = Sersic2D(amplitude=10.0, r_eff=30.0, n=2.0, x_0=160.0,
                      y_0=140.0, ellip=0.4, theta=np.deg2rad(40.0))
    shape = (300, 300)
    yy, xx = np.mgrid[0:300, 0:300]
    noise = make_noise_image(shape, distribution='gaussian', mean=0.0,
                             stddev=0.5, seed=0)
    data = galaxy(xx, yy) + noise

    disk = Gaussian2D(amplitude=10.0, x_mean=160.0, y_mean=140.0,
                      x_stddev=30.0, y_stddev=4.0,
                      theta=np.deg2rad(40.0))
    disky = data + disk(xx, yy)

    geometry = EllipseGeometry(x0=160, y0=140, sma=20, ellipticity=0.3,
                               orientation=20.0 * u.deg)
    isolist = Ellipse(data, geometry=geometry).fit_image()
    isolist_disky = Ellipse(disky, geometry=geometry).fit_image()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                             sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.15, wspace=0.1)
    for ax, name in zip(axes.ravel(), ('a3', 'b3', 'a4', 'b4')):
        for iso, label in ((isolist, 'Original'), (isolist_disky, 'Disky')):
            idx = (iso.sma > 5) & (iso.sma < 80)
            ax.errorbar(iso.sma[idx], getattr(iso, name)[idx],
                        yerr=getattr(iso, f'{name}_err')[idx], fmt='o',
                        markersize=4, label=label)
        ax.axhline(0.0, color='gray', linewidth=1)
        ax.set_title(name)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(-0.03, 0.06)
    for ax in axes[1]:
        ax.set_xlabel('Semimajor Axis Length (pix)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Amplitude')

The ``b4`` amplitude of the disky galaxy peaks at about 0.04 near a
semimajor axis length of 30 pixels, well above its error of about
0.007. The other amplitudes, and all four amplitudes of the original
galaxy, scatter around zero.


.. _isophote-model-image:

Model Images and Residuals
--------------------------

The :meth:`~photutils.isophote.IsophoteList.make_model_image` method
builds a model image of the galaxy from an
`~photutils.isophote.IsophoteList`. The intensity and the geometry of
the isophotes are interpolated onto a fine grid of semimajor axis
lengths, and the interpolated ellipses are drawn on the output image.
Subtracting the model from the data shows the features that do not
follow the elliptical isophotes::

    >>> model_image = isolist.make_model_image(data.shape)
    >>> residual = data - model_image

Let's plot the original data, the elliptical model image, and the
residual image:

.. plot::

    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Sersic2D
    from astropy.visualization import simple_norm
    from photutils.datasets import make_noise_image
    from photutils.isophote import Ellipse, EllipseGeometry

    galaxy = Sersic2D(amplitude=10.0, r_eff=30.0, n=2.0, x_0=160.0,
                      y_0=140.0, ellip=0.4, theta=np.deg2rad(40.0))
    shape = (300, 300)
    yy, xx = np.mgrid[0:300, 0:300]
    noise = make_noise_image(shape, distribution='gaussian', mean=0.0,
                             stddev=0.5, seed=0)
    data = galaxy(xx, yy) + noise

    geometry = EllipseGeometry(x0=160, y0=140, sma=20, ellipticity=0.3,
                               orientation=20.0 * u.deg)
    ellipse = Ellipse(data, geometry=geometry)
    isolist = ellipse.fit_image()

    model_image = isolist.make_model_image(data.shape)
    residual = data - model_image

    norm = simple_norm(data, 'sqrt', percent=99.0)
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(14, 5))
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)
    ax1.imshow(data, origin='lower', norm=norm)
    ax1.set_title('Data')

    ax2.imshow(model_image, origin='lower', norm=norm)
    ax2.set_title('Ellipse Model')

    ax3.imshow(residual, origin='lower', vmin=-2, vmax=2)
    ax3.set_title('Residual')

The residuals are consistent with the noise, except at the central
pixels, where the intensity of this galaxy changes steeply within a
pixel.

Pixels that are outside of the largest isophote have no model value
and are set to the ``fill_value`` keyword, which defaults to 0. For an
image that is not background subtracted, set it to the background
level of the image. With ``high_harmonics=True``, the third and fourth
harmonics (see :ref:`isophote-harmonics`) are added to the model, so
that it follows the deviations of the isophotes from ellipses. The
``sma_interval`` keyword is the spacing of the grid of semimajor axis
lengths. Larger values are faster but can leave gaps in the model.


.. _isophote_fixed_ellipses:

Measuring Along Fixed Ellipses
------------------------------

The :meth:`~photutils.isophote.Ellipse.measure_isophotes` method
measures an image along a given set of ellipses without fitting them.
This is similar to running the STSDAS ``ellipse`` task with an input
ellipse table (the ``inellip`` parameter). A common use is to measure
an image in a second band along the isophotes fit to a reference image,
so that both intensity profiles are measured along the same elliptical
paths (e.g., for a color profile). The two images must be aligned on the
same pixel grid.

Here we simulate a second-band image of the same galaxy that is half as
bright, and measure it along the isophotes fit above. The outermost
isophotes are dominated by noise and can have negative intensities, so
the color is calculated only where both intensities are positive::

    >>> noise2 = make_noise_image(shape, distribution='gaussian', mean=0.0,
    ...                           stddev=0.5, seed=5678)
    >>> data2 = 0.5 * galaxy(xx, yy) + noise2
    >>> isolist_band2 = Ellipse(data2).measure_isophotes(isolist)
    >>> positive = (isolist.intensity > 0) & (isolist_band2.intensity > 0)
    >>> ratio = (isolist_band2.intensity[positive]
    ...          / isolist.intensity[positive])
    >>> color = -2.5 * np.log10(ratio)

The returned `~photutils.isophote.IsophoteList` has one isophote for
each input ellipse, with the same geometry. The input can also be a
sequence of `~photutils.isophote.EllipseGeometry` objects, which is
useful for measuring an image along user-defined ellipses::

    >>> geometries = [EllipseGeometry(x0=160, y0=140, sma=sma, ellipticity=0.4,
    ...                               orientation=40.0 * u.deg)
    ...               for sma in (5.0, 10.0, 20.0)]
    >>> isolist_user = Ellipse(data).measure_isophotes(geometries)
    >>> print(isolist_user.sma)
    [ 5. 10. 20.]


.. _isophote-low-snr:

Low Signal-to-Noise Images
--------------------------

The fitting algorithm corrects the ellipse parameters by dividing the
harmonic amplitudes by the local radial gradient (see
:ref:`isophote-geometry-corrections`). The gradient is measured from
the mean intensities of two ellipses whose semimajor axis lengths
differ by only 10%. For a low surface brightness galaxy, or any image
where the noise of a pixel is comparable to the galaxy intensity, the
difference between these two means can be smaller than its noise. The
corrections are then dominated by noise. The ellipse wanders or
diverges, and the fit stops after a few isophotes. If the first
isophote cannot be fit, :meth:`~photutils.isophote.Ellipse.fit_image`
emits a "No meaningful fit was possible" warning and returns an empty
list.

No setting of the fit parameters solves this reliably. In particular,
the ``max_gradient_rel_err`` parameter only controls where the outward
growth stops, so it does not help the first isophote. Multiplying the
image by a constant does not help either, because the algorithm does
not depend on the intensity scale.

Whether the first isophote can be fit depends on the noise along one
ellipse. A fit that fails therefore often succeeds from a slightly
different starting semimajor axis length (the ``sma_start`` parameter),
e.g., one that is 10% larger or smaller.

The opposite failure is also possible.
:meth:`~photutils.isophote.Ellipse.fit_image` is not a detection
test. If there is no detectable galaxy at the starting position, the
fit can follow the noise and return isophotes without a warning.
Check that the galaxy is detected before fitting it, e.g., with the
tools described in :ref:`image_segmentation`.

The most effective solution is to increase the signal-to-noise ratio
of the pixels by smoothing the image before fitting it, e.g., with a
Gaussian kernel with a standard deviation of 2 to 3 pixels:

.. doctest-skip::

    >>> from astropy.convolution import Gaussian2DKernel, convolve
    >>> from photutils.isophote import Ellipse
    >>> smoothed = convolve(data, Gaussian2DKernel(2.0))
    >>> ellipse = Ellipse(smoothed, geometry=geometry)
    >>> isolist = ellipse.fit_image()

Fits of the smoothed image are much more likely to succeed, and they
reach larger semimajor axis lengths.

Smoothing lowers the central intensities and makes the isophotes
rounder at semimajor axis lengths comparable to the kernel size. To
measure intensities that are not affected by the smoothing, use the
:meth:`~photutils.isophote.Ellipse.measure_isophotes` method to
measure the original image along the ellipses that were fit to the
smoothed image (see :ref:`isophote_fixed_ellipses`):

.. doctest-skip::

    >>> isolist_data = Ellipse(data).measure_isophotes(isolist)

If the ellipse geometry of the galaxy is already known, the isophotes
can also be extracted without fitting them. With
``max_iterative_sma=0``, all the isophotes have the geometry of the
input `~photutils.isophote.EllipseGeometry`. The ``max_sma`` parameter
must then be set, because otherwise the outward growth stops where the
gradient error exceeds ``max_gradient_rel_err``:

.. doctest-skip::

    >>> geometry_known = EllipseGeometry(x0=160, y0=140, sma=20,
    ...                                  ellipticity=0.4,
    ...                                  orientation=40.0 * u.deg)
    >>> ellipse = Ellipse(data, geometry=geometry_known)
    >>> isolist = ellipse.fit_image(max_iterative_sma=0.0, max_sma=60.0)

The ``'mean'`` and ``'median'`` integration modes with a larger
``step`` (e.g., 0.3) average more pixels for each sample point (see
:ref:`isophote-integration-modes`). They can help for moderately noisy
images, but they are less reliable than smoothing the image.


.. _isophote-algorithm:

Algorithm Notes
---------------

The overall fitting procedure is described in the Notes section of the
`~photutils.isophote.Ellipse` class. This section gives more details
about some of its steps.

.. _isophote-geometry-corrections:

Geometry Corrections
^^^^^^^^^^^^^^^^^^^^

The basic elliptical isophote fitting algorithm, as described in
`Jedrzejewski (1987, MNRAS 226, 747)
<https://ui.adsabs.harvard.edu/abs/1987MNRAS.226..747J/abstract>`_,
computes corrections for the current ellipse's geometrical parameters
by essentially "projecting" the fitted harmonic amplitudes onto the
image plane:

.. math::

    {\delta}_{X0} = \frac {-B_{1}} {I'}

.. math::

    {\delta}_{Y0} = \frac {-A_{1} (1 - {\epsilon})} {I'}

.. math::

    {\delta}_{\epsilon} = \frac {-2 B_{2} (1 - {\epsilon})} {I' a_0}

.. math::

    {\delta}_{\Theta} = \frac {2 A_{2} (1 - {\epsilon})} {I' a_0 [(1 - {\epsilon}) ^ 2 - 1]}

where :math:`{\delta}_{X0}` and :math:`{\delta}_{Y0}` are the center
corrections along the major and minor axes of the ellipse,
respectively, :math:`\epsilon` is the ellipticity, :math:`\Theta` is
the orientation, :math:`A_i` and :math:`B_i` are the coefficients
of :math:`\sin(i\phi)` and :math:`\cos(i\phi)` in the harmonic fit
to the sampled intensities as a function of the polar angle
:math:`\phi` measured from the major axis (Jedrzejewski 1987 writes
these terms with the eccentric anomaly :math:`E`, but this
implementation and the STSDAS ``ellipse`` task use the polar angle),
and :math:`I'` is the derivative of the intensity along
the major axis direction evaluated at a semimajor axis length of
:math:`a_0`. The center corrections are then rotated by the
orientation to give the corrections along the image x and y axes.

The orientation correction is roughly inversely proportional to the
ellipticity of the current ellipse. If that ellipticity is about half
of the galaxy ellipticity, the correction overshoots by a factor of
about two, and the orientation would oscillate between two values
without the ellipticity ever being corrected. Therefore, a correction
that reverses the correction that the previous iteration made to the
same parameter is halved. This also keeps the center and ellipticity
corrections of noisy isophotes from oscillating across the solution.
The STSDAS ``ellipse`` task does not damp the corrections.

.. _isophote-gradient:

Radial Gradient
^^^^^^^^^^^^^^^

The radial intensity gradient is the most critical quantity computed
by the fitting algorithm. As can be seen from the above formulae,
small :math:`I'` values lead to large values for the correction terms.
Thus, :math:`I'` errors may lead to large fluctuations in these terms,
when :math:`I'` itself is small. This usually happens at the fainter,
outer regions of galaxy images. `Busko (1996, ASPC 101, 139)
<https://ui.adsabs.harvard.edu/abs/1996ASPC..101..139B/abstract>`_
found by numerical experiments that the precision to which a given
ellipse can be fitted is related to the relative error in the local
radial gradient.

Because of the gradient's critical role, the algorithm has a number of
features to allow its estimation even under difficult conditions. The
default gradient computation, the one used by the algorithm when it
first starts to fit a new isophote, is based on two intensity samples.
The first is extracted at the current ellipse and the second at a
similar ellipse with a 10% larger semimajor axis.

If the gradient so estimated is not meaningful, the algorithm extracts
the second sample again, this time using a 20% larger semimajor axis.
In this context, a meaningful gradient is negative and is either
steeper than one third of a reference gradient or measured with a
relative error smaller than 20%. The reference is the gradient of the
previous iteration of the fit of the same isophote. If no reference is
available, which is the case for the first iteration of each
isophote, the reference is the current gradient plus its error, so
any negative gradient is meaningful.

If still no meaningful gradient can be measured, the algorithm uses
the reference gradient multiplied by a factor of 0.8, which makes it
shallower. The current gradient error then cannot be computed and is
NaN.

The factor of 0.8 is roughly what is expected between two isophotes
from semimajor-axis geometrical-sampling steps of 10% to 20% and a de
Vaucouleurs law or the inner region of an exponential disk (within
about 5 effective radii). These rules follow the STSDAS ``ellipse``
task, with two differences. First, the STSDAS task keeps the reference
from one isophote to the next, starting from a fixed guess. Second,
the STSDAS task never accepts a gradient that is shallower than one
third of the reference. Here such a gradient is accepted if its
relative error is smaller than 20%. Otherwise, a single gradient that
is much steeper than the following ones would replace them for
several iterations.

The use of approximate gradient estimators may seem in contradiction
with the fact that isophote fitting errors depend on gradient error,
as well as with the fact that the algorithm itself is so sensitive to
the gradient value. The rationale behind the use of approximate
estimators, however, is based on the fact that the gradient value is
used only to compute increments, not the ellipse parameters
themselves. Approximate estimators are useful along the first steps in
the iteration sequence, in particular when local image contamination
(stars, defects, etc.) might make it difficult to find the correct
path toward the solution. However, if the gradient is still not well
determined at convergence, the subsequent error computations, and the
algorithm's behavior from that point on, will take the fact into
account properly. For instance, the 3rd and 4th harmonic amplitude
errors depend on the gradient relative error, and if this is not
computable at the current isophote, the algorithm uses a fixed
relative gradient error of 0.8 in order to generate sensible estimates
for those harmonic errors.

Error Estimation
^^^^^^^^^^^^^^^^

Most parameters computed directly at each isophote have their errors
computed by standard error propagation. Errors in the ellipse geometry
parameters, on the other hand, cannot be estimated in the same way,
since these parameters are not computed directly but result from a
number of updates from a starting guess value. An error analysis based
on numerical experiments (`Busko 1996, ASPC 101, 139
<https://ui.adsabs.harvard.edu/abs/1996ASPC..101..139B/abstract>`_)
showed that the best error estimators for these geometrical parameters
can be found by simply "projecting" the harmonic amplitude errors that
come from the least-squares covariance matrix by the same formulae
that are used to "project" the associated parameter updates (see
:ref:`isophote-geometry-corrections`). In other words, errors for the
ellipse center, ellipticity, and orientation are computed by those
formulae, but replacing the least-squares amplitudes by their errors.
This is empirical and difficult to justify in terms of any theoretical
error analysis, but it produces sensible error estimators in practice.
The geometry was not fitted for an isophote with a stop code of 4
(e.g., beyond ``max_iterative_sma`` or from
:meth:`~photutils.isophote.Ellipse.measure_isophotes`), or with a stop
code of 1 and no iterations, so its geometry errors are NaN.

Flux Accuracy
^^^^^^^^^^^^^

Each isophote also holds the integrated flux and the number of valid
pixels inside its ellipse (the ``total_flux_ellipse`` and
``n_pixels_ellipse`` attributes) and inside the circle with the same
center and a radius equal to the semimajor axis length
(``total_flux_circle`` and ``n_pixels_circle``). The integrated fluxes
and areas computed by `~photutils.isophote.Ellipse` were checked
against results produced by the IRAF ``noao.digiphot.apphot`` tasks
``phot`` and ``polyphot``, using artificial images. Quantities
computed by `~photutils.isophote.Ellipse` match the reference ones to
within 0.1% in all tested cases.


API Reference
-------------

:doc:`../reference/isophote_api`
