Elliptical Isophote Analysis (`photutils.isophote`)
===================================================

Introduction
------------

The `~photutils.isophote` package provides tools to fit elliptical
isophotes to a galaxy image. The isophotes are measured using the
iterative method described by `Jedrzejewski (1987, MNRAS 226, 747)
<https://ui.adsabs.harvard.edu/abs/1987MNRAS.226..747J/abstract>`_.
The fit gives the intensity, ellipticity, orientation, and center of
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

Let's show this initial ellipse:

.. doctest-skip::

    >>> import matplotlib.pyplot as plt
    >>> from astropy.visualization import simple_norm
    >>> from photutils.aperture import EllipticalAperture
    >>> aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
    ...                           geometry.sma * (1 - geometry.ellipticity),
    ...                           theta=geometry.orientation)
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
    from photutils.aperture import EllipticalAperture
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
    aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                              geometry.sma * (1 - geometry.ellipticity),
                              theta=geometry.orientation)
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
                   0.0  392.48758849279517 ...      0         0
    0.5346972612827552   268.2307594317041 ...     14         0
    0.5881669874110307  257.93087422324777 ...     10         0
    0.6469836861521338  247.13865766087417 ...     10         0
    0.7116820547673471  235.60867454024628 ...     10         0
                   ...                 ... ...    ...       ...
    111.19834626984479  0.3001491098553055 ...     50         2
    122.31818089682928 0.19406535096049538 ...     12         0
     134.5499989865122 0.09571824540441994 ...     22         0
    148.00499888516345 0.11318072729707639 ...     50         0
     162.8054987736798 0.05465160382491108 ...      3         1
    Length = 62 rows

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
orientation of 90 degrees. This works only for a galaxy that is close
to the center of the image. The galaxy in this example is 14 pixels
from the image center. The fit with the default geometry emits a
"No meaningful fit was possible" warning and returns an empty list:

.. doctest-skip::

    >>> isolist_default = Ellipse(data).fit_image()
    >>> len(isolist_default)
    0

A poor initial ellipticity or orientation can also prevent the fits
from converging, even for an image with a high signal-to-noise ratio.
If most of the isophotes have a stop code of 2 (see
:ref:`isophote-results`), run the fit again with a different initial
ellipticity or orientation.

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
center if the criterion exceeds the ``threshold`` parameter. The new
center is a whole-pixel position. Otherwise, the center is not
changed.

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
isophote. The first isophote in a list returned by
:meth:`~photutils.isophote.Ellipse.fit_image` is the central pixel of
the galaxy, with a semimajor axis length of zero.

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
  The isophote was extracted with the geometry of a previously fitted
  isophote.

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

    >>> measured = ellipse.measure_isophotes(loaded)  # doctest: +SKIP

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
intensities differ from the mean of the sample by more than
``clip_sigma`` standard deviations. The ``n_clip`` keyword is the
number of clipping iterations. The default is 0, which means that no
sigma clipping is performed. Sigma clipping is well suited to small
features such as stars, and their positions do not need to be known.
Let's fit the masked image with three iterations of 3-sigma clipping::

    >>> ellipse2 = Ellipse(masked_data, geometry=geometry)
    >>> isolist2 = ellipse2.fit_image(clip_sigma=3.0, n_clip=3)

Now let's compare the two fits using the largest distance between the
fitted centers and the galaxy center, for the isophotes with semimajor
axis lengths from 10 to 90 pixels::

    >>> for iso in (isolist1, isolist2):  # doctest: +SKIP
    ...     idx = (iso.sma > 10) & (iso.sma < 90)
    ...     shift = np.hypot(iso.x0[idx] - 160.0, iso.y0[idx] - 140.0)
    ...     print(np.round(shift.max(), 1))
    6.0
    0.7

Without rejection, the star and the companion pull the fitted centers
by up to 6 pixels, the ellipticity reaches 0.53, and the fit stops at
a semimajor axis length of about 92 pixels. With the mask and sigma
clipping, the centers stay within 1 pixel of the galaxy center and the
ellipticity stays between 0.39 and 0.42. The mask alone is not enough
for this image, because the star still disturbs the fit.

Let's plot the image with the masked region, and the residuals of the
model images of the two fits (see :ref:`isophote-model-image`):

.. plot::

    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D, Sersic2D
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
    isolist2 = ellipse2.fit_image(clip_sigma=3.0, n_clip=3)

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
isophotes that cross large masked regions.

Cosmic rays and defective pixels are removed very effectively by
combining sigma clipping with the ``'median'`` integration mode (see
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
(1 + ``step``), where ``step`` defaults to 0.1. With
``linear_growth=True``, ``step`` is instead an increment in pixels.
The ``min_sma`` and ``max_sma`` keywords limit the range of the
semimajor axis lengths::

    >>> isolist_linear = ellipse.fit_image(step=5.0, linear_growth=True,
    ...                                    max_sma=60.0)
    >>> print(isolist_linear.sma)
    [ 0.  5. 10. 15. 20. 25. 30. 35. 40. 45. 50. 55.]

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

The same approach helps when a fit stalls at some semimajor axis
length, e.g., where the isophotes of the galaxy change shape or twist
abruptly. The fit can be restarted beyond that length with a new
initial ellipse.


.. _isophote-harmonics:

Deviations from Ellipses
------------------------

The isophotes of real galaxies are not perfect ellipses. After the
ellipse of an isophote is fitted, the third and fourth harmonics of
its sampled intensities are measured. The ``a3`` and ``b3`` attributes
are the amplitudes of the :math:`\sin(3E)` and :math:`\cos(3E)` terms,
and ``a4`` and ``b4`` are the amplitudes of the :math:`\sin(4E)` and
:math:`\cos(4E)` terms, where :math:`E` is the angle from the
semimajor axis. The amplitudes are divided by the semimajor axis
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
    >>> isolist2 = Ellipse(data2).measure_isophotes(isolist)
    >>> mask = (isolist.intensity > 0) & (isolist2.intensity > 0)
    >>> ratio = isolist2.intensity[mask] / isolist.intensity[mask]
    >>> color = -2.5 * np.log10(ratio)

The returned `~photutils.isophote.IsophoteList` has one isophote for
each input ellipse, with the same geometry. The input can also be a
sequence of `~photutils.isophote.EllipseGeometry` objects, which is
useful for measuring an image along user-defined ellipses::

    >>> geometries = [EllipseGeometry(x0=160, y0=140, sma=sma, ellipticity=0.4,
    ...                               orientation=40.0 * u.deg)
    ...               for sma in (5.0, 10.0, 20.0)]
    >>> isolist3 = Ellipse(data).measure_isophotes(geometries)
    >>> print(isolist3.sma)
    [ 5. 10. 20.]

Additional Example Notebooks (Online)
-------------------------------------

Additional example notebooks showing examples with real data and
advanced usage are available online:

* `Basic example of the Ellipse fitting tool <https://github.com/astropy/photutils-datasets/blob/main/notebooks/isophote/isophote_example1.ipynb>`_

* `Running Ellipse with sigma clipping <https://github.com/astropy/photutils-datasets/blob/main/notebooks/isophote/isophote_example2.ipynb>`_

* `Building an image model from results obtained by Ellipse fitting <https://github.com/astropy/photutils-datasets/blob/main/notebooks/isophote/isophote_example3.ipynb>`_

* `Advanced Ellipse example: multiband photometry and masked arrays <https://github.com/astropy/photutils-datasets/blob/main/notebooks/isophote/isophote_example4.ipynb>`_


API Reference
-------------

:doc:`../reference/isophote_api`


.. toctree::
    :hidden:

    isophote_faq.rst
