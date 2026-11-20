Elliptical Isophote Analysis (`photutils.isophote`)
===================================================

Introduction
------------

The `~photutils.isophote` package provides tools to fit elliptical
isophotes to a galaxy image. The isophotes in the image are measured
using an iterative method described by `Jedrzejewski (1987, MNRAS 226,
747)
<https://ui.adsabs.harvard.edu/abs/1987MNRAS.226..747J/abstract>`_.
See the documentation of the :class:`~photutils.isophote.Ellipse`
class for details about the algorithm. Please also see the
:ref:`isophote-faq`.

Getting Started
---------------

For this example, let's create a simple simulated galaxy image::

    >>> import numpy as np
    >>> from astropy.modeling.models import Gaussian2D
    >>> from photutils.datasets import make_noise_image

    >>> g = Gaussian2D(100.0, 75, 75, 20, 12, theta=np.deg2rad(40.0))
    >>> ny = nx = 150
    >>> y, x = np.mgrid[0:ny, 0:nx]
    >>> noise = make_noise_image((ny, nx), distribution='gaussian', mean=0.0,
    ...                          stddev=2.0, seed=1234)
    >>> data = g(x, y) + noise

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from photutils.datasets import make_noise_image

    g = Gaussian2D(100.0, 75, 75, 20, 12, theta=np.deg2rad(40.0))
    ny = nx = 150
    y, x = np.mgrid[0:ny, 0:nx]
    noise = make_noise_image((ny, nx), distribution='gaussian', mean=0.0,
                             stddev=2.0, seed=1234)
    data = g(x, y) + noise
    fig, ax = plt.subplots()
    ax.imshow(data, origin='lower')

We must provide the elliptical isophote fitter with an initial ellipse
to be fitted. This ellipse geometry is defined with the
`~photutils.isophote.EllipseGeometry` class. Here we'll define an
initial ellipse whose position angle is offset from the data::

    >>> from photutils.isophote import EllipseGeometry
    >>> geometry = EllipseGeometry(x0=75, y0=75, sma=20, eps=0.5,
    ...                            pa=np.deg2rad(20.0))

Let's show this initial ellipse guess:

.. doctest-skip::

    >>> import matplotlib.pyplot as plt
    >>> from photutils.aperture import EllipticalAperture
    >>> aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
    ...                           geometry.sma * (1 - geometry.eps),
    ...                           theta=geometry.pa)
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(data, origin='lower')
    >>> aper.plot(color='white')

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from photutils.aperture import EllipticalAperture
    from photutils.datasets import make_noise_image
    from photutils.isophote import EllipseGeometry

    g = Gaussian2D(100.0, 75, 75, 20, 12, theta=np.deg2rad(40.0))
    ny = nx = 150
    y, x = np.mgrid[0:ny, 0:nx]
    noise = make_noise_image((ny, nx), distribution='gaussian', mean=0.0,
                             stddev=2.0, seed=1234)
    data = g(x, y) + noise

    geometry = EllipseGeometry(x0=75, y0=75, sma=20, eps=0.5,
                               pa=np.deg2rad(20.0))
    aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                              geometry.sma * (1 - geometry.eps),
                              theta=geometry.pa)
    fig, ax = plt.subplots()
    ax.imshow(data, origin='lower')
    aper.plot(color='white')

Next, we create an instance of the `~photutils.isophote.Ellipse`
class, inputting the data to be fitted and the initial ellipse
geometry object::

    >>> from photutils.isophote import Ellipse
    >>> ellipse = Ellipse(data, geometry=geometry)

To perform the elliptical isophote fit, we run the
:meth:`~photutils.isophote.Ellipse.fit_image` method::

    >>> isolist = ellipse.fit_image()

The result is a list of isophotes as an
`~photutils.isophote.IsophoteList` object, whose attributes are the
fit values for each `~photutils.isophote.Isophote` sorted by the
semimajor axis length. Let's print the fit position angles
(radians)::

    >>> print(isolist.pa)  # doctest: +SKIP
    [0.00000000e+00 2.81367008e-02 2.51508622e-02 1.73093341e-03
     3.10376907e+00 3.03272800e+00 2.91084574e+00 2.86751015e+00
     3.26302191e-02 3.26302191e-02 3.26302191e-02 2.83168988e-02
     3.11013380e+00 8.97640073e-02 1.04325194e-01 2.50113062e-01
     3.14000525e-01 3.80371206e-01 4.56646548e-01 7.06300858e-01
     1.14535190e+00 1.00610573e+00 9.86403064e-01 6.13496115e-01
     6.13496115e-01 6.52734459e-01 7.40866519e-01 7.64426009e-01
     6.98377494e-01 6.77763437e-01 6.81948740e-01 6.28697050e-01
     6.63983445e-01 6.91218272e-01 6.96058510e-01 6.93662476e-01
     6.86066367e-01 6.78819039e-01 6.90736903e-01 7.06300858e-01
     6.87365632e-01 6.78462107e-01 6.91947810e-01 6.88590345e-01
     6.89747001e-01 7.04595575e-01 6.98160861e-01 7.01980599e-01
     6.86722137e-01 6.86722137e-01 6.86722137e-01 7.09026632e-01
     7.09026632e-01 6.56348640e-01 6.56348640e-01]

We can also show the isophote values as a table, which is again sorted
by the semimajor axis length (``sma``)::

    >>> print(isolist.to_table())  # doctest: +SKIP
           sma                 intens         ... n_iter stop_code
                                              ...
    ------------------ ---------------------- ... ------ ---------
                   0.0     103.36486267669721 ...      0         0
    0.5346972612827552     101.85758347160414 ...     10         0
    0.5881669874110307     101.68719303286363 ...     10         0
    0.6469836861521338      101.5058988661627 ...     11         0
    0.7116820547673471     101.37369354568321 ...     10         0
                   ...                    ... ...    ...       ...
    51.874849202000036      3.406252920574752 ...     10         0
     57.06233412220004     1.5402218528922258 ...     50         2
     62.76856753442005     0.7335240348369152 ...     50         2
     69.04542428786206     0.3611664673700657 ...     50         2
     75.94996671664828 -0.0020389321260497247 ...     38         0
      83.5449633883131   -0.04143562472379122 ...      1         5
    Length = 55 rows

Let's plot the ellipticity, position angle, and the center x and y
position as a function of the semimajor axis length:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from photutils.datasets import make_noise_image
    from photutils.isophote import Ellipse, EllipseGeometry

    g = Gaussian2D(100.0, 75, 75, 20, 12, theta=np.deg2rad(40.0))
    ny = nx = 150
    y, x = np.mgrid[0:ny, 0:nx]
    noise = make_noise_image((ny, nx), distribution='gaussian', mean=0.0,
                             stddev=2.0, seed=1234)
    data = g(x, y) + noise
    geometry = EllipseGeometry(x0=75, y0=75, sma=20, eps=0.5,
                               pa=np.deg2rad(20.0))
    ellipse = Ellipse(data, geometry=geometry)
    isolist = ellipse.fit_image()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
                                                 figsize=(8, 8))
    fig.subplots_adjust(hspace=0.35, wspace=0.35)

    ax1.errorbar(isolist.sma, isolist.eps, yerr=isolist.ellip_err,
                 fmt='o', markersize=4)
    ax1.set_xlabel('Semimajor Axis Length (pix)')
    ax1.set_ylabel('Ellipticity')

    ax2.errorbar(isolist.sma, np.rad2deg(isolist.pa),
                 yerr=np.rad2deg(isolist.pa_err), fmt='o', markersize=4)
    ax2.set_xlabel('Semimajor Axis Length (pix)')
    ax2.set_ylabel('PA (deg)')

    ax3.errorbar(isolist.sma, isolist.x0, yerr=isolist.x0_err, fmt='o',
                 markersize=4)
    ax3.set_xlabel('Semimajor Axis Length (pix)')
    ax3.set_ylabel('x0')

    ax4.errorbar(isolist.sma, isolist.y0, yerr=isolist.y0_err, fmt='o',
                 markersize=4)
    ax4.set_xlabel('Semimajor Axis Length (pix)')
    ax4.set_ylabel('y0')

We can build an elliptical model image from the
`~photutils.isophote.IsophoteList` object using the
:func:`~photutils.isophote.build_ellipse_model` function::

    >>> from photutils.isophote import build_ellipse_model
    >>> model_image = build_ellipse_model(data.shape, isolist)
    >>> residual = data - model_image

Finally, let's plot the original data, overplotted with some isophotes,
the elliptical model image, and the residual image:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from photutils.datasets import make_noise_image
    from photutils.isophote import (Ellipse, EllipseGeometry,
                                    build_ellipse_model)

    g = Gaussian2D(100.0, 75, 75, 20, 12, theta=np.deg2rad(40.0))
    ny = nx = 150
    y, x = np.mgrid[0:ny, 0:nx]
    noise = make_noise_image((ny, nx), distribution='gaussian', mean=0.0,
                             stddev=2.0, seed=1234)
    data = g(x, y) + noise
    geometry = EllipseGeometry(x0=75, y0=75, sma=20, eps=0.5,
                               pa=np.deg2rad(20.0))
    ellipse = Ellipse(data, geometry=geometry)
    isolist = ellipse.fit_image()

    model_image = build_ellipse_model(data.shape, isolist)
    residual = data - model_image

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(14, 5))
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)
    ax1.imshow(data, origin='lower')
    ax1.set_title('Data')

    smas = np.linspace(10, 50, 5)
    for sma in smas:
        iso = isolist.get_closest(sma)
        x, y = iso.sampled_coordinates()
        ax1.plot(x, y, color='white')

    ax2.imshow(model_image, origin='lower')
    ax2.set_title('Ellipse Model')

    ax3.imshow(residual, origin='lower')
    ax3.set_title('Residual')


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

    >>> noise2 = make_noise_image((ny, nx), distribution='gaussian',
    ...                           mean=0.0, stddev=2.0, seed=5678)
    >>> data2 = 0.5 * g(x, y) + noise2
    >>> isolist2 = Ellipse(data2).measure_isophotes(isolist)
    >>> mask = (isolist.intens > 0) & (isolist2.intens > 0)
    >>> color = -2.5 * np.log10(isolist2.intens[mask] / isolist.intens[mask])

The returned `~photutils.isophote.IsophoteList` has one isophote for
each input ellipse, with the same geometry. The input can also be a
sequence of `~photutils.isophote.EllipseGeometry` objects, which is
useful for measuring an image along user-defined ellipses::

    >>> geometries = [EllipseGeometry(x0=75, y0=75, sma=sma, eps=0.4,
    ...                               pa=np.deg2rad(40.0))
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
