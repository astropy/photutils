Elliptical Isophote Analysis (`photutils.isophote`)
===================================================

Introduction
------------

The `~photutils.isophote` package provides tools to fit elliptical
isophotes to a galaxy image.  The isophotes in the image are measured
using an iterative method described by `Jedrzejewski (1987; MNRAS 226,
747)
<https://ui.adsabs.harvard.edu/abs/1987MNRAS.226..747J/abstract>`_.
See the documentation of the :class:`~photutils.isophote.Ellipse`
class for details about the algorithm.  Please also see the
:ref:`isophote-faq`.

Getting Started
---------------

For this example, let's create a simple simulated galaxy image::

    >>> import numpy as np
    >>> from astropy.modeling.models import Gaussian2D
    >>> from photutils.datasets import make_noise_image

    >>> g = Gaussian2D(100.0, 75, 75, 20, 12, theta=40.0 * np.pi / 180.0)
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

    g = Gaussian2D(100.0, 75, 75, 20, 12, theta=40.0 * np.pi / 180.0)
    ny = nx = 150
    y, x = np.mgrid[0:ny, 0:nx]
    noise = make_noise_image((ny, nx), distribution='gaussian', mean=0.0,
                             stddev=2.0, seed=1234)
    data = g(x, y) + noise
    plt.imshow(data, origin='lower')

We must provide the elliptical isophote fitter with an initial ellipse
to be fitted.  This ellipse geometry is defined with the
`~photutils.isophote.EllipseGeometry` class.  Here we'll define an
initial ellipse whose position angle is offset from the data::

    >>> from photutils.isophote import EllipseGeometry
    >>> geometry = EllipseGeometry(x0=75, y0=75, sma=20, eps=0.5,
    ...                            pa=20.0 * np.pi / 180.0)

Let's show this initial ellipse guess:

.. doctest-skip::

    >>> import matplotlib.pyplot as plt
    >>> from photutils.aperture import EllipticalAperture
    >>> aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
    ...                           geometry.sma * (1 - geometry.eps),
    ...                           geometry.pa)
    >>> plt.imshow(data, origin='lower')
    >>> aper.plot(color='white')

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from photutils.aperture import EllipticalAperture
    from photutils.datasets import make_noise_image
    from photutils.isophote import EllipseGeometry

    g = Gaussian2D(100.0, 75, 75, 20, 12, theta=40.0 * np.pi / 180.0)
    ny = nx = 150
    y, x = np.mgrid[0:ny, 0:nx]
    noise = make_noise_image((ny, nx), distribution='gaussian', mean=0.0,
                             stddev=2.0, seed=1234)
    data = g(x, y) + noise

    geometry = EllipseGeometry(x0=75, y0=75, sma=20, eps=0.5,
                               pa=20.0 * np.pi / 180.0)
    aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                              geometry.sma * (1 - geometry.eps), geometry.pa)
    plt.imshow(data, origin='lower')
    aper.plot(color='white')

Next, we create an instance of the `~photutils.isophote.Ellipse`
class, inputting the data to be fitted and the initial ellipse
geometry object::

    >>> from photutils.isophote import Ellipse
    >>> ellipse = Ellipse(data, geometry)

To perform the elliptical isophote fit, we run the
:meth:`~photutils.isophote.Ellipse.fit_image` method:

.. doctest-requires:: scipy

    >>> isolist = ellipse.fit_image()

The result is a list of isophotes as an
`~photutils.isophote.IsophoteList` object, whose attributes are the
fit values for each `~photutils.isophote.Isophote` sorted by the
semimajor axis length.  Let's print the fit position angles
(radians):

.. doctest-requires:: scipy

    >>> print(isolist.pa)  # doctest: +SKIP
    [ 0.          0.16838914  0.18453378  0.20310945  0.22534975  0.25007781
      0.28377499  0.32494582  0.38589202  0.40480013  0.39527698  0.38448771
      0.40207495  0.40207495  0.28201524  0.28201524  0.19889817  0.1364335
      0.1364335   0.13405719  0.17848892  0.25687327  0.35750355  0.64882699
      0.72489435  0.91472008  0.94219702  0.87393299  0.82572916  0.7886367
      0.75523282  0.7125274   0.70481612  0.7120097   0.71250791  0.69707669
      0.7004807   0.70709823  0.69808124  0.68621341  0.69437566  0.70548293
      0.70427021  0.69978326  0.70410887  0.69532744  0.69440413  0.70062534
      0.68614488  0.7177538   0.7177538   0.7029571   0.7029571   0.7029571 ]

We can also show the isophote values as a table, which is again sorted
by the semimajor axis length (``sma``):

.. doctest-requires:: scipy

    >>> print(isolist.to_table())  # doctest: +SKIP
         sma            intens        intens_err   ... flag niter stop_code
                                                   ...
    -------------- --------------- --------------- ... ---- ----- ---------
               0.0   102.237692914             0.0 ...    0     0         0
    0.534697261283   101.212218041 0.0280377938856 ...    0    10         0
    0.588166987411   101.095404456  0.027821598428 ...    0    10         0
    0.646983686152   100.971770355 0.0272405762608 ...    0    10         0
    0.711682054767   100.842254551 0.0262991125932 ...    0    10         0
               ...             ...             ... ...  ...   ...       ...
      51.874849202   3.44800874483 0.0881592058138 ...    0    50         2
     57.0623341222   1.64031530995 0.0913122295433 ...    0    50         2
     62.7685675344  0.692631010404 0.0786846787635 ...    0    32         0
     69.0454242879  0.294659388337 0.0681758007533 ...    0     8         5
     75.9499667166 0.0534892334515 0.0692483210903 ...    0     2         5
    Length = 54 rows

Let's plot the ellipticity, position angle, and the center x and y
position as a function of the semimajor axis length:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from photutils.datasets import make_noise_image
    from photutils.isophote import Ellipse, EllipseGeometry

    g = Gaussian2D(100.0, 75, 75, 20, 12, theta=40.0 * np.pi / 180.0)
    ny = nx = 150
    y, x = np.mgrid[0:ny, 0:nx]
    noise = make_noise_image((ny, nx), distribution='gaussian', mean=0.0,
                             stddev=2.0, seed=1234)
    data = g(x, y) + noise
    geometry = EllipseGeometry(x0=75, y0=75, sma=20, eps=0.5,
                               pa=20.0 * np.pi / 180.0)
    ellipse = Ellipse(data, geometry)
    isolist = ellipse.fit_image()

    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(hspace=0.35, wspace=0.35)

    plt.subplot(2, 2, 1)
    plt.errorbar(isolist.sma, isolist.eps, yerr=isolist.ellip_err,
                 fmt='o', markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('Ellipticity')

    plt.subplot(2, 2, 2)
    plt.errorbar(isolist.sma, isolist.pa / np.pi * 180.0,
                 yerr=isolist.pa_err / np.pi * 80.0, fmt='o', markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('PA (deg)')

    plt.subplot(2, 2, 3)
    plt.errorbar(isolist.sma, isolist.x0, yerr=isolist.x0_err, fmt='o',
                 markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('x0')

    plt.subplot(2, 2, 4)
    plt.errorbar(isolist.sma, isolist.y0, yerr=isolist.y0_err, fmt='o',
                 markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('y0')

We can build an elliptical model image from the
`~photutils.isophote.IsophoteList` object using the
:func:`~photutils.isophote.build_ellipse_model` function ( NOTE: this
function requires `scipy <https://www.scipy.org/>`_):

.. doctest-requires:: scipy

    >>> from photutils.isophote import build_ellipse_model
    >>> model_image = build_ellipse_model(data.shape, isolist)
    >>> residual = data - model_image

Finally, let's plot the original data, overplotted with some of the
isophotes, the elliptical model image, and the residual image:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling.models import Gaussian2D
    from photutils.datasets import make_noise_image
    from photutils.isophote import (Ellipse, EllipseGeometry,
                                    build_ellipse_model)

    g = Gaussian2D(100.0, 75, 75, 20, 12, theta=40.0 * np.pi / 180.0)
    ny = nx = 150
    y, x = np.mgrid[0:ny, 0:nx]
    noise = make_noise_image((ny, nx), distribution='gaussian', mean=0.0,
                             stddev=2.0, seed=1234)
    data = g(x, y) + noise
    geometry = EllipseGeometry(x0=75, y0=75, sma=20, eps=0.5,
                               pa=20.0 * np.pi / 180.0)
    ellipse = Ellipse(data, geometry)
    isolist = ellipse.fit_image()

    model_image = build_ellipse_model(data.shape, isolist)
    residual = data - model_image

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(14, 5), nrows=1, ncols=3)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)
    ax1.imshow(data, origin='lower')
    ax1.set_title('Data')

    smas = np.linspace(10, 50, 5)
    for sma in smas:
        iso = isolist.get_closest(sma)
        x, y, = iso.sampled_coordinates()
        ax1.plot(x, y, color='white')

    ax2.imshow(model_image, origin='lower')
    ax2.set_title('Ellipse Model')

    ax3.imshow(residual, origin='lower')
    ax3.set_title('Residual')


Additional Example Notebooks (online)
-------------------------------------

Additional example notebooks showing examples with real data and
advanced usage are available online:

* `Basic example of the Ellipse fitting tool <https://github.com/astropy/photutils-datasets/blob/main/notebooks/isophote/isophote_example1.ipynb>`_

* `Running Ellipse with sigma-clipping <https://github.com/astropy/photutils-datasets/blob/main/notebooks/isophote/isophote_example2.ipynb>`_

* `Building an image model from results obtained by Ellipse fitting <https://github.com/astropy/photutils-datasets/blob/main/notebooks/isophote/isophote_example3.ipynb>`_

* `Advanced Ellipse example: multi-band photometry and masked arrays <https://github.com/astropy/photutils-datasets/blob/main/notebooks/isophote/isophote_example4.ipynb>`_


Reference/API
-------------

.. automodapi:: photutils.isophote
    :no-heading:


.. toctree::
    :hidden:

    isophote_faq.rst
