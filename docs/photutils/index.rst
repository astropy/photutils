Source detection and photometry (`photutils`)
=============================================

Introduction
------------

The `photutils` package is destined to implement functions for

* detecting sources on astronomical images
* estimating morphological parameters of those sources
  (e.g., centroid and shape parameters)
* performing photometry (both aperture and PSF)

The code and the documentation are available at the following links:

* Code: https://github.com/astropy/photutils
* Docs: https://photutils.readthedocs.org/

Dependencies
------------

`photutils` requires the following packages to be available:

* `numpy <http://www.numpy.org/>`__
* `astropy <http://www.astropy.org/>`__
* `imageutils <https://imageutils.readthedocs.org/en/latest/imageutils/index.html>`__
  (planned to be included in the Astropy core as ``astropy.image`` before the 1.0 release)

You will also need `Cython <http://cython.org/>`__ installed to build
from source, unless you are installing a numbered release. (The
released packages have the necessary C files packaged with them, and
hence do not require Cython.)

Some functionality is only available if the following optional dependencies are installed:

* `scipy <http://www.scipy.org/>`__
* `scikit-image <http://scikit-image.org/>`__
* `matplotlib <http://matplotlib.org/>`__

Datasets bundled with photutils
-------------------------------

In this documentation, we use example datasets by calling functions such as
:func:`~photutils.datasets.load_star_image`. This function returns an Astropy
:class:`~astropy.io.fits.ImageHDU` object, and is equivalent to doing:

.. doctest-skip::

  >>> from astropy.io import fits
  >>> hdu = fits.open('dataset.fits')[0]

where the ``[0]`` accesses the first HDU in the FITS file.

Getting Started
---------------

Given a data array, the following example uses `photutils` to find sources
and perform aperture photometry on them. We start off by selecting a subset
of the data and subtracting the median in order to get rid of the background:

  >>> import numpy as np
  >>> from photutils import datasets
  >>> hdu = datasets.load_star_image()   # doctest: +REMOTE_DATA
  >>> image = hdu.data[500:700, 500:700]   # doctest: +REMOTE_DATA
  >>> image -= np.median(image)   # doctest: +REMOTE_DATA

In the remainder of the example, we assume that the data is
background-subtracted. `photutils` supports different source detection
algorithms, and this example uses `~photutils.daofind`. The parameters of the
detected sources are returned as an Astropy `~astropy.table.Table`:

  >>> from photutils import daofind
  >>> from astropy.stats import median_absolute_deviation as mad
  >>> bkg_sigma = 1.48 * mad(image)   # doctest: +REMOTE_DATA
  >>> sources = daofind(image, fwhm=4.0, threshold=3*bkg_sigma)   # doctest: +REMOTE_DATA
  >>> print sources   # doctest: +REMOTE_DATA
   id      xcen          ycen      ...  peak       flux           mag
  --- ------------- -------------- ... ------ ------------- ---------------
    1   5.711139137  3.74389258926 ... 8750.0  1.1355985048 -0.138062030087
    2 36.9311803628 0.999800648995 ... 8829.0 4.30200108896  -1.58417628996
    3 135.905023257  9.10944508218 ... 8880.0 2.88029075308  -1.14859082538
    4  55.046682459  11.0380934125 ... 8659.0 2.19524464207 -0.853707314613
    5 93.5564238612  7.38602419245 ... 7440.0  7.0473473444  -2.12006419302
  ...           ...            ... ...    ...           ...             ...
  102 124.313272579  188.305229159 ... 6702.0 6.64693826121  -2.05655411233
  103 25.7834226993  196.021796422 ... 8795.0 3.55206363482  -1.37620184378
  104 111.525751196  195.731917995 ... 8109.0 7.94107811571   -2.2496986704
  105 116.449998422  195.059233325 ... 3299.0 2.88250295796  -1.14942440408

Given the list of source locations, we now compute the sum of the pixel
values in identical circular apertures. The
:func:`~photutils.aperture_photometry` function returns an Astropy
`~astropy.table.Table` with the results of the photometry:

  >>> from photutils import aperture_photometry, CircularAperture
  >>> positions = zip(sources['xcen'], sources['ycen'])   # doctest: +REMOTE_DATA
  >>> apertures = CircularAperture(positions, r=4)   # doctest: +REMOTE_DATA
  >>> phot_table = aperture_photometry(image, apertures)   # doctest: +REMOTE_DATA
  >>> print phot_table   # doctest: +REMOTE_DATA
   aperture_sum         pixel_center [2]                input_center [2]
                              pix                             pix
  ------------- ------------------------------- -------------------------------
  385745.166653    5.711139137 .. 3.74389258926    5.711139137 .. 3.74389258926
  181641.994629 36.9311803628 .. 0.999800648995 36.9311803628 .. 0.999800648995
   356417.35327  135.905023257 .. 9.10944508218  135.905023257 .. 9.10944508218
  368058.976195   55.046682459 .. 11.0380934125   55.046682459 .. 11.0380934125
  66815.0304252  93.5564238612 .. 7.38602419245  93.5564238612 .. 7.38602419245
  49273.9292606  19.1364892873 .. 9.04066195256  19.1364892873 .. 9.04066195256
  203403.359483  67.6959251013 .. 13.0152136495  67.6959251013 .. 13.0152136495
            ...                             ...                             ...
  143839.371649  14.5104744617 .. 185.863679944  14.5104744617 .. 185.863679944
  142341.472824  153.652189075 .. 185.982720367  153.652189075 .. 185.982720367
  137250.763761  197.312181397 .. 186.177540419  197.312181397 .. 186.177540419
  45084.8737867  124.313272579 .. 188.305229159  124.313272579 .. 188.305229159
  341222.112791  25.7834226993 .. 196.021796422  25.7834226993 .. 196.021796422
  82795.7145661  111.525751196 .. 195.731917995  111.525751196 .. 195.731917995
  31232.9117818  116.449998422 .. 195.059233325  116.449998422 .. 195.059233325

The sum of the pixels is given in the column ``aperture_sum``. We can now plot the image and the apertures:

.. doctest-skip::

  >>> import matplotlib.patches as patches
  >>> import matplotlib.pylab as plt
  >>> plt.imshow(image, cmap='gray_r', origin='lower')
  >>> apertures.plot(color='blue', lw=1.5, alpha=0.5)

.. plot::

  import numpy as np
  import matplotlib.pylab as plt
  import matplotlib.patches as patches
  from astropy.stats import median_absolute_deviation as mad
  from photutils import datasets, daofind, aperture_photometry, CircularAperture
  hdu = datasets.load_star_image()
  image = hdu.data[500:700, 500:700]
  image -= np.median(image)
  bkg_sigma = 1.48 * mad(image)
  sources = daofind(image, fwhm=4.0, threshold=3*bkg_sigma)
  positions = zip(sources['xcen'], sources['ycen'])
  radius = 4.
  apertures = CircularAperture(positions, radius)
  phot_table = aperture_photometry(image, apertures)
  brightest_source_id = phot_table['aperture_sum'].argmax()
  plt.imshow(image, cmap='gray_r', origin='lower')
  apertures.plot(color='blue', lw=1.5, alpha=0.5)


Using `photutils`
-----------------

.. toctree::
    :maxdepth: 2

    aperture.rst
    psf.rst
    datasets.rst
    detection.rst
    segmentation.rst
    morphology.rst
    geometry.rst
    utils.rst


.. toctree::
  :maxdepth: 1

  high-level_API.rst


.. note::
   We also have a series of IPython notebooks that demonstrate how to use photutils.
   You can view them online `here <http://nbviewer.ipython.org/github/astropy/photutils-datasets/tree/master/notebooks/>`__
   or download them `here <https://github.com/astropy/photutils-datasets>`__ if you'd like to execute them on your machine.
   Contributions welcome!


.. _coordinate-conventions:

Coordinate Conventions in `photutils`
-------------------------------------

In this module the coordinates are zero-indexed, meaning that ``(x, y)
= (0., 0.)`` corresponds to the center of the lower-left array
element.  For example, the value of ``data[0, 0]`` is taken as the
value over the range ``-0.5 < x <= 0.5``, ``-0.5 < y <= 0.5``. Note
that this differs from the SourceExtractor_ convention, in which the
center of the lower-left array element is ``(1, 1)``.

The ``x`` coordinate corresponds to the second (fast) array index and
the ``y`` coordinate corresponds to the first (slow) index. So
``data[y, x]`` gives the value at coordinates (x, y). Along with the
zero-indexing, this means that the array is defined over the
coordinate range ``-0.5 < x <= data.shape[1] - 0.5``,
``-0.5 < y <= data.shape[0] - 0.5``.


.. _SourceExtractor: http://www.astromatic.net/software/sextractor
