Source detection and photometry (`photutils`)
=============================================

Introduction
------------

The `photutils` package contains functions for

* detecting sources in astronomical images
* estimating morphological parameters of those sources (e.g., centroid
  and shape parameters)
* performing aperture and PSF photometry

The code and the documentation are available at the following links:

* Code: https://github.com/astropy/photutils
* Docs: https://photutils.readthedocs.org/

Dependencies
------------

`photutils` requires the following packages to be available:

* `numpy <http://www.numpy.org/>`__
* `astropy <http://www.astropy.org/>`__
* `imageutils <https://imageutils.readthedocs.org/en/latest/imageutils/index.html>`__
  (planned to be included in the Astropy core as ``astropy.image``
  before the astropy 1.0 release)

You will also need `Cython`_ installed to build `photutils` from
source, unless you are installing a numbered release.  The released
packages have the necessary C files packaged with them, and hence do
not require `Cython`_.

.. _Cython: http://cython.org

Some functionality is available only if the following optional
dependencies are installed:

* `scipy <http://www.scipy.org/>`__
* `scikit-image <http://scikit-image.org/>`__
* `matplotlib <http://matplotlib.org/>`__

.. _coordinate-conventions:

Coordinate Conventions in `photutils`
-------------------------------------

In `photutils`, pixel coordinates are zero-indexed, meaning that ``(x,
y) = (0, 0)`` corresponds to the center of the lowest, leftmost array
element.  This means that the value of ``data[0, 0]`` is taken as the
value over the range ``-0.5 < x <= 0.5``, ``-0.5 < y <= 0.5``.  Note
that this differs from the SourceExtractor_, IRAF_, FITS, and ds9_
conventions, in which the center of the lowest, leftmost array element
is ``(1, 1)``.

The ``x`` (column) coordinate corresponds to the second (fast) array
index and the ``y`` (row) coordinate corresponds to the first (slow)
index.  ``data[y, x]`` gives the value at coordinates (x, y).  Along
with zero-indexing, this means that an array is defined over the
coordinate range ``-0.5 < x <= data.shape[1] - 0.5``, ``-0.5 < y <=
data.shape[0] - 0.5``.

.. _SourceExtractor: http://www.astromatic.net/software/sextractor
.. _IRAF: http://iraf.noao.edu/
.. _ds9: http://ds9.si.edu/

Bundled Datasets
----------------

In this documentation, we use example datasets provided by calling
functions such as :func:`~photutils.datasets.load_star_image`.  This
function returns an Astropy :class:`~astropy.io.fits.ImageHDU` object,
and is equivalent to doing:

.. doctest-skip::

  >>> from astropy.io import fits
  >>> hdu = fits.open('dataset.fits')[0]

where the ``[0]`` accesses the first HDU in the FITS file.

Getting Started
---------------

The following example uses `photutils` to find sources in an
astronomical image and perform aperture photometry on them.  We start
by selecting a subset of the data and subtracting a rough estimate of
the background, calculated from the image median:

  >>> import numpy as np
  >>> from photutils import datasets
  >>> hdu = datasets.load_star_image()   # doctest: +REMOTE_DATA
  >>> image = hdu.data[500:700, 500:700]   # doctest: +REMOTE_DATA
  >>> image -= np.median(image)   # doctest: +REMOTE_DATA

In the remainder of this example, we assume that the data is
background-subtracted.

`photutils` supports different source detection algorithms.  For this
example, we use :func:`~photutils.daofind`.  The parameters of the
detected sources are returned as an Astropy `~astropy.table.Table`:

  >>> from photutils import daofind
  >>> from astropy.stats import median_absolute_deviation as mad
  >>> bkg_sigma = 1.48 * mad(image)   # doctest: +REMOTE_DATA
  >>> sources = daofind(image, fwhm=4.0, threshold=3*bkg_sigma)   # doctest: +REMOTE_DATA
  >>> print sources   # doctest: +REMOTE_DATA
   id   xcentroid     ycentroid    ...  peak       flux           mag
  --- ------------- -------------- ... ------ ------------- ---------------
    1 182.838658938 0.167670190537 ... 3824.0 2.80776269472  -1.12090099874
    2 189.204308134 0.260813525338 ... 4913.0 3.87972808432  -1.47200322145
    3 5.79464911433  2.61254240807 ... 7752.0 4.11012469577    -1.534637495
    4 36.8470627804  1.32202279582 ... 8739.0 7.44464840931  -2.17961048004
    5  3.2565602452  5.41895201748 ... 6935.0 3.81933341907  -1.45496893236
  ...           ...            ... ...    ...           ...             ...
  148 124.313272579  188.305229159 ... 6702.0 6.64752182198  -2.05664942919
  149 24.2572074962  194.714942814 ... 8342.0 3.27284810346  -1.28731462421
  150 116.449998422  195.059233325 ... 3299.0 2.88258147736  -1.14945397913
  151 18.9580860645  196.342065132 ... 3854.0 2.38772046688 -0.944958705225
  152 111.525751196  195.731917995 ... 8109.0 7.94179994681  -2.24979735757

Given the list of source locations, we now compute the sum of the
pixel values in identical circular apertures with a radius of 4
pixels.  The :func:`~photutils.aperture_photometry` function returns
an Astropy `~astropy.table.Table` with the results of the photometry:

  >>> from photutils import aperture_photometry, CircularAperture
  >>> positions = zip(sources['xcentroid'], sources['ycentroid'])   # doctest: +REMOTE_DATA
  >>> apertures = CircularAperture(positions, r=4)   # doctest: +REMOTE_DATA
  >>> phot_table = aperture_photometry(image, apertures)   # doctest: +REMOTE_DATA
  >>> print phot_table   # doctest: +REMOTE_DATA
     aperture_sum    xcenter       ycenter
                       pix           pix
    ------------- ------------- --------------
    18121.7594837 182.838658938 0.167670190537
    29836.5152158 189.204308134 0.260813525338
    331979.819037 5.79464911433  2.61254240807
    183705.093284 36.8470627804  1.32202279582
    349468.978627  3.2565602452  5.41895201748
              ...           ...            ...
    45084.8737867 124.313272579  188.305229159
    355778.007298 24.2572074962  194.714942814
    31232.9117818 116.449998422  195.059233325
    162076.262752 18.9580860645  196.342065132
    82795.7145661 111.525751196  195.731917995

The sum of the pixel values within the apertures are given in the
column ``aperture_sum``.  We now plot the image and the defined
apertures:

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
  positions = zip(sources['xcentroid'], sources['ycentroid'])
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
    detection.rst
    segmentation.rst
    morphology.rst
    geometry.rst
    datasets.rst
    utils.rst


.. toctree::
  :maxdepth: 1

  high-level_API.rst


.. note::
    We also have a series of IPython notebooks that demonstrate how to
    use `photutils`.  You can view them online `here
    <http://nbviewer.ipython.org/github/astropy/photutils-datasets/tree/master/notebooks/>`__
    or download them `here
    <https://github.com/astropy/photutils-datasets>`__ if you'd like
    to execute them on your machine.  Contributions are welcome!
