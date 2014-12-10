Getting Started
===============

The following example uses photutils to find sources in an
astronomical image and perform circular aperture photometry on them.

We start by loading an image from the bundled datasets and selecting a
subset of the image.  We then subtract a rough estimate of the
background, calculated using the image median:

    >>> import numpy as np
    >>> from photutils import datasets
    >>> hdu = datasets.load_star_image()    # doctest: +REMOTE_DATA
    >>> image = hdu.data[500:700, 500:700]    # doctest: +REMOTE_DATA
    >>> image -= np.median(image)    # doctest: +REMOTE_DATA

In the remainder of this example, we assume that the data is
background-subtracted.

Photutils supports several source detection algorithms.  For this
example, we use :func:`~photutils.daofind` to detect the stars in the
image.  We set the detection threshold at the 3-sigma noise level,
estimated using the median absolution deviation of the image.  The
parameters of the detected sources are returned as an Astropy
`~astropy.table.Table`:

    >>> from photutils import daofind
    >>> from astropy.stats import median_absolute_deviation as mad
    >>> bkg_sigma = 1.48 * mad(image)    # doctest: +REMOTE_DATA
    >>> sources = daofind(image, fwhm=4., threshold=3.*bkg_sigma)    # doctest: +REMOTE_DATA
    >>> print(sources)    # doctest: +REMOTE_DATA
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

Using the list of source locations (``xcentroid`` and ``ycentroid``),
we now compute the sum of the pixel values in circular apertures with
a radius of 4 pixels.  The :func:`~photutils.aperture_photometry`
function returns an Astropy `~astropy.table.Table` with the results of
the photometry:

    >>> from photutils import aperture_photometry, CircularAperture
    >>> positions = (sources['xcentroid'], sources['ycentroid'])    # doctest: +REMOTE_DATA
    >>> apertures = CircularAperture(positions, r=4.)    # doctest: +REMOTE_DATA
    >>> phot_table = aperture_photometry(image, apertures)    # doctest: +REMOTE_DATA
    >>> print(phot_table)   # doctest: +REMOTE_DATA
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

    >>> import matplotlib.pylab as plt
    >>> plt.imshow(image, cmap='gray_r', origin='lower')
    >>> apertures.plot(color='blue', lw=1.5, alpha=0.5)

.. plot::

    import numpy as np
    import matplotlib.pylab as plt
    from astropy.stats import median_absolute_deviation as mad
    from photutils import datasets, daofind, aperture_photometry, CircularAperture
    hdu = datasets.load_star_image()
    image = hdu.data[500:700, 500:700]
    image -= np.median(image)
    bkg_sigma = 1.48 * mad(image)
    sources = daofind(image, fwhm=4., threshold=3.*bkg_sigma)
    positions = (sources['xcentroid'], sources['ycentroid'])
    apertures = CircularAperture(positions, r=4.)
    phot_table = aperture_photometry(image, apertures)
    brightest_source_id = phot_table['aperture_sum'].argmax()
    plt.imshow(image, cmap='gray_r', origin='lower')
    apertures.plot(color='blue', lw=1.5, alpha=0.5)


.. note::
    We also have a series of IPython notebooks that demonstrate how to
    use photutils.  You can view them `online
    <http://nbviewer.ipython.org/github/astropy/photutils-datasets/tree/master/notebooks/>`_
    or `download <https://github.com/astropy/photutils-datasets>`_
    them if you'd like to execute them on your machine.  Contributions
    are welcome!
