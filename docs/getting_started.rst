Getting Started with Photutils
==============================

The following example uses Photutils to find sources in an
astronomical image and then perform circular aperture photometry on
them.

We start by loading an image from the bundled datasets and selecting a
subset of the image::

    >>> import numpy as np
    >>> from photutils.datasets import load_star_image
    >>> hdu = load_star_image()  # doctest: +REMOTE_DATA
    >>> image = hdu.data[500:700, 500:700].astype(float)  # doctest: +REMOTE_DATA

We then subtract a rough estimate of the background, calculated using
the image median::

    >>> image -= np.median(image)  # doctest: +REMOTE_DATA

In the remainder of this example, we assume that the data is
background-subtracted.

Photutils supports several source detection algorithms.  For this
example, we use :class:`~photutils.detection.DAOStarFinder` to detect
the stars in the image.  We set the detection threshold at the 3-sigma
noise level, estimated using the median absolute deviation
(`~astropy.stats.mad_std`) of the image. The parameters of the
detected sources are returned as an Astropy `~astropy.table.Table`:

.. doctest-requires:: scipy

    >>> from photutils.detection import DAOStarFinder
    >>> from astropy.stats import mad_std
    >>> bkg_sigma = mad_std(image)  # doctest: +REMOTE_DATA
    >>> daofind = DAOStarFinder(fwhm=4.0, threshold=3.0 * bkg_sigma)  # doctest: +REMOTE_DATA
    >>> sources = daofind(image)  # doctest: +REMOTE_DATA
    >>> for col in sources.colnames:  # doctest: +REMOTE_DATA
    ...     sources[col].info.format = '%.8g'  # for consistent table output
    >>> print(sources)  # doctest: +REMOTE_DATA
     id xcentroid ycentroid  sharpness  ... sky peak    flux       mag
    --- --------- ---------- ---------- ... --- ---- --------- -----------
      1 182.83866 0.16767019 0.85099873 ...   0 3824 2.8028346  -1.1189937
      2 189.20431 0.26081353  0.7400477 ...   0 4913 3.8729185  -1.4700959
      3 5.7946491  2.6125424 0.39589731 ...   0 7752 4.1029107  -1.5327302
      4 36.847063  1.3220228 0.29594528 ...   0 8739 7.4315818  -2.1777032
      5 3.2565602   5.418952 0.35985495 ...   0 6935 3.8126298  -1.4530616
    ...       ...        ...        ... ... ...  ...       ...         ...
    147 197.24864  186.16647 0.31211532 ...   0 8302 7.5814629  -2.1993825
    148 124.31327  188.30523  0.5362742 ...   0 6702 6.6358543  -2.0547421
    149 24.257207  194.71494 0.44169546 ...   0 8342 3.2671037  -1.2854073
    150    116.45  195.05923 0.67080547 ...   0 3299 2.8775221  -1.1475467
    151 18.958086  196.34207 0.56502139 ...   0 3854 2.3835296 -0.94305138
    152 111.52575  195.73192 0.45827852 ...   0 8109 7.9278607    -2.24789
    Length = 152 rows

Using the source locations (i.e., the ``xcentroid`` and ``ycentroid``
columns), we now define circular apertures centered at these positions
with a radius of 4 pixels and compute the sum of the pixel values
within the apertures.  The
:func:`~photutils.aperture.aperture_photometry` function returns an
Astropy `~astropy.table.QTable` with the results of the photometry:

.. doctest-requires:: scipy

    >>> from photutils.aperture import aperture_photometry, CircularAperture
    >>> positions = np.transpose((sources['xcentroid'], sources['ycentroid']))  # doctest: +REMOTE_DATA
    >>> apertures = CircularAperture(positions, r=4.0)  # doctest: +REMOTE_DATA
    >>> phot_table = aperture_photometry(image, apertures)  # doctest: +REMOTE_DATA
    >>> for col in phot_table.colnames:  # doctest: +REMOTE_DATA
    ...     phot_table[col].info.format = '%.8g'  # for consistent table output
    >>> print(phot_table)  # doctest: +REMOTE_DATA
     id  xcenter   ycenter   aperture_sum
           pix       pix
    --- --------- ---------- ------------
      1 182.83866 0.16767019    18121.759
      2 189.20431 0.26081353    29836.515
      3 5.7946491  2.6125424    331979.82
      4 36.847063  1.3220228    183705.09
      5 3.2565602   5.418952    349468.98
    ...       ...        ...          ...
    148 124.31327  188.30523    45084.874
    149 24.257207  194.71494    355778.01
    150    116.45  195.05923    31232.912
    151 18.958086  196.34207    162076.26
    152 111.52575  195.73192    82795.715
    Length = 152 rows

The sum of the pixel values within the apertures are given in the
``aperture_sum`` column.

Finally, we plot the image and the defined apertures:

.. doctest-skip::

    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(image, cmap='gray_r', origin='lower')
    >>> apertures.plot(color='blue', lw=1.5, alpha=0.5)

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.stats import mad_std
    from photutils.aperture import CircularAperture, aperture_photometry
    from photutils.datasets import load_star_image
    from photutils.detection import DAOStarFinder

    hdu = load_star_image()
    image = hdu.data[500:700, 500:700].astype(float)
    image -= np.median(image)
    bkg_sigma = mad_std(image)
    daofind = DAOStarFinder(fwhm=4.0, threshold=3.0 * bkg_sigma)
    sources = daofind(image)
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=4.0)
    phot_table = aperture_photometry(image, apertures)
    brightest_source_id = phot_table['aperture_sum'].argmax()
    plt.imshow(image, cmap='gray_r', origin='lower')
    apertures.plot(color='blue', lw=1.5, alpha=0.5)
