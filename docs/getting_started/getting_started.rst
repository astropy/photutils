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
detected sources are returned as an Astropy `~astropy.table.Table`::

    >>> from photutils.detection import DAOStarFinder
    >>> from astropy.stats import mad_std
    >>> bkg_sigma = mad_std(image)  # doctest: +REMOTE_DATA
    >>> daofind = DAOStarFinder(fwhm=4.0, threshold=3.0 * bkg_sigma)  # doctest: +REMOTE_DATA
    >>> sources = daofind(image)  # doctest: +REMOTE_DATA
    >>> for col in sources.colnames:  # doctest: +REMOTE_DATA
    ...     sources[col].info.format = '%.5g'  # for consistent table output
    >>> print(sources)  # doctest: +REMOTE_DATA
     id xcentroid ycentroid sharpness ... peak    flux      mag   daofind_mag
    --- --------- --------- --------- ... ---- ---------- ------- -----------
      1    182.84   0.16767     0.851 ... 3824      15750 -10.493      -1.119
      2     189.2   0.26081   0.74005 ... 4913      25870 -11.032     -1.4701
      3    5.7946    2.6125    0.3959 ... 7752 1.3525e+05 -12.828     -1.5327
      4    36.847     1.322   0.29595 ... 8739 1.3158e+05 -12.798     -2.1777
      5    3.2566     5.419   0.35985 ... 6935 1.2454e+05 -12.738     -1.4531
    ...       ...       ...       ... ...  ...        ...     ...         ...
    148    124.31    188.31   0.53627 ... 6702      44592 -11.623     -2.0547
    149    24.257    194.71    0.4417 ... 8342 1.5645e+05 -12.986     -1.2854
    150    116.45    195.06   0.67081 ... 3299      17418 -10.602     -1.1475
    151    18.958    196.34   0.56502 ... 3854      48800 -11.721    -0.94305
    152    111.53    195.73   0.45828 ... 8109      74247 -12.177     -2.2479
    Length = 152 rows

Using the source locations (i.e., the ``xcentroid`` and ``ycentroid``
columns), we now define circular apertures centered at these positions
with a radius of 4 pixels and compute the sum of the pixel values
within the apertures.  The
:func:`~photutils.aperture.aperture_photometry` function returns an
Astropy `~astropy.table.QTable` with the results of the photometry::

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
