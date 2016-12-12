Getting Started with Photutils
==============================

The following example uses Photutils to find sources in an
astronomical image and perform circular aperture photometry on them.

We start by loading an image from the bundled datasets and selecting a
subset of the image.  We then subtract a rough estimate of the
background, calculated using the image median:

    >>> import numpy as np
    >>> from photutils import datasets
    >>> hdu = datasets.load_star_image()    # doctest: +REMOTE_DATA
    Downloading ...
    >>> image = hdu.data[500:700, 500:700].astype(float)    # doctest: +REMOTE_DATA
    >>> image -= np.median(image)    # doctest: +REMOTE_DATA

In the remainder of this example, we assume that the data is
background-subtracted.

Photutils supports several source detection algorithms.  For this
example, we use :class:`~photutils.detection.DAOStarFinder` to detect
the stars in the image.  We set the detection threshold at the 3-sigma
noise level, estimated using the median absolute deviation
(`~astropy.stats.mad_std`) of the image. The parameters of the
detected sources are returned as an Astropy `~astropy.table.Table`:

.. doctest-requires:: scipy, skimage

    >>> from photutils import DAOStarFinder
    >>> from astropy.stats import mad_std
    >>> bkg_sigma = mad_std(image)    # doctest: +REMOTE_DATA
    >>> daofind = DAOStarFinder(fwhm=4., threshold=3.*bkg_sigma)    # doctest: +REMOTE_DATA
    >>> sources = daofind(image)    # doctest: +REMOTE_DATA
    >>> print(sources)    # doctest: +REMOTE_DATA
     id   xcentroid     ycentroid    ...  peak       flux           mag
    --- ------------- -------------- ... ------ ------------- ---------------
      1 182.838658938 0.167670190537 ... 3824.0 2.80283459469  -1.11899367311
      2 189.204308134 0.260813525338 ... 4913.0 3.87291850311  -1.47009589582
      3 5.79464911433  2.61254240807 ... 7752.0  4.1029107294  -1.53273016937
      4 36.8470627804  1.32202279582 ... 8739.0 7.43158178793  -2.17770315441
      5  3.2565602452  5.41895201748 ... 6935.0 3.81262984074  -1.45306160673
    ...           ...            ... ...    ...           ...             ...
    148 124.313272579  188.305229159 ... 6702.0 6.63585429303  -2.05474210356
    149 24.2572074962  194.714942814 ... 8342.0  3.2671036996  -1.28540729858
    150 116.449998422  195.059233325 ... 3299.0 2.87752205766   -1.1475466535
    151 18.9580860645  196.342065132 ... 3854.0 2.38352961224 -0.943051379595
    152 111.525751196  195.731917995 ... 8109.0  7.9278607401  -2.24789003194
    Length = 152 rows

Using the list of source locations (``xcentroid`` and ``ycentroid``),
we now compute the sum of the pixel values in circular apertures with
a radius of 4 pixels.  The :func:`~photutils.aperture_photometry`
function returns an Astropy `~astropy.table.Table` with the results of
the photometry:

.. doctest-requires:: scipy, skimage

    >>> from photutils import aperture_photometry, CircularAperture
    >>> positions = (sources['xcentroid'], sources['ycentroid'])    # doctest: +REMOTE_DATA
    >>> apertures = CircularAperture(positions, r=4.)    # doctest: +REMOTE_DATA
    >>> phot_table = aperture_photometry(image, apertures)    # doctest: +REMOTE_DATA
    >>> print(phot_table)   # doctest: +SKIP
     id      xcenter             ycenter        aperture_sum
               pix                 pix
    --- ------------------ ------------------- -------------
      1  182.8386589381308 0.16767019053693752 18121.7594837
      2 189.20430813403388 0.26081352533766516 29836.5152158
      3  5.794649114329246   2.612542408073547 331979.819037
      4  36.84706278043582  1.3220227958153257 183705.093284
      5 3.2565602452007325   5.418952017476508 349468.978627
    ...                ...                 ...           ...
    148  124.3132725793939  188.30522915858668 45084.8737867
    149 24.257207496209027  194.71494281419265 355778.007298
    150 116.44999842177826  195.05923332483115 31232.9117818
    151 18.958086064485013   196.3420651316401 162076.262752
    152 111.52575119605933  195.73191799469373 82795.7145661
    Length = 152 rows

The sum of the pixel values within the apertures are given in the
column ``aperture_sum``.  We now plot the image and the defined
apertures:

.. doctest-skip::

    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(image, cmap='gray_r', origin='lower')
    >>> apertures.plot(color='blue', lw=1.5, alpha=0.5)

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.stats import mad_std
    from photutils import (datasets, DAOStarFinder, aperture_photometry,
                           CircularAperture)
    hdu = datasets.load_star_image()
    image = hdu.data[500:700, 500:700].astype(float)
    image -= np.median(image)
    bkg_sigma = mad_std(image)
    daofind = DAOStarFinder(fwhm=4., threshold=3.*bkg_sigma)
    sources = daofind(image)
    positions = (sources['xcentroid'], sources['ycentroid'])
    apertures = CircularAperture(positions, r=4.)
    phot_table = aperture_photometry(image, apertures)
    brightest_source_id = phot_table['aperture_sum'].argmax()
    plt.imshow(image, cmap='gray_r', origin='lower')
    apertures.plot(color='blue', lw=1.5, alpha=0.5)
