.. _build-epsf:

Building an effective Point Spread Function (ePSF)
==================================================

The ePSF
--------

The instrumental PSF is a combination of many factors that are
generally difficult to model.  `Anderson and King (2000; PASP 112,
1360) <http://adsabs.harvard.edu/abs/2000PASP..112.1360A>`_ showed
that accurate stellar photometry and astrometry can be derived by
modeling the net PSF, which they call the effective PSF (ePSF).  The
ePSF is an empirical model describing what fraction of a star's light
will land in a particular pixel.  The constructed ePSF is typically
oversampled with respect to the detector pixels.


Building an ePSF
----------------

Photutils provides tools for building an ePSF following the
prescription of `Anderson and King (2000; PASP 112, 1360)
<http://adsabs.harvard.edu/abs/2000PASP..112.1360A>`_ The process
involves iterating between the ePSF itself and the stars used to build
it.

To begin, we must first define a sample of stars used to build the
ePSF.  Ideally these stars should be bright (high S/N) and isolated to
prevent contamination from nearby stars.  One may use the star-finding
tools in Photutils (e.g.  :class:`~photutils.detection.DAOStarFinder`
or :class:`~photutils.detection.IRAFStarFinder`) to identify an
initial sample of stars.  However, the step of creating a good sample
of stars will also likely require visual inspection and manual
selection to ensure stars are sufficiently isolated and of good
quality (e.g. no cosmic rays, detector artifacts, etc.).

Let's start by loading a simulated HST/WFC3 image in the F160W band::

    >>> from photutils import datasets
    >>> hdu = datasets.load_simulated_hst_star_image()
    >>> data = hdu.data

The simulated image does not contain any background or noise, so let's add
those to the image::

    >>> from photutils.datasets import make_noise_image
    >>> data +=  make_noise_image(data.shape, type='gaussian', mean=10.,
    ...                           stddev=5., random_state=12345)

Let's show the image:

.. plot::
    :include-source:

    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm
    from photutils import datasets
    from photutils.datasets import make_noise_image

    hdu = datasets.load_simulated_hst_star_image()
    data = hdu.data
    data +=  make_noise_image(data.shape, type='gaussian', mean=10.,
                              stddev=5., random_state=12345)
    norm = simple_norm(data, 'sqrt', percent=99.)
    plt.imshow(data, norm=norm, origin='lower', cmap='viridis')

For this example we'll use the :func:`~photutils.detection.find_peaks`
function to identify the stars and their initial positions.  We will
not use the centroiding option in
:func:`~photutils.detection.find_peaks` to simulate the effect of
having imperfect initial guesses for the positions of the stars.  Here we
set the detection threshold value to 500.0 to select only the brightest
stars:

.. doctest-requires:: scipy

    >>> from photutils import find_peaks
    >>> peaks_tbl = find_peaks(data, threshold=500.)
    >>> peaks_tbl['peak_value'].info.format = '%.8g'  # for consistent table output
    >>> print(peaks_tbl)
    x_peak y_peak peak_value
    ------ ------ ----------
       849      2  1077.6815
       182      4  1715.4047
       324      4  3017.0213
       100      9  1142.7459
       824      9  1296.9274
       ...    ...        ...
       751    992   809.9401
       114    994   1583.773
       299    994  656.79885
       207    998  2808.3181
       691    999  2614.8846
    Length = 429 rows

Note that the stars are sufficiently separated in the simulated image
that we do not need to exclude any stars due to crowding.  In practice
this step will require some manual inspection and selection.

Next, we need to extract cutouts of the stars using the
:func:`~photutils.psf.extract_stars` function.  This function requires
a table of star positions either in pixel or sky coordinates.  For
this example we are using the pixel coordinates, which need to be in
table columns called simply ``x`` and ``y``.  Let's create that
table:

.. doctest-requires:: scipy

    >>> from astropy.table import Table
    >>> stars_tbl = Table()
    >>> stars_tbl['x'] = peaks_tbl['x_peak']
    >>> stars_tbl['y'] = peaks_tbl['y_peak']

The star cutouts from which we build the ePSF must have the background
subtracted.  Here we'll use the sigma-clipped median value as the
background level.  If the background in the image varies across the
image, one should use more sophisticated methods (e.g.
`~photutils.background.Background2D`).

Let's subtract the background from the image::

    >>> from astropy.stats import sigma_clipped_stats
    >>> mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.)
    >>> data -= median_val

The :func:`~photutils.psf.extract_stars` function requires the input
data as an `~astropy.nddata.NDData` object.  An
`~astropy.nddata.NDData` object is easy to create from our data
array::

    >>> from astropy.nddata import NDData
    >>> nddata = NDData(data=data)

We are now ready to create our star cutouts using the
:func:`~photutils.psf.extract_stars` function.  For this simple
example we are extracting stars from a single image using a single
catalog.  The :func:`~photutils.psf.extract_stars` can also extract
stars from multiple images using a separate catalog for each image or
a single catalog.  When using a single catalog, the star positions
must be in sky coordinates (as `~astropy.coordinates.SkyCoord`
objects) and the `~astropy.nddata.NDData` objects must contain valid
`~astropy.wcs.WCS` objects.  In the case of using multiple images
(i.e. dithered images) and a single catalog, the same physical star
will be "linked" across images, meaning it will be constrained to have
the same sky coordinate in each input image.

Let's extract 25 x 25 pixel cutouts of our selected stars:

.. doctest-requires:: scipy

    >>> from photutils.psf import extract_stars
    >>> stars = extract_stars(nddata, stars_tbl, size=25)

The function returns a `~photutils.psf.EPSFStars` object containing
the cutouts of our selected stars.  The function extracted 403 stars,
from which we'll build our ePSF.  Let's show the first 25 of them:

.. doctest-skip::

    >>> import matplotlib.pyplot as plt
    >>> from astropy.visualization import simple_norm
    >>> nrows = 5
    >>> ncols = 5
    >>> fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),
    ...                        squeeze=True)
    >>> ax = ax.ravel()
    >>> for i in range(nrows*ncols):
    ...     norm = simple_norm(stars[i], 'log', percent=99.)
    ...     ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')

.. plot::

    from astropy.visualization import simple_norm
    from photutils import datasets

    hdu = datasets.load_simulated_hst_star_image()
    data = hdu.data
    from photutils.datasets import make_noise_image
    data +=  make_noise_image(data.shape, type='gaussian', mean=10.,
                              stddev=5., random_state=12345)

    from photutils import find_peaks
    peaks_tbl = find_peaks(data, threshold=500.)

    from astropy.table import Table
    stars_tbl = Table()
    stars_tbl['x'] = peaks_tbl['x_peak']
    stars_tbl['y'] = peaks_tbl['y_peak']

    from astropy.stats import sigma_clipped_stats
    mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.)
    data -= median_val

    from astropy.nddata import NDData
    nddata = NDData(data=data)

    from photutils.psf import extract_stars
    stars = extract_stars(nddata, stars_tbl, size=25)

    import matplotlib.pyplot as plt
    nrows = 5
    ncols = 5
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),
                           squeeze=True)
    ax = ax.ravel()
    for i in range(nrows*ncols):
        norm = simple_norm(stars[i], 'log', percent=99.)
        ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')

With the star cutouts in hand, we are ready to construct the ePSF with
the :class:`~photutils.psf.EPSFBuilder` class.  We'll create an ePSF
with an oversampling factor of 4.0.  Here we limit the maximum number
of iterations to 3 (to limit it's run time), but in practice one
should use about 10 or more iterations.  The
:class:`~photutils.psf.EPSFBuilder` class has many other options to
control the ePSF build process, including changing the centering
function, the smoothing kernel, and the centering accuracy.  Please
see the :class:`~photutils.psf.EPSFBuilder` documentation for further
details.

We first initialize an :class:`~photutils.psf.EPSFBuilder` instance
with our desired parameters and then input the cutouts of our selected
stars to the instance:

.. doctest-requires:: scipy

    >>> from photutils import EPSFBuilder
    >>> epsf_builder = EPSFBuilder(oversampling=4, maxiters=3,
    ...                            progress_bar=False)
    >>> epsf, fitted_stars = epsf_builder(stars)

The returned values are the ePSF, as an
:class:`~photutils.psf.EPSFModel` object, and our input stars fitted
with the constructed ePSF, as a new :class:`~photutils.psf.EPSFStars`
object with fitted star positions and fluxes.

Finally, let's show the constructed ePSF:

.. doctest-skip::

    >>> import matplotlib.pyplot as plt
    >>> from astropy.visualization import simple_norm
    >>> norm = simple_norm(epsf.data, 'log', percent=99.)
    >>> plt.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
    >>> plt.colorbar()

.. plot::

    from astropy.visualization import simple_norm
    from photutils import datasets

    hdu = datasets.load_simulated_hst_star_image()
    data = hdu.data
    from photutils.datasets import make_noise_image
    data +=  make_noise_image(data.shape, type='gaussian', mean=10.,
                              stddev=5., random_state=12345)

    from photutils import find_peaks
    peaks_tbl = find_peaks(data, threshold=500.)

    from astropy.table import Table
    stars_tbl = Table()
    stars_tbl['x'] = peaks_tbl['x_peak']
    stars_tbl['y'] = peaks_tbl['y_peak']

    from astropy.stats import sigma_clipped_stats
    mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.)
    data -= median_val

    from astropy.nddata import NDData
    nddata = NDData(data=data)

    from photutils.psf import extract_stars
    stars = extract_stars(nddata, stars_tbl, size=25)

    from photutils import EPSFBuilder
    epsf_builder = EPSFBuilder(oversampling=4, maxiters=3,
                               progress_bar=False)
    epsf, fitted_stars = epsf_builder(stars)

    import matplotlib.pyplot as plt
    norm = simple_norm(epsf.data, 'log', percent=99.)
    plt.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
    plt.colorbar()

The :class:`~photutils.psf.EPSFModel` object is a subclass of
:class:`~photutils.psf.FittableImageModel`, thus it can be used as a
the PSF model for the `PSF-fitting machinery in Photutils
<http://photutils.readthedocs.io/en/latest/psf.html>`_ (i.e.
`~photutils.psf.BasicPSFPhotometry`,
`~photutils.psf.IterativelySubtractedPSFPhotometry`, or
`~photutils.psf.DAOPhotPSFPhotometry`).
