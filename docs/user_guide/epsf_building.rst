.. _build-epsf:

Building an effective Point Spread Function (ePSF)
==================================================

The ePSF
--------

The instrumental PSF is a combination of many factors that are
generally difficult to model. `Anderson and King 2000 (PASP 112, 1360)
<https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A/abstract>`_
showed that accurate stellar photometry and astrometry can be derived
by modeling the net PSF, which they call the effective PSF (ePSF). The
ePSF is an empirical model describing what fraction of a star's light
will land in a particular pixel. The constructed ePSF may be oversampled
with respect to the detector pixels.

Oversampling matters when the PSF is undersampled by the detector, e.g.,
a FWHM of only one or two pixels. Since stars can land at fractional
pixel positions on the detector, the appearance of such a PSF varies
with the star's position within a pixel, and an oversampled ePSF
captures this pixel-phase variation so that the PSF can be interpolated
to the exact position of any star. When the PSF is well sampled (a FWHM
of a few pixels or more), an ePSF with no oversampling already captures
its shape, and a larger oversampling factor only adds noise and requires
more stars (see :ref:`epsf-guidelines`).


Building an ePSF
----------------

Photutils provides tools for building an ePSF following the
prescription of `Anderson and King 2000 (PASP 112, 1360)
<https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A/abstract>`_
and subsequent enhancements detailed mainly
in `Anderson 2016 (WFC3 ISR 2016-12)
<https://ui.adsabs.harvard.edu/abs/2016wfc..rept...12A/abstract>`_.
The process iteratively refines the ePSF model and star positions: the
current ePSF is fitted to the stars to improve their centers, and then
the ePSF is rebuilt using the improved star positions.

To begin, we must first define a sample of stars used to build the
ePSF. Ideally these stars should be bright (high S/N) and isolated to
prevent contamination from nearby stars. One may use the star-finding
tools in Photutils (e.g., :class:`~photutils.detection.DAOStarFinder`
or :class:`~photutils.detection.IRAFStarFinder`) to identify an initial
sample of stars. However, the step of creating a good sample of stars
generally requires visual inspection and manual selection to ensure
stars are sufficiently isolated and of good quality (e.g., no cosmic
rays, detector artifacts, etc.). To produce a good ePSF, one should
have a reasonably large sample of stars (e.g., several hundred for an
oversampling factor of 4) in order to sample the PSF at all subpixel
phases and to help reduce the effects of noise. Otherwise, the resulting
ePSF may be noisy or biased. See :ref:`epsf-guidelines` for guidance on
choosing the oversampling factor and the star sample.

Let's start by loading a simulated HST/WFC3 image in the F160W band::

    >>> from photutils.datasets import load_simulated_hst_star_image
    >>> hdu = load_simulated_hst_star_image()  # doctest: +REMOTE_DATA
    >>> data = hdu.data  # doctest: +REMOTE_DATA

The simulated image does not contain any background or noise, so let's
add those to the image::

    >>> from photutils.datasets import make_noise_image
    >>> data += make_noise_image(data.shape, distribution='gaussian',
    ...                          mean=10.0, stddev=5.0, seed=0)  # doctest: +REMOTE_DATA

Let's show the image:

.. plot::

    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm
    from photutils.datasets import (load_simulated_hst_star_image,
                                    make_noise_image)

    hdu = load_simulated_hst_star_image()
    data = hdu.data
    data += make_noise_image(data.shape, distribution='gaussian', mean=10.0,
                             stddev=5.0, seed=0)

    fig, ax = plt.subplots(figsize=(8, 8))
    norm = simple_norm(data, 'sqrt', percent=99.0)
    ax.imshow(data, norm=norm, origin='lower')

For this example we'll use the
:class:`~photutils.detection.DAOStarFinder` class to identify the
brighter stars and their initial positions::

    >>> from photutils.detection import DAOStarFinder
    >>> finder = DAOStarFinder(threshold=100.0, fwhm=1.5)  # doctest: +REMOTE_DATA
    >>> sources = finder(data)  # doctest: +REMOTE_DATA
    >>> for col in sources.colnames:  # doctest: +REMOTE_DATA
    ...     if col not in ('id', 'n_pixels'):
    ...         sources[col].info.format = '%.2f'  # for consistent table output
    >>> sources.pprint(max_width=76)  # doctest: +REMOTE_DATA
     id x_centroid y_centroid sharpness ...   peak    flux    mag   daofind_mag
    --- ---------- ---------- --------- ... ------- -------- ------ -----------
      1     848.53       2.15      0.87 ... 1062.18  4258.95  -9.07       -2.41
      2     181.85       3.74      0.91 ... 1722.27  5828.71  -9.41       -2.93
      3     323.87       3.69      0.91 ... 3016.37 10252.06 -10.03       -3.55
      4      99.89       8.95      0.96 ... 1144.52  3496.04  -8.86       -2.47
      5     824.12       9.36      0.90 ... 1311.20  4685.32  -9.18       -2.64
    ...        ...        ...       ... ...     ...      ...    ...         ...
    478     888.44     991.86      0.85 ...  194.27  1005.88  -7.51       -0.52
    479     114.16     993.40      0.84 ... 1588.31  6810.15  -9.58       -2.84
    480     298.36     993.87      0.84 ...  655.37  2979.57  -8.69       -1.88
    481     207.21     998.17      0.91 ... 2811.02  8614.10  -9.84       -3.48
    482     691.02     998.77      0.98 ... 2611.22  5768.68  -9.40       -3.39
    Length = 482 rows

Let's show the detected stars overlaid on the image:

.. plot::

    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm
    from photutils.datasets import (load_simulated_hst_star_image,
                                    make_noise_image)
    from photutils.detection import DAOStarFinder

    hdu = load_simulated_hst_star_image()
    data = hdu.data
    data += make_noise_image(data.shape, distribution='gaussian', mean=10.0,
                             stddev=5.0, seed=0)

    finder = DAOStarFinder(threshold=100.0, fwhm=1.5)
    sources = finder(data)

    fig, ax = plt.subplots(figsize=(8, 8))
    norm = simple_norm(data, 'sqrt', percent=99.0)
    ax.imshow(data, norm=norm, origin='lower')
    ax.scatter(sources['x_centroid'], sources['y_centroid'],
               s=80, edgecolor='red', facecolor='none', lw=1.5)

Note that the stars are sufficiently separated in the simulated image
that we do not need to exclude any stars due to crowding. In practice
this step will require some manual inspection and selection.


Extracting Star Cutouts
-----------------------

Next, we need to extract cutouts of the stars using the
:func:`~photutils.psf.extract_stars` function. This function requires
a table of star positions either in pixel or sky coordinates. For this
example we are using pixel coordinates, which need to be in table
columns called ``x`` and ``y``.

We'll extract 25 x 25 pixel cutouts of our selected stars. Let's
explicitly exclude stars that are too close to the image boundaries
(because they cannot be extracted)::

    >>> size = 25
    >>> hsize = (size - 1) / 2
    >>> x = sources['x_centroid']  # doctest: +REMOTE_DATA
    >>> y = sources['y_centroid']  # doctest: +REMOTE_DATA
    >>> mask = ((x > hsize) & (x < (data.shape[1] - 1 - hsize)) &
    ...         (y > hsize) & (y < (data.shape[0] - 1 - hsize)))  # doctest: +REMOTE_DATA

Now let's create the table of good star positions::

    >>> from astropy.table import Table
    >>> stars_tbl = Table()
    >>> stars_tbl['x'] = x[mask]  # doctest: +REMOTE_DATA
    >>> stars_tbl['y'] = y[mask]  # doctest: +REMOTE_DATA

The star cutouts from which we build the ePSF must have the
background subtracted. Here we'll use the sigma-clipped median value
as the background level. If the background in the image varies
across the image, one should use more sophisticated methods (e.g.,
`~photutils.background.Background2D`).

The background level must be measured from pixels that are free of
star light. The extended wings of the stars cover a large fraction of
this image, and sigma clipping does not remove them. The median of the
whole image is therefore biased high by about 0.35 counts. That is a
small fraction of the noise, but summed over a 25 x 25 pixel cutout it
is about 3% of the flux of a typical star in this image, and
subtracting it would make the ePSF too concentrated. To avoid this
bias, we first mask the pixels within 18 pixels of each detected
star::

    >>> import numpy as np
    >>> from photutils.utils import circular_footprint
    >>> from scipy.ndimage import binary_dilation
    >>> star_mask = np.zeros(data.shape, dtype=bool)  # doctest: +REMOTE_DATA
    >>> yidx = np.round(sources['y_centroid']).astype(int)  # doctest: +REMOTE_DATA
    >>> xidx = np.round(sources['x_centroid']).astype(int)  # doctest: +REMOTE_DATA
    >>> star_mask[yidx, xidx] = True  # doctest: +REMOTE_DATA
    >>> star_mask = binary_dilation(
    ...     star_mask, structure=circular_footprint(18))  # doctest: +REMOTE_DATA

Now let's subtract the background, measured from the unmasked pixels,
from the image::

    >>> from astropy.stats import sigma_clipped_stats
    >>> mean_val, median_val, std_val = sigma_clipped_stats(
    ...     data, sigma=2.0, mask=star_mask)  # doctest: +REMOTE_DATA
    >>> data -= median_val  # doctest: +REMOTE_DATA

The :func:`~photutils.psf.extract_stars` function requires the input
data as an `~astropy.nddata.NDData` object. An `~astropy.nddata.NDData`
object is easy to create from our data array::

    >>> from astropy.nddata import NDData
    >>> nddata = NDData(data=data)  # doctest: +REMOTE_DATA

We are now ready to create our star cutouts using the
:func:`~photutils.psf.extract_stars` function. For this simple example
we are extracting stars from a single image using a single catalog. The
:func:`~photutils.psf.extract_stars` function can also extract stars
from multiple images using a separate catalog for each image or a single
catalog. When using a single catalog with multiple images, the star
positions must be in sky coordinates (as `~astropy.coordinates.SkyCoord`
objects) and the `~astropy.nddata.NDData` objects must contain valid
`~astropy.wcs.WCS` objects. In the case of using multiple images (i.e.,
dithered images) and a single catalog, the same physical star will be
"linked" across images, meaning it will be constrained to have the same
sky coordinate and, by default, the same flux in each input image (see
:ref:`epsf-linked-stars`).

Let's extract the 25 x 25 pixel cutouts of our selected stars::

    >>> from photutils.psf import extract_stars
    >>> stars = extract_stars(nddata, stars_tbl, size=25)  # doctest: +REMOTE_DATA

The function returns an `~photutils.psf.EPSFStars` object containing the
cutouts of our selected stars that will be used to build the ePSF. Let's
show the first 25 of them:

.. doctest-skip::

    >>> import matplotlib.pyplot as plt
    >>> from astropy.visualization import simple_norm
    >>> nrows = 5
    >>> ncols = 5
    >>> fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),
    ...                        squeeze=True)
    >>> ax = ax.ravel()
    >>> for i in range(nrows * ncols):
    ...     norm = simple_norm(stars[i], 'log', percent=99.0)
    ...     ax[i].imshow(stars[i], norm=norm, origin='lower')

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.nddata import NDData
    from astropy.stats import sigma_clipped_stats
    from astropy.table import Table
    from astropy.visualization import simple_norm
    from photutils.datasets import (load_simulated_hst_star_image,
                                    make_noise_image)
    from photutils.detection import DAOStarFinder
    from photutils.psf import extract_stars
    from photutils.utils import circular_footprint
    from scipy.ndimage import binary_dilation

    hdu = load_simulated_hst_star_image()
    data = hdu.data
    data += make_noise_image(data.shape, distribution='gaussian', mean=10.0,
                             stddev=5.0, seed=0)
    finder = DAOStarFinder(threshold=100.0, fwhm=1.5)
    sources = finder(data)

    size = 25
    hsize = (size - 1) / 2
    x = sources['x_centroid']
    y = sources['y_centroid']
    mask = ((x > hsize) & (x < (data.shape[1] - 1 - hsize))
            & (y > hsize) & (y < (data.shape[0] - 1 - hsize)))
    stars_tbl = Table()
    stars_tbl['x'] = x[mask]
    stars_tbl['y'] = y[mask]

    star_mask = np.zeros(data.shape, dtype=bool)
    yidx = np.round(sources['y_centroid']).astype(int)
    xidx = np.round(sources['x_centroid']).astype(int)
    star_mask[yidx, xidx] = True
    star_mask = binary_dilation(star_mask, structure=circular_footprint(18))
    mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.0,
                                                        mask=star_mask)
    data -= median_val

    nddata = NDData(data=data)

    stars = extract_stars(nddata, stars_tbl, size=25)

    nrows = 5
    ncols = 5
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),
                           squeeze=True)
    ax = ax.ravel()
    for i in range(nrows * ncols):
        norm = simple_norm(stars[i], 'log', percent=99.0)
        ax[i].imshow(stars[i], norm=norm, origin='lower')


Constructing the ePSF
---------------------

With the star cutouts, we are ready to construct the ePSF with the
:class:`~photutils.psf.EPSFBuilder` class. We'll create an ePSF
with an oversampling factor of 4, which is appropriate for these
undersampled stars (a FWHM of about 1.5 pixels). We use the default
maximum of 10 iterations (``maxiters=10``). The build stops early once
the star centers have converged. Do not stop the build after only a few
iterations. An ePSF that has not converged can differ from the true
ePSF by a few percent of its peak, and the fitted star positions are
less accurate. The
:class:`~photutils.psf.EPSFBuilder` class has many options to control
the ePSF build process, including the smoothing kernel, the fitting box,
the recentering function, and the convergence criterion. Please see the
:class:`~photutils.psf.EPSFBuilder` documentation for further details.

We first initialize an :class:`~photutils.psf.EPSFBuilder` instance with
our desired parameters and then input the cutouts of our selected stars
to the instance::

    >>> from photutils.psf import EPSFBuilder
    >>> epsf_builder = EPSFBuilder(oversampling=4,
    ...                            progress_bar=False)  # doctest: +REMOTE_DATA
    >>> result = epsf_builder(stars)  # doctest: +REMOTE_DATA

The :class:`~photutils.psf.EPSFBuilder` returns an
`~photutils.psf.EPSFBuildResults` object containing the constructed ePSF,
the fitted stars, and detailed information about the build process. This
result object supports tuple unpacking, so both of the following work::

    >>> # Access result attributes
    >>> epsf = result.epsf  # doctest: +REMOTE_DATA
    >>> fitted_stars = result.fitted_stars  # doctest: +REMOTE_DATA

    >>> # Tuple unpacking also works
    >>> epsf, fitted_stars = result  # doctest: +REMOTE_DATA

The `~photutils.psf.EPSFBuildResults` object provides useful diagnostic
information about the build process::

    >>> result.converged  # doctest: +REMOTE_DATA
    True
    >>> result.iterations  # doctest: +REMOTE_DATA
    10
    >>> result.n_excluded_stars  # doctest: +REMOTE_DATA
    0

The results also report the fraction of stars whose centers converged
(``final_converged_fraction``), the largest center movement in the final
iteration (``final_center_accuracy``), and the smoothing kernel and
fitting box that were used (``smoothing_kernel`` and ``fit_shape``). See
`~photutils.psf.EPSFBuildResults` for the full list.

The returned ``epsf`` is an `~photutils.psf.ImagePSF` object, and
``fitted_stars`` is a new `~photutils.psf.EPSFStars` object with the
updated star positions and fluxes from fitting the final ePSF model.

Finally, let's show the constructed ePSF:

.. doctest-skip::

    >>> import matplotlib.pyplot as plt
    >>> from astropy.visualization import simple_norm
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> norm = simple_norm(epsf.data, 'log', percent=99.0)
    >>> axim = ax.imshow(epsf.data, norm=norm, origin='lower')
    >>> fig.colorbar(axim)

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.nddata import NDData
    from astropy.stats import sigma_clipped_stats
    from astropy.table import Table
    from astropy.visualization import simple_norm
    from photutils.datasets import (load_simulated_hst_star_image,
                                    make_noise_image)
    from photutils.detection import DAOStarFinder
    from photutils.psf import EPSFBuilder, extract_stars
    from photutils.utils import circular_footprint
    from scipy.ndimage import binary_dilation

    hdu = load_simulated_hst_star_image()
    data = hdu.data
    data += make_noise_image(data.shape, distribution='gaussian', mean=10.0,
                             stddev=5.0, seed=0)

    finder = DAOStarFinder(threshold=100.0, fwhm=1.5)
    sources = finder(data)

    size = 25
    hsize = (size - 1) / 2
    x = sources['x_centroid']
    y = sources['y_centroid']
    mask = ((x > hsize) & (x < (data.shape[1] - 1 - hsize))
            & (y > hsize) & (y < (data.shape[0] - 1 - hsize)))
    stars_tbl = Table()
    stars_tbl['x'] = x[mask]
    stars_tbl['y'] = y[mask]

    star_mask = np.zeros(data.shape, dtype=bool)
    yidx = np.round(sources['y_centroid']).astype(int)
    xidx = np.round(sources['x_centroid']).astype(int)
    star_mask[yidx, xidx] = True
    star_mask = binary_dilation(star_mask, structure=circular_footprint(18))
    mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.0,
                                                        mask=star_mask)
    data -= median_val

    nddata = NDData(data=data)

    stars = extract_stars(nddata, stars_tbl, size=25)

    epsf_builder = EPSFBuilder(oversampling=4, progress_bar=False)
    epsf, fitted_stars = epsf_builder(stars)

    fig, ax = plt.subplots(figsize=(8, 8))
    norm = simple_norm(epsf.data, 'log', percent=99.0)
    axim = ax.imshow(epsf.data, norm=norm, origin='lower')
    fig.colorbar(axim)

The `~photutils.psf.ImagePSF` object can be
used as a PSF model for :ref:`PSF Photometry
<psf-photometry>` (i.e., `~photutils.psf.PSFPhotometry` or
`~photutils.psf.IterativePSFPhotometry`).


Customizing the ePSF Builder
----------------------------

The :class:`~photutils.psf.EPSFBuilder` class provides several options
to customize the ePSF build process.

Smoothing Kernel
^^^^^^^^^^^^^^^^

The ``smoothing_kernel`` parameter controls the smoothing applied to
the ePSF during each iteration. The smoothing helps to reduce noise
in the ePSF, especially when the star sample is small or noisy. The
smoothing kernels are least-squares polynomial smoothers. Each grid
value is replaced by the value at the center of a polynomial fit to
the surrounding grid values, which removes noise while preserving the
polynomial shape of the ePSF within the kernel window.

The default is ``'auto'``, which uses a quartic (fourth-degree)
polynomial kernel whose width is 0.7 times the FWHM of the ePSF in
oversampled grid points, measured in each iteration along its narrowest
axis. The width is rounded down to an odd number of grid points, and no
smoothing is applied when it would be smaller than 5 grid points, i.e.,
for undersampled ePSFs with fewer than about 7 grid points per FWHM,
where a fixed 5x5 kernel would lower the peak of the ePSF. The kernel is
square, so with anisotropic oversampling the axis with the fewer grid
points per FWHM sets its size. The chosen kernel is reported in the
``smoothing_kernel`` attribute of the results, and it can be input as a
fixed ``smoothing_kernel`` to reproduce the build. If the FWHM cannot be
measured, the ``'quartic'`` kernel is used and a warning is emitted.

You can also use ``'quartic'`` or ``'quadratic'`` for the fixed 5x5
fourth- and second-degree polynomial kernels of `Anderson and King 2000
(PASP 112, 1360)
<https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A/abstract>`_,
provide a custom 2D array, or set it to `None` for no smoothing::

    >>> epsf_builder = EPSFBuilder(oversampling=4,
    ...                            smoothing_kernel='quadratic',
    ...                            progress_bar=False)  # doctest: +REMOTE_DATA

The fixed kernels are applied on the oversampled grid, so their physical
width is ``5 / oversampling`` detector pixels. The 5x5 quartic kernel
was developed for HST data with an oversampling factor of 4, where it
is about 0.7 FWHM wide. When using a fixed kernel for an undersampled
ePSF with fewer than about seven grid points per FWHM, the kernel lowers
the peak of the ePSF, and ``smoothing_kernel=None`` is a better choice,
especially when the stars have high signal-to-noise. Smoothing is most
useful for well-sampled ePSFs built from noisy or few stars.

Alias Filter Passband
^^^^^^^^^^^^^^^^^^^^^

Independently of the smoothing kernel, when the oversampling factor is
greater than one the builder applies a low-pass filter to the ePSF in
every iteration. The filter has unit gain up to a passband frequency,
a smooth transition, and zero gain at and above one cycle per detector
pixel. A pixel-integrated PSF has essentially no signal at one cycle
per pixel, but that is the frequency at which the pixel sampling of the
stars aliases onto the oversampled grid. Together with depositing each
star pixel residual on the oversampled grid points within 0.375 pixel
of the pixel center along each axis, the filter prevents noise from
heterogeneous, contaminated, or low signal-to-noise stars from growing
into a checkerboard pattern in the ePSF.

The ``alias_passband`` parameter sets the end of the passband in cycles
per detector pixel. The default (``'auto'``) is 0.8 cycles per pixel, or
0.7 for an oversampling factor of 2. The default is the best choice for
most data. A different value can help in two cases, which depend on how
much real signal the ePSF has just below one cycle per pixel. That is
set by the optical cutoff frequency of the telescope expressed in cycles
per pixel:

.. math::

    \nu_c = \frac{D \, p}{\lambda}

where :math:`D` is the telescope diameter, :math:`\lambda` is the
shortest wavelength of the bandpass (in the same units as :math:`D`),
and :math:`p` is the pixel scale in radians per pixel. A telescope
transmits no signal above this frequency. For example, for HST (:math:`D
= 2.4` m) WFC3/IR (0.13 arcsec per pixel) at 1.1 microns, :math:`\nu_c
= 2.4 \times 6.3 \times 10^{-7} / 1.1 \times 10^{-6} = 1.4` cycles per
pixel.

.. list-table::
    :header-rows: 1
    :widths: 22 33 45

    * - :math:`\nu_c` (cycles/pixel)
      - Examples
      - Guidance for ``alias_passband``
    * - greater than about 1
      - HST WFC3/IR F110W, JWST NIRCam F070W, JWST NIRISS F090W, Roman
        WFI F062 and F106
      - The default, or 0.9. The ePSF has real signal up to nearly one
        cycle per pixel. In tests the default recovered the peak of
        these ePSFs to within about 1 percent, except for HST WFC3/IR
        F110W (3 percent low), which 0.9 recovered to within 0.2
        percent. For the others 0.9 increased the noise in the core by
        10 to 65 percent.
    * - about 0.9 to 1
      - HST WFC3/IR F160W
      - The default (0.8)
    * - less than about 0.9
      - JWST NIRCam F115W and redder, JWST MIRI, Roman WFI F158 and
        F213, most ground-based data
      - The default, or 0.7. There is no signal to preserve near one
        cycle per pixel. In tests 0.7 lowered the residuals in the core
        by 10 to 40 percent and converged in fewer iterations.

Try 0.9 for a strongly undersampled detector if the default ePSF is too
broad, i.e., if the stars have positive residuals at their centers after
the fitted ePSF is subtracted::

    >>> epsf_builder = EPSFBuilder(oversampling=4, alias_passband=0.9,
    ...                            maxiters=20,
    ...                            progress_bar=False)  # doctest: +REMOTE_DATA

Do not use 0.7 unless the cutoff frequency is known to be low. It leaves
the peak of a strongly undersampled ePSF low by 2 to 6 percent.

A passband that is wider than needed has a cost. A star that is sampled
once per pixel constrains the frequencies near one cycle per pixel only
weakly, because a small shift of the star center has nearly the same
effect on its pixel values. A wider passband therefore makes the build
converge more slowly and makes it more sensitive to noise. Use 0.9 only
with a large star sample (a few hundred stars), allow more iterations
(``maxiters`` of 20 or more), and check that the build converged. Do not
use it with an oversampling factor of 2.

The filter acts separately along the x and y axes. The signal that it
removes from an undersampled ePSF therefore shows as a faint ripple
pattern, with a period of about one pixel, along the row and the column
through the center of the ePSF. To remove this pattern, the builder
refines the ePSF after the building iterations. Each refinement
iteration (``refinement_iters``, 5 by default) updates the ePSF five
times with the star centers and fluxes held fixed and then refits the
stars with the updated ePSF. These updates use a wider and smoother
low-pass filter, with unit gain up to 1.1 cycles per pixel and zero
gain at and above 1.33 cycles per pixel. It does not remove signal near
one cycle per pixel, so it leaves no ripple pattern. It removes only
the frequencies that the star residuals do not constrain. Such a filter
cannot be used from the start of the build, because the build then
converges slowly and is more sensitive to the initial star centers.

The refinement is applied only for an oversampling factor of 4 or
larger, and it roughly doubles the run time of the build. For a
well-sampled ePSF it has little to restore and it adds a small amount
of noise (up to about 10 percent of the residual of the ePSF). Set
``refinement_iters=0`` to skip it.

Setting ``alias_passband=None`` turns the filter off. This is rarely
appropriate. Without the filter, noise at the alias frequencies
accumulates over the iterations, the build can stall before it
converges, and heterogeneous or contaminated star samples can grow a
checkerboard pattern. In tests with simulated HST, JWST, and Roman star
fields, the unfiltered ePSF was less accurate than the filtered one in
nearly every case, even for large, clean, and homogeneous star samples.
The option is provided for experimentation, e.g., to check how much
the filter changes a particular ePSF. Always compare the result with
a filtered build. The filter is never applied along an axis with an
oversampling factor of 1.

If the subpixel phases of the fitted star centers are strongly
non-uniform at the end of the build, which indicates biased star
centers, a warning is emitted. In that case the star sample should be
inspected for stars with different PSFs, saturated or contaminated
cutouts, or spurious detections.

.. _epsf-linked-stars:

Linked Stars from Dithered Images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the same star is observed in several dithered images, the cutouts
can be linked as a `~photutils.psf.LinkedEPSFStar` (this happens
automatically when :func:`~photutils.psf.extract_stars` is given
multiple images and a single catalog of sky coordinates). After each
fitting iteration, the builder constrains the centers of the linked
stars to a single sky coordinate and, by default, their fluxes to
their mean value. Averaging both the positions and the fluxes across
dithers is the key step of `Anderson and King 2000 (PASP 112, 1360)
<https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A/abstract>`_
that breaks the degeneracy between the flux of a star and its subpixel
position caused by intra-pixel sensitivity variations. Without it, the
pixel-phase dependence of the individual flux measurements is absorbed
into the ePSF. The flux constraint assumes that the linked images have
the same flux scale (e.g., the same exposure time and throughput). If
they do not, set ``constrain_fluxes=False``::

    >>> epsf_builder = EPSFBuilder(oversampling=4,
    ...                            constrain_fluxes=False,
    ...                            progress_bar=False)  # doctest: +REMOTE_DATA

To link stars across images, provide a single catalog with sky
coordinates and multiple `~astropy.nddata.NDData` objects, each with a
valid WCS:

.. doctest-skip::

    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord
    >>> catalog = Table()
    >>> catalog['skycoord'] = SkyCoord(ra=[...]*u.deg, dec=[...]*u.deg)
    >>> stars = extract_stars([nddata1, nddata2], catalog, size=25)

Customizing the ePSF Fitting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~photutils.psf.EPSFBuilder` class allows you to customize
the fitting process using the ``fit_shape`` parameter. This parameter
specifies the size of the box (in detector pixels) centered on each
star used for fitting. The default is ``'auto'``, which uses a square
box of twice the FWHM of the ePSF in detector pixels (measured in
each iteration along its narrowest axis), with a minimum of 5 pixels
and a maximum of the star cutout size. The chosen box is reported in
the ``fit_shape`` attribute of the results. A fixed box can be given
instead. A smaller box speeds up the fitting, but it should still cover
the core of the star. A box that is much smaller than the star uses only
its flat core, which biases the fitted centers and can prevent the build
from converging. The 5-pixel box of Anderson and King is about 2.5 FWHM
wide for HST data but only about 1 FWHM wide for a star with a FWHM of 5
pixels::

    >>> epsf_builder = EPSFBuilder(oversampling=4,
    ...                            fit_shape=7,
    ...                            progress_bar=False)  # doctest: +REMOTE_DATA

You can also customize the fitter itself by passing a
`~astropy.modeling.fitting.Fitter` instance::

    >>> from astropy.modeling.fitting import LMLSQFitter
    >>> fitter = LMLSQFitter()  # doctest: +REMOTE_DATA
    >>> epsf_builder = EPSFBuilder(oversampling=4,
    ...                            fitter=fitter, fit_shape=7,
    ...                            progress_bar=False)  # doctest: +REMOTE_DATA

Sigma Clipping
^^^^^^^^^^^^^^

The ``sigma_clip`` parameter controls the sigma clipping applied when
stacking the ePSF residuals in each iteration. The default uses sigma
clipping with ``sigma=3.0`` and ``maxiters=10``. You can provide your
own `~astropy.stats.SigmaClip` instance to customize this behavior::

    >>> from astropy.stats import SigmaClip
    >>> sigclip = SigmaClip(sigma=2.5, maxiters=5)  # doctest: +REMOTE_DATA
    >>> epsf_builder = EPSFBuilder(oversampling=4,
    ...                            sigma_clip=sigclip,
    ...                            progress_bar=False)  # doctest: +REMOTE_DATA

Setting ``sigma_clip=None`` disables sigma clipping entirely.


Including Weights
-----------------

If your input `~astropy.nddata.NDData` object contains uncertainty
information, the :func:`~photutils.psf.extract_stars` function will
automatically create weights for each star cutout. These weights are
used during the ePSF fitting process to give more weight to pixels with
lower uncertainties.

To include weights, provide an ``uncertainty`` attribute in
your `~astropy.nddata.NDData` object. The uncertainty can be
any of the `~astropy.nddata.NDUncertainty` subclasses (e.g.,
`~astropy.nddata.StdDevUncertainty`)::

    >>> import numpy as np
    >>> from astropy.nddata import StdDevUncertainty
    >>> uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))  # doctest: +REMOTE_DATA, +SKIP
    >>> nddata = NDData(data=data, uncertainty=uncertainty)  # doctest: +REMOTE_DATA, +SKIP



.. _epsf-guidelines:

Guidelines for Building a Good ePSF
-----------------------------------

The quality of an ePSF depends more on the input stars and on a
sensible choice of the oversampling factor than on the other builder
parameters. The following guidelines are based on `Anderson and King
2000 (PASP 112, 1360)
<https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A/abstract>`_ and
on the systematic tests of `Godden and Blundell 2026 (RASTI 5, 1)
<https://doi.org/10.1093/rasti/rzaf063>`_.

Choosing the oversampling factor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ePSF is tabulated on a grid with a spacing of ``1 / oversampling``
detector pixels and is evaluated between grid points by cubic spline
interpolation. The interpolation is accurate when there are at least
about four grid points per FWHM of the ePSF, so a good rule of thumb
is ``oversampling >= 4 / FWHM`` with the FWHM in pixels (measured
along the narrowest direction of an elongated PSF). For example, use
an oversampling of 3 or 4 for a FWHM of 1.5 pixels, 2 for a FWHM of 2
pixels, and 1 for a FWHM of 4 pixels or more.

Do not use a larger oversampling factor than the data require. A
pixel-integrated PSF has essentially no structure on scales smaller
than a pixel once the PSF is well sampled, so extra grid points add
no information. They do, however, divide the star samples among more
grid cells and make the ePSF noisier, and they require more stars. For
well-sampled data (a FWHM of a few pixels or more), an oversampling of 1
is usually the best choice.

Choosing the star sample
^^^^^^^^^^^^^^^^^^^^^^^^

Each of the ``oversampling**2`` subpixel cells within a pixel must
be sampled by the centers of several stars. With randomly placed
stars, plan on at least about 10 stars per cell, i.e., roughly ``10 *
oversampling**2`` stars (about 40 for an oversampling of 2, 90 for 3,
and 160 for 4), and considerably more if the stars are faint. Godden
and Blundell estimate that about 240 randomly placed stars are needed
for an oversampling of 4 to have a 95 percent probability of at least
six samples in every cell. A set of exposures dithered by fractions of
a pixel that uniformly cover the subpixel phases is far more effective
than random placement and also allows the star fluxes and positions to
be constrained across images (see :ref:`epsf-linked-stars`).

The stars should be bright but unsaturated, isolated (no neighbors
within the cutout), free of cosmic rays and detector artifacts, and have
a clean background subtraction so that the total flux of each cutout
is a reliable normalization. Just as important, all of the stars must
share the same PSF. Do not combine exposures with different seeing or
focus, and do not mix regions of the field where the PSF differs unless
the variation is small compared to the accuracy you need. Heterogeneous
stars produce pixel-to-pixel noise in the oversampled grid that biases
the fitted star centers toward particular subpixel phases, and the
builder emits a warning if the subpixel phases of the fitted centers are
strongly non-uniform at the end of the build. In that case, inspect the
star sample rather than increasing the number of iterations.

Finally, check the result. The subpixel phases of the fitted star
centers should be uniformly distributed, and the fitted fluxes and
positions of the stars (or of an independent set of stars) should not
depend on their subpixel phase.
