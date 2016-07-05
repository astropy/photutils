Background and Background Noise Estimation
==========================================

Introduction
------------

To accurately measure the photometry and morphological properties of
astronomical sources, one requires an accurate estimate of the
background, which can be from both the sky and the detector.
Similarly, having an accurate estimate of the background rms noise is
important for determining the significance of source detections and
for estimating photometric errors.

Unfortunately, accurate background and background noise estimation is
a difficult task.  Further, because astronomical images can cover a
wide variety of scenes, there is not a single background estimation
method that will always be applicable.  Photutils provides some tools
for estimating the background and background noise in your data, but
ultimately you have the flexibility of determining the background most
appropriate for your data.


Scalar Background Level and Noise Estimation
--------------------------------------------

Simple Statistics
^^^^^^^^^^^^^^^^^

If the background level and noise are relatively constant across an
image, the simplest way to estimate these values is to derive scalar
values for these quantities using simple approximations.  Of course,
when computing the image statistics one must take into account the
astronomical sources present in the images, which add a positive tail
to the distribution of pixel intensities.  For example, one may
consider using the image median as the background level and the image
standard deviation as the 1-sigma background noise, but the resulting
values are obviously biased by the presence of real sources.

A slightly better method involves using statistics that are robust
against the presence of outliers, such as the biweight location for
the background level and biweight midvariance or `median absolute
deviation (MAD)
<http://en.wikipedia.org/wiki/Median_absolute_deviation>`_ for the
background noise estimation.  However, for most astronomical scenes
these methods will also be biased by the presence of astronomical
sources in the image.

As an example, we load a synthetic image comprised of 100 sources with
a Gaussian-distributed background whose mean is 5 and standard
deviation is 2::

    >>> from photutils.datasets import make_100gaussians_image
    >>> data = make_100gaussians_image()

Let's plot the image:

.. doctest-skip::

    >>> import matplotlib.pylab as plt
    >>> from astropy.visualization import SqrtStretch
    >>> from astropy.visualization.mpl_normalize import ImageNormalize
    >>> norm = ImageNormalize(stretch=SqrtStretch())
    >>> plt.imshow(data, norm=norm, origin='lower', cmap='Greys_r')

.. plot::

    import matplotlib.pylab as plt
    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    from photutils.datasets import make_100gaussians_image
    data = make_100gaussians_image()
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(data, norm=norm, origin='lower', cmap='Greys_r')

The image median and biweight location are both larger than the true
background level of 5::

    >>> import numpy as np
    >>> from astropy.stats import biweight_location
    >>> print(np.median(data))
    5.2255295184
    >>> print(biweight_location(data))
    5.1867597555

Similarly, using the biweight midvariance and median absolute
deviation to estimate the background noise level give values that are
larger than the true value of 2::

    >>> from astropy.stats import biweight_midvariance, mad_std
    >>> print(biweight_midvariance(data))   # doctest: +SKIP
    2.22011175104
    >>> print(mad_std(data))    # doctest: +FLOAT_CMP
    2.1443728009


Sigma-Clipped Statistics
^^^^^^^^^^^^^^^^^^^^^^^^

The most widely used technique to remove the sources from the image
statistics is called sigma clipping.  Briefly, pixels that are above
or below a specified sigma level from the median are discarded and the
statistics are recalculated.  The procedure is typically repeated over
a number of iterations or until convergence is reached.  This method
provides a better estimate of the background and background noise
levels::

    >>> from astropy.stats import sigma_clipped_stats
    >>> mean, median, std = sigma_clipped_stats(data, sigma=3.0, iters=5)
    >>> print((mean, median, std))    # doctest: +FLOAT_CMP
    (5.1991386516217908, 5.1555874333582912, 2.0942752121329691)


Masking Sources
^^^^^^^^^^^^^^^

An even better procedure is to exclude the sources in the image by
masking them.  Of course, this technique requires one to `identify the
sources in the data <detection.html>`_, which in turn depends on the
background and background noise.  Therefore, this method for
estimating the background and background rms requires an iterative
procedure.

Photutils provides a convenience function,
:func:`~photutils.detection.make_source_mask`, for creating source
masks.  It uses sigma-clipped statistics as the first estimate of the
background and noise levels for the source detection.  Sources are
then identified using image segmentation.  Finally, the source masks
are dilated to ensure that the extended regions of detected sources
are completely masked.

Here we use a aggressive 2-sigma detection threshold to maximize the
source detections and dilate using a 11x11 box:

.. doctest-requires:: scipy

    >>> from photutils import make_source_mask
    >>> mask = make_source_mask(data, snr=2, npixels=5, dilate_size=11)
    >>> mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
    >>> print((mean, median, std))    # doctest: +FLOAT_CMP
    (5.0010134754755695, 5.0005849056043763, 1.970887100626572)

Of course, the source detection and masking procedure can be iterated
further.  Even with one iteration we are within 0.02% of the true
background and 1.5% of the true background rms.

.. _scipy: http://www.scipy.org/


2D Background Level and Noise Estimation
---------------------------------------------

If the background or the background noise varies across the image,
then you will generally want to generate a 2D image of the background
and background rms (or compute these values locally).  This can be
accomplished by applying the above techniques to subregions of the
image.  A common procedure is to use sigma-clipped statistics in each
mesh of a grid that covers the input data to create a low-resolution
background map.  The final background or background rms map can then
be generated by interpolating the low-resolution map.

Photutils provides the :class:`~photutils.background.Background2D` and
:class:`~photutils.background.BackgroundIDW2D` classes to estimate the
2D background and background rms noise in an astronomical image.  To
resize the low-resolution maps,
:class:`~photutils.background.Background2D` uses bicublic spline
interpolation, while :class:`~photutils.background.BackgroundIDW2D`
uses inverse-distance weighted interpolation (see
`~photutils.utils.ShepardIDWInterpolator`).

Both classes require the size of the box (``box_size``) in which to
estimate the background.  Selecting the box size requires some care by
the user.  The box size should generally be larger than the typical
size of sources in the image, but small enough to encapsulate any
background variations.  For best results, the box size should also be
chosen so that the data are covered by an integer number of boxes in
both dimensions.  The ``edge_method`` keyword is used to determine how
to handle the case where the image size is not an integer multiple of
the ``box_size`` in either dimension.

The background level in each of the meshes is based on sigma-clipped
statistics, where the sigma-clipping is controlled controlled by the
``sigmaclip_sigma`` and the ``sigclip_iters`` input parameters.  The
background level in each mesh is estimated using one of several
defined methods or by using a custom method (see
:ref:`background_methods` below).  The background rms in each mesh is
estimated by the sigma-clipped standard deviation.

After the background has been determined in each of the boxes, the
low-resolution background can be median filtered in a box of size
``filter_size`` to suppress local under- or overestimations (e.g., due
to bright galaxies in a particular box).  Likewise, the median filter
can be applied only to those boxes which are above a specified
threshold (``filter_threshold``).

For this example, we will add a strong background gradient to the
image::

    >>> ny, nx = data.shape
    >>> y, x = np.mgrid[:ny, :nx]
    >>> gradient =  x * y / 5000.
    >>> data2 = data + gradient
    >>> plt.imshow(data2, norm=norm, origin='lower', cmap='Greys_r')    # doctest: +SKIP

.. plot::

    import matplotlib.pylab as plt
    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    from photutils.datasets import make_100gaussians_image
    data = make_100gaussians_image()
    ny, nx = data.shape
    y, x = np.mgrid[:ny, :nx]
    gradient =  x * y / 5000.
    data2 = data + gradient
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(data2, norm=norm, origin='lower', cmap='Greys_r')

We start by creating a `~photutils.background.Background2D` object
using a box size of 50x50, a 3x3 median filter, and the "median"
background method, which estimates the background using the
sigma-clipped median in each box:

.. doctest-requires:: scipy

    >>> from photutils.background import Background2D
    >>> bkg = Background2D(data2, (50, 50), filter_size=(3, 3),
    ...                    method='median')

The 2D background map and background rms maps are retrieved using the
``background`` and ``background_rms`` attributes, respectively.  The
low-resolution versions of these maps are the ``bkg_mesh2d`` and
``bkgrms_mesh2d`` attributes, respectively.   The global median value
of the low-resolution background and background rms maps is provided
with the ``background_median`` and ``background_rms_median``
attributes, respectively:

.. doctest-requires:: scipy

    >>> print(bkg.background_median)
    10.8219978626
    >>> print(bkg.background_rms_median)
    2.29882053968

Let's plot the background image:

.. doctest-skip::

    >>> plt.imshow(bkg.background, origin='lower', cmap='Greys_r')

.. plot::

    import matplotlib.pylab as plt
    from photutils.datasets import make_100gaussians_image
    from photutils.background import Background2D
    data = make_100gaussians_image()
    ny, nx = data.shape
    y, x = np.mgrid[:ny, :nx]
    gradient =  x * y / 5000.
    data2 = data + gradient
    bkg = Background2D(data2, (50, 50), filter_size=(3, 3), method='median')
    plt.imshow(bkg.background, origin='lower', cmap='Greys_r')

and the background-subtracted image:

.. doctest-skip::

    >>> plt.imshow(data2 - bkg.background, norm=norm, origin='lower',
    ...            cmap='Greys_r')

.. plot::

    import matplotlib.pylab as plt
    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    from photutils.datasets import make_100gaussians_image
    from photutils.background import Background2D
    data = make_100gaussians_image()
    ny, nx = data.shape
    y, x = np.mgrid[:ny, :nx]
    gradient =  x * y / 5000.
    data2 = data + gradient
    bkg = Background2D(data2, (50, 50), filter_size=(3, 3), method='median')
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(data2 - bkg.background, norm=norm, origin='lower', cmap='Greys_r')


.. _background_methods:

Background Methods
^^^^^^^^^^^^^^^^^^

The background level in each of the background meshes can by estimated
using one of several defined methods or by using a custom method.  For
all methods, the statistics are calculated from the sigma-clipped data
values in each mesh.  The defined methods are ``'mean'``,
``'median'``, ``'sextractor'``, and ``'mode_estimator'``.  ``'mean'``
and ``'median'`` are simply the sigma-clipped mean and median,
respectively, in each background mesh.  For ``'sextractor'``, the
background in each mesh is ``(2.5 * median) - (1.5 * mean)``.
However, if ``(mean - median) / std > 0.3`` then the ``median`` is
used instead (despite what the `SExtractor
<http://www.astromatic.net/software/sextractor>`_ User's Manual says,
this is the method it always uses).  For ``'mode_estimator'``, the
background is ``(3 * median) - (2 * mean)``.

A custom calculation can also be defined for the background level by
setting ``method='custom'`` and inputing a custom function to the
``backfunc`` parameter.  The custom function must must take in a 2D
`~numpy.ma.MaskedArray` of size ``NxZ``, where the ``Z`` axis (axis=1)
contains the sigma-clipped pixels in each background mesh, and outputs
a 1D `~numpy.ndarray` low-resolution background map of length ``N``.

We demonstrate this capability using a custom function that is simply
the median of the sigma-clipped data in each mesh (this is the same
calculation used by ``method='median'``).  We start by defining the
custom function::

    >>> def backfunc(data):
    ...    return np.ma.median(data, axis=1)

Now we can pass the function to the
`~photutils.background.Background2D` class:

.. doctest-requires:: scipy

    >>> b = Background2D(data, (50, 50), filter_size=(3, 3),
    ...                  method='custom', backfunc=backfunc)
    >>> bkg = b.background


Masking
^^^^^^^

Masks can also be input into `~photutils.background.Background2D`.  As
described above, this can be employed to mask sources in the image to
estimate the background with an iterative procedure.

Additionally, input masks are often necessary if your data array
includes regions without data coverage (e.g., from a rotated image or
an image from a mosaic).  Otherwise the data values in the regions
without coverage (e.g., usually zeros) will adversely contribute to
the background statistics.

Let's create such an image (this requires `scipy`_) and plot it:

.. doctest-requires:: scipy

    >>> from scipy.ndimage import rotate
    >>> data3 = rotate(data2, -45.)
    >>> norm = ImageNormalize(stretch=SqrtStretch())    # doctest: +SKIP
    >>> plt.imshow(data3, origin='lower', cmap='Greys_r', norm=norm)    # doctest: +SKIP

.. plot::

    from photutils.datasets import make_100gaussians_image
    from scipy.ndimage.interpolation import rotate
    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    import matplotlib.pylab as plt
    data = make_100gaussians_image()
    ny, nx = data.shape
    y, x = np.mgrid[:ny, :nx]
    gradient =  x * y / 5000.
    data2 = data + gradient
    data3 = rotate(data2, -45.)
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(data3, origin='lower', cmap='Greys_r', norm=norm)

Now we create a coverage mask and input it into
`~photutils.background.Background2D` to exclude the regions where we
have no data.  For real data, one can usually create a coverage mask
from a weight or rms image.  For this example we also use a smaller
box size to help capture the strong gradient in the background:

.. doctest-requires:: scipy

    >>> mask = (data3 == 0)
    >>> bkg3 = Background2D(data3, (25, 25), filter_size=(3, 3), mask=mask)

Masks are never applied to the returned background map because the
input ``mask`` can represent either a coverage mask or a source mask,
or a combination of both.  We need to manually apply the coverage mask
to the returned background map:

.. doctest-requires:: scipy

    >>> back3 = bkg3.background * ~mask
    >>> norm = ImageNormalize(stretch=SqrtStretch())    # doctest: +SKIP
    >>> plt.imshow(back3, origin='lower', cmap='Greys_r', norm=norm)    # doctest: +SKIP

.. plot::

    from photutils.datasets import make_100gaussians_image
    from photutils.background import Background2D
    from scipy.ndimage.interpolation import rotate
    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    import matplotlib.pylab as plt
    data = make_100gaussians_image()
    ny, nx = data.shape
    y, x = np.mgrid[:ny, :nx]
    gradient =  x * y / 5000.
    data2 = data + gradient
    data3 = rotate(data2, -45.)
    mask = (data3 == 0)
    bkg3 = Background2D(data3, (25, 25), filter_size=(3, 3), mask=mask)
    back3 = bkg3.background * ~mask
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(back3, origin='lower', cmap='Greys_r', norm=norm)

Finally, let's subtract the background from the image and plot it:

.. doctest-skip::

    >>> norm = ImageNormalize(stretch=SqrtStretch())
    >>> plt.imshow(data3 - back3, origin='lower', cmap='Greys_r', norm=norm)

.. plot::

    from photutils.datasets import make_100gaussians_image
    from photutils.background import Background2D
    from scipy.ndimage.interpolation import rotate
    import matplotlib.pylab as plt
    from astropy.visualization import SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    data = make_100gaussians_image()
    ny, nx = data.shape
    y, x = np.mgrid[:ny, :nx]
    gradient =  x * y / 5000.
    data2 = data + gradient
    data3 = rotate(data2, -45.)
    mask = (data3 == 0)
    bkg3 = Background2D(data3, (25, 25), filter_size=(3, 3), mask=mask)
    back3 = bkg3.background * ~mask
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(data3 - back3, origin='lower', cmap='Greys_r', norm=norm)

Some small residual background is still present in the image, but the
background subtraction can be improved through further interations and
by masking the sources.


Reference/API
-------------

.. automodapi:: photutils.background
    :no-heading:
