.. _background:

Background Estimation (`photutils.background`)
==============================================

Introduction
------------

To accurately measure the photometry and morphological properties
of astronomical sources, one requires an accurate estimate of the
background, which can be from both the sky and the detector. Similarly,
having an accurate estimate of the background noise is important for
determining the significance of source detections and for estimating
photometric errors.

Unfortunately, accurate background and background noise estimation is a
difficult task. Further, because astronomical images can cover a wide
variety of scenes, there is not a single background estimation method
that will always be applicable. Photutils provides tools for estimating
the background and background noise in your data, but they will likely
require some tweaking to optimize the background estimate for your data.


Scalar Background and Noise Estimation
--------------------------------------

Simple Statistics
^^^^^^^^^^^^^^^^^

If the background level and noise are relatively constant across an
image, the simplest way to estimate these values is to derive scalar
quantities using simple approximations. When computing the image
statistics one must take into account the astronomical sources present
in the images, which add a positive tail to the distribution of pixel
intensities. For example, one may consider using the image median as
the background level and the image standard deviation as the 1-sigma
background noise, but the resulting values are biased by the presence of
real sources.

A slightly better method involves using statistics that
are robust against the presence of outliers, such as the
biweight location for the background level and biweight
scale or normalized `median absolute deviation (MAD)
<https://en.wikipedia.org/wiki/Median_absolute_deviation>`__ for the
background noise estimation. However, for most astronomical scenes these
methods will also be biased by the presence of astronomical sources in
the image.

As an example, we load a synthetic image composed of 100 sources with
a Gaussian-distributed background whose mean is 5 and standard
deviation is 2::

    >>> from photutils.datasets import make_100gaussians_image
    >>> data = make_100gaussians_image()

Let's plot the image:

.. doctest-skip::

    >>> import matplotlib.pyplot as plt
    >>> from astropy.visualization import simple_norm
    >>> norm = simple_norm(data, 'sqrt', percent=99.5)
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(data, norm=norm, origin='lower')

.. plot::

    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm
    from photutils.datasets import make_100gaussians_image

    data = make_100gaussians_image()
    norm = simple_norm(data, 'sqrt', percent=99.5)
    fig, ax = plt.subplots()
    ax.imshow(data, norm=norm, origin='lower')
    ax.set_title('Data')

The image median and biweight location are both larger than the true
background level of 5::

    >>> import numpy as np
    >>> from astropy.stats import biweight_location
    >>> print(np.median(data))
    5.222396450477202
    >>> print(biweight_location(data))
    5.187556942771537

Similarly, using the median absolute deviation to estimate the
background noise level gives a value that is larger than the true
value of 2::

    >>> from astropy.stats import mad_std
    >>> print(mad_std(data))
    2.1497096320053166


Sigma Clipping Sources
^^^^^^^^^^^^^^^^^^^^^^

Sigma clipping is a general technique for removing outliers from data.
Values that lie more than a specified number of standard deviations
(sigma) above or below a central value, typically the median, are
rejected, and the central value and standard deviation are then
recalculated from the remaining values. The procedure is repeated for a
number of iterations or until no more values are rejected.

In an image where the sources cover only a small fraction of the
pixels, as is typical of extragalactic fields, the background pixels
form the bulk of the distribution of pixel values, and the pixels in
sources form a tail of high values. This is not the case for crowded
fields or for images dominated by extended emission (e.g., a globular
cluster, the Galactic plane, or a galaxy that fills the image), where
the source pixels are not a minority.

When the background pixels dominate, a commonly used technique to
remove the sources from the image statistics is to sigma clip the
pixel values, which treats the bright source pixels as outliers. This
method provides a better estimate of the background and background
noise levels::

    >>> from astropy.stats import sigma_clipped_stats
    >>> mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    >>> print(np.array((mean, median, std)))
    [5.19968673 5.15244174 2.09423739]

However, sigma clipping removes only the source pixels that are
significantly brighter than the noise. The fainter outer regions of the
sources cannot be distinguished from the noise in individual pixels, so
they are not clipped and still bias the background estimate high (here,
the median is 5.15 instead of the true value of 5). Masking the
sources, described next, reduces this bias.


Masking Sources
^^^^^^^^^^^^^^^

An even better procedure is to exclude the sources in the image by
masking them. Making a source mask from detected sources requires one
to :ref:`identify the sources in the data <source_detection>`, which in
turn depends on an estimate of the background and background noise.
The source detection and background estimation can therefore be
iterated to improve both. The Moran's I source mask described in
:ref:`morans_i_mask` does not depend on the background, so it does not
need this iteration.

One method to create a source mask is to use a
:ref:`segmentation image <image_segmentation>`. Here we use the
`~photutils.segmentation.detect_threshold` convenience function to get a
rough estimate of the threshold at the 2-sigma background noise level.
Then we use the `~photutils.segmentation.detect_sources` function to
generate a `~photutils.segmentation.SegmentationImage`. Finally, we use
the :meth:`~photutils.segmentation.SegmentationImage.make_source_mask`
method with a circular dilation footprint to create the source mask::

    >>> from astropy.stats import SigmaClip
    >>> from photutils.segmentation import detect_threshold, detect_sources
    >>> from photutils.utils import circular_footprint
    >>> sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    >>> threshold = detect_threshold(data, n_sigma=2.0, sigma_clip=sigma_clip)
    >>> segment_img = detect_sources(data, threshold, n_pixels=10)
    >>> footprint = circular_footprint(radius=10)
    >>> mask = segment_img.make_source_mask(footprint=footprint)
    >>> mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
    >>> print(np.array((mean, median, std)))
    [5.00257401 4.99641799 1.97009566]

Even with a single pass, we are within 0.1% of the true background value
and 1.5% of the true background RMS.


.. _morans_i_mask:

Masking Sources Using the Moran's I Statistic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A source mask can also be made without first estimating the
background, using the `~photutils.background.make_morans_i_mask`
function. It is based on the `Moran's I
<https://en.wikipedia.org/wiki/Moran%27s_I>`_ statistic, which measures
the spatial autocorrelation of the pixel values. The value of the
statistic at each pixel is calculated from the square patch of pixels
centered on that pixel (21 pixels on a side by default). It measures
how strongly the pixel values in the patch are correlated with the
values of their nearest neighbors. Patches that contain only
uncorrelated background noise have values near zero. Patches that
contain a source, including the faint outer regions of extended
sources, have larger values.

The statistic does not change if a constant is added to the data or if
the data are multiplied by a constant. The source mask therefore does
not depend on the background level or on the amplitude of the background
noise::

    >>> from photutils.background import make_morans_i_mask
    >>> mask = make_morans_i_mask(data)
    >>> mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
    >>> print(np.array((mean, median, std)))  # doctest: +FLOAT_CMP
    [4.9990853  4.99195359 1.96672525]

The results are comparable to those from the segmentation-based mask
above, but neither an iteration nor a detection threshold in data units
was needed.

The statistic is unaffected by the background only where the
background is nearly constant across a patch. A background that
changes by more than about the noise level across a single patch makes
neighboring pixels more alike than distant ones, which is the same
correlation that a source produces. Such a gradient therefore raises
the statistic in every patch. If the whole image has such a strong
gradient, the typical background value of the statistic cannot be
estimated, and few or no sources are flagged. Gradients that are small
compared to the noise across a patch, like the one in the 2D example
below, have little effect.

The map of the statistic can be calculated with the
`~photutils.background.make_morans_i_map` function. Let's plot the
data, the Moran's I map, and the source mask:

.. doctest-skip::

    >>> from photutils.background import make_morans_i_map
    >>> stat = make_morans_i_map(data)
    >>> fig, ax = plt.subplots(3, 1, figsize=(6, 11), layout='constrained')
    >>> ax[0].imshow(data, norm=norm, origin='lower')
    >>> im = ax[1].imshow(stat, origin='lower')
    >>> fig.colorbar(im, ax=ax[1], label="Moran's I")
    >>> ax[2].imshow(mask, origin='lower', cmap='gray')

.. plot::

    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm
    from photutils.background import make_morans_i_map, make_morans_i_mask
    from photutils.datasets import make_100gaussians_image

    data = make_100gaussians_image()
    stat = make_morans_i_map(data)
    mask = make_morans_i_mask(data)

    norm = simple_norm(data, 'sqrt', percent=99.5)
    fig, ax = plt.subplots(3, 1, figsize=(6, 11), layout='constrained')
    ax[0].imshow(data, norm=norm, origin='lower')
    ax[0].set_title('Data')
    im = ax[1].imshow(stat, origin='lower')
    fig.colorbar(im, ax=ax[1], label="Moran's I")
    ax[1].set_title("Moran's I")
    ax[2].imshow(mask, origin='lower', cmap='gray')
    ax[2].set_title('Source Mask')

By default, pixels are flagged as sources where the statistic is more
than ``n_sigma`` standard deviations (3 by default) above its typical
background value. The standard deviation and the typical background
value are both estimated from the map of the statistic itself. This
matters for images with noise that is correlated between adjacent
pixels (e.g., resampled or drizzled mosaics), where the statistic is
significantly larger than zero even without sources. The typical
background value is taken from the lowest peak of the distribution of
the statistic, so it can be estimated even in crowded images where most
of the patches contain sources. If nearly all of the patches contain
sources, it cannot be estimated and a warning is raised.

Alternatively, the ``percent`` keyword sets the threshold so that the
given percentage of the pixels with a value of the statistic, those
with the largest values, is flagged. A fixed percentage or a smaller
``patch_size`` can be used when the typical background value cannot be
estimated. The final masked fraction differs somewhat from ``percent``
because of the steps described next (e.g., ``percent=35`` masks 31% of
this image).

Holes in the flagged regions are then filled, and a binary opening
removes small isolated regions. A source dominates the statistic of
every patch that contains its detectable pixels, so the flagged region
around a source extends to approximately half the patch size beyond
where the source is detected. This margin also covers the faint outer
regions of the source that sigma clipping misses. Because the patch is
square, the flagged regions around compact sources are square. The
default number of iterations of the binary opening
(``opening_iterations``) is set from the patch size so that the
opening rounds the corners of these regions without enlarging them and
removes only the faintest, marginal detections.

The ``patch_size`` keyword sets the side length of the patch, which
must be an odd integer larger than 3. Smaller patches give a noisier
statistic (its standard deviation for uncorrelated noise is
approximately ``0.5 / patch_size``) and smaller margins around the
sources. They also leave more patches free of sources in crowded
images. Larger patches give a less noisy statistic, but larger margins
and lower spatial resolution, so nearby sources merge into larger
masked regions.

The statistic responds to how much the pixel values vary within each
patch, not to how bright they are. The faint outer regions of very
extended sources change little across a patch, so they may not be
flagged even where they are well above the noise. For images with such
sources, a larger ``patch_size`` can be used, which also enlarges the
masked regions of all of the other sources, or the Moran's I mask can
be combined with a detection-based source mask, such as the
segmentation-based mask described above.

The mask is intended to be used as the ``mask`` input to
:class:`~photutils.background.Background2D` (see
:ref:`background_2d_masking`). Bad pixels can be excluded from the
statistic with the ``mask`` keyword of
`~photutils.background.make_morans_i_mask`. Bad pixels with extreme
values (e.g., hot pixels) should always be excluded, because such a
pixel dominates the statistic of every patch that contains it, which
hides the sources near it. Nonfinite data values and pixels with a
nonpositive weight are excluded automatically. Excluded pixels are not
flagged as sources unless they are enclosed by source pixels, so the
bad-pixel mask must be combined with the source mask when the
background is estimated. Let's treat column 300 as bad and exclude it::

    >>> from photutils.background import Background2D
    >>> bad_mask = np.zeros(data.shape, dtype=bool)
    >>> bad_mask[:, 300] = True
    >>> source_mask = make_morans_i_mask(data, mask=bad_mask)
    >>> bkg = Background2D(data, (50, 50), mask=source_mask | bad_mask,
    ...                    exclude_percentile=50.0)

Because the source mask covers a large fraction of the image, the
``exclude_percentile`` keyword is increased so that boxes that are
partly covered by the mask are still used (see
:ref:`background_2d_masking`). `~photutils.background.Background2D`
masks nonfinite data values automatically, so they do not need to be
included in the combined mask.

Blank regions that are filled with a constant value instead of NaN
(e.g., the zeros outside of the footprint of a mosaic) should also be
excluded with the ``mask`` keyword. The step at the edges of such a
region is otherwise flagged as a source, and the hole filling then
flags everything that those edges enclose, which can be the whole
image (see :ref:`background_2d_blank_regions`).

The pixels can also be weighted with the ``weights`` keyword (e.g., by
an inverse-variance map). The patch means, the patch variance, and the
neighbor means are then weighted, so noisier pixels count less. This is
useful when the noise varies across the image.


2D Background and Noise Estimation
----------------------------------

If the background or the background noise varies across the image,
then you will generally want to generate a 2D image of the background
and background RMS (or compute these values locally). This can be
accomplished by applying the above techniques to subregions of the
image. A common procedure is to divide the input data into a grid of
nonoverlapping boxes and to use sigma-clipped statistics in each box to
create a low-resolution background image, in which each box gives one
pixel, or mesh. The final background or background RMS image can then
be generated by interpolating the low-resolution image. As with the
scalar estimates above, the sources should be masked first (see
:ref:`background_2d_masking`).

Photutils provides the :class:`~photutils.background.Background2D`
class to estimate the 2D background and background noise in an
astronomical image. :class:`~photutils.background.Background2D`
requires the size of the box (``box_size``) in which to estimate the
background. Selecting the box size requires some care by the user.
The box size should be small enough to follow the background
variations. If the sources are not masked, it should also be larger
than the typical size of the sources, so that no box is dominated by a
source. With a source mask, a box that is mostly covered by sources is
excluded instead, so smaller boxes can be used, as long as each box
keeps enough unmasked pixels for its statistics.

The box size is a tradeoff. Smaller boxes follow a background gradient
more closely and give smaller errors near the edges of the image, where
the background is extrapolated beyond the centers of the outermost
boxes. Those errors increase with the box size and with the steepness
of the gradient. Larger boxes contain more pixels, so their estimates
are less noisy. Where the background changes slowly, as in most images,
larger boxes are therefore the better choice.

If possible, the box size should also be chosen so that the data are
covered by an integer number of boxes in both dimensions. If that is
not the case, the boxes along the top and/or right edges are only
partly filled with data. The missing area counts as masked pixels for
the ``exclude_percentile`` keyword (see :ref:`background_2d_masking`),
so a narrow edge box is excluded and the background there is
extrapolated from the neighboring boxes.

For this example, we will create a test image by adding a strong
background gradient to the image defined above::

    >>> ny, nx = data.shape
    >>> y, x = np.mgrid[:ny, :nx]
    >>> gradient = x * y / 5000.0
    >>> data2 = data + gradient

Let's plot the image:

.. doctest-skip::

    >>> norm = simple_norm(data2, 'sqrt', percent=99.5)
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(data2, norm=norm, origin='lower')

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.visualization import simple_norm
    from photutils.datasets import make_100gaussians_image

    data = make_100gaussians_image()
    ny, nx = data.shape
    y, x = np.mgrid[:ny, :nx]
    gradient = x * y / 5000.0
    data2 = data + gradient
    norm = simple_norm(data2, 'sqrt', percent=99.5)
    fig, ax = plt.subplots()
    ax.imshow(data2, norm=norm, origin='lower')
    ax.set_title('Data with added background gradient')

The sources in the image would bias the background estimate high, so
we first make a source mask with the
`~photutils.background.make_morans_i_mask` function (see
:ref:`morans_i_mask`). It does not require an estimate of the
background, so it can be applied directly to the image with the
gradient::

    >>> from photutils.background import make_morans_i_mask
    >>> mask = make_morans_i_mask(data2)

Let's plot the source mask:

.. doctest-skip::

    >>> fig, ax = plt.subplots()
    >>> ax.imshow(mask, origin='lower', cmap='gray')

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from photutils.background import make_morans_i_mask
    from photutils.datasets import make_100gaussians_image

    data = make_100gaussians_image()
    ny, nx = data.shape
    y, x = np.mgrid[:ny, :nx]
    gradient = x * y / 5000.0
    data2 = data + gradient
    mask = make_morans_i_mask(data2)
    fig, ax = plt.subplots()
    ax.imshow(mask, origin='lower', cmap='gray')
    ax.set_title('Source Mask')

Next, we create a `~photutils.background.Background2D` object using a
box size of 10x10 and a 3x3 median filter, and we input the source mask
with the ``mask`` keyword. We will estimate the background level in each
box as the sigma-clipped median using an instance of
:class:`~photutils.background.MedianBackground` (see
:ref:`background_2d_customizing` for the other estimators). Because the
source mask covers a large fraction of the image, we also increase the
``exclude_percentile`` keyword so that boxes with up to half of their
pixels masked are used (see :ref:`background_2d_masking` below)::

    >>> from astropy.stats import SigmaClip
    >>> from photutils.background import Background2D, MedianBackground
    >>> sigma_clip = SigmaClip(sigma=3.0)
    >>> bkg_estimator = MedianBackground()
    >>> bkg = Background2D(data2, (10, 10), filter_size=(3, 3),
    ...                    sigma_clip=sigma_clip, bkg_estimator=bkg_estimator,
    ...                    mask=mask, exclude_percentile=50.0)

The image has 300x500 pixels, so it is covered by exactly 30x50 boxes
of this size. Such small boxes are used here because the gradient of
this example is unusually steep. The background changes by the noise
level over about 30 pixels. Most images call for much larger boxes.

The 2D background and background RMS images are retrieved using the
``background`` and ``background_rms`` properties, respectively, on the
returned object. The low-resolution versions of these images are stored
in the ``background_mesh`` and ``background_rms_mesh`` properties,
respectively. The global median values of the low-resolution background
and background RMS images can be accessed with the ``background_median``
and ``background_rms_median`` properties, respectively::

    >>> print(bkg.background_median)  # doctest: +FLOAT_CMP
    10.548921536796389
    >>> print(round(bkg.background_rms_median, 3))
    1.965

For comparison, the true background of the example image has a median
of 10.56, and the true noise is 2. The background RMS image can be
used, for example, as the per-pixel noise when setting a source
detection threshold (see :ref:`source_detection`).

Let's plot the background image:

.. doctest-skip::

    >>> fig, ax = plt.subplots()
    >>> ax.imshow(bkg.background, origin='lower')

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.stats import SigmaClip
    from photutils.background import (Background2D, MedianBackground,
                                      make_morans_i_mask)
    from photutils.datasets import make_100gaussians_image

    data = make_100gaussians_image()
    ny, nx = data.shape
    y, x = np.mgrid[:ny, :nx]
    gradient = x * y / 5000.0
    data2 = data + gradient
    mask = make_morans_i_mask(data2)
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data2, (10, 10), filter_size=(3, 3),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator,
                       mask=mask, exclude_percentile=50.0)
    fig, ax = plt.subplots()
    ax.imshow(bkg.background, origin='lower')
    ax.set_title('Estimated Background')

and the background-subtracted image:

.. doctest-skip::

    >>> data2_sub = data2 - bkg.background
    >>> norm = simple_norm(data2_sub, 'sqrt', percent=99.9)
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(data2_sub, norm=norm, origin='lower')

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.stats import SigmaClip
    from astropy.visualization import simple_norm
    from photutils.background import (Background2D, MedianBackground,
                                      make_morans_i_mask)
    from photutils.datasets import make_100gaussians_image

    data = make_100gaussians_image()
    ny, nx = data.shape
    y, x = np.mgrid[:ny, :nx]
    gradient = x * y / 5000.0
    data2 = data + gradient
    mask = make_morans_i_mask(data2)
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data2, (10, 10), filter_size=(3, 3),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator,
                       mask=mask, exclude_percentile=50.0)
    data2_sub = data2 - bkg.background
    norm = simple_norm(data2_sub, 'sqrt', percent=99.9)
    fig, ax = plt.subplots()
    ax.imshow(data2_sub, norm=norm, origin='lower')
    ax.set_title('Background-subtracted Data')

The background is extrapolated beyond the centers of the outermost
boxes, so it is less accurate near the edges of the image. In this
example, the background within 5 pixels of the top and right edges,
where it increases toward the edge, is underestimated by about 0.4
and 0.2, respectively. Within 5 pixels of the bottom and left edges,
where it decreases toward the edge, it is overestimated by about 0.5
and 0.3, respectively. A few boxes farther in, the mean error is about
0.1 or less.


.. _background_2d_masking:

Masking Sources and Bad Pixels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Masks can be input into `~photutils.background.Background2D` with two
keywords, ``mask`` and ``coverage_mask``. The ``coverage_mask`` keyword
is described in :ref:`background_2d_blank_regions`.

The ``mask`` keyword excludes pixels, such as sources and bad pixels,
from the background statistics. The masked pixels still receive
background and background RMS values in the output maps, which are
interpolated from the surrounding boxes. Sigma clipping alone removes
only the bright parts of the sources. Their fainter outer regions remain
in the boxes and bias the background estimate high, especially near the
sources. Because the true background of the example image is known (a
level of 5 plus the gradient), we can measure that bias under the
sources for the estimates made with and without the source mask::

    >>> bkg_nomask = Background2D(data2, (10, 10), filter_size=(3, 3),
    ...                           sigma_clip=sigma_clip,
    ...                           bkg_estimator=bkg_estimator)
    >>> background = 5.0 + gradient
    >>> bias = bkg_nomask.background[mask] - background[mask]
    >>> print(round(np.mean(bias), 3))
    0.421
    >>> bias = bkg.background[mask] - background[mask]
    >>> print(round(np.mean(bias), 3))
    0.026

Without the source mask, the background under the sources is
overestimated by about 0.42, which is about 21% of the background noise.
With the source mask, the bias is about 0.03, which is about 1% of the
background noise. The RMS error of the background over the whole image
also decreases, from about 0.37 to about 0.25.

A box is excluded from the low-resolution background map if more than
``exclude_percentile`` percent of its pixels are masked (10 by
default), including the pixels removed by sigma clipping. The
background in an excluded box is interpolated from the neighboring
boxes. The Moran's I mask covers each source out to approximately half
the patch size beyond where it is detected, so in this example about
half of the 10x10 boxes have more than 10% of their pixels in the
source mask. Setting ``exclude_percentile=50`` keeps the boxes in which
at least half of the pixels remain, which still leaves at least 50
pixels for the statistics of each box. A larger value admits boxes with
fewer pixels, whose estimates are noisier, so ``exclude_percentile``
should not be increased more than needed.

Bad pixels can be included in ``mask`` as well. They should also be
excluded when the source mask is made, and then combined with it, as
shown in :ref:`morans_i_mask`.


.. _background_2d_blank_regions:

Images With Blank Regions
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``coverage_mask`` keyword can be used to mask blank regions
without data coverage (e.g., from a rotated image or an image from a
mosaic). If such regions contain zeros (or another constant fill
value), they bias the background statistics unless they are masked.
NaN values are masked automatically, but the output background and
background RMS maps would still contain interpolated values in the
regions without data. Unlike ``mask``, ``coverage_mask`` is applied to
the output background and background RMS maps. The ``fill_value``
keyword defines the value assigned in the output background and
background RMS maps where the input ``coverage_mask`` is `True`.

Let's create a rotated image that has blank areas and plot it:

.. doctest-skip::

    >>> from astropy.visualization import simple_norm
    >>> from scipy.ndimage import rotate
    >>> data3 = rotate(data2, -45.0)
    >>> norm = simple_norm(data3, 'sqrt', percent=99.5)
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(data3, norm=norm, origin='lower')

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.visualization import simple_norm
    from photutils.datasets import make_100gaussians_image
    from scipy.ndimage import rotate

    data = make_100gaussians_image()
    ny, nx = data.shape
    y, x = np.mgrid[:ny, :nx]
    gradient = x * y / 5000.0
    data2 = data + gradient
    data3 = rotate(data2, -45.0)
    norm = simple_norm(data3, 'sqrt', percent=99.5)
    fig, ax = plt.subplots()
    ax.imshow(data3, norm=norm, origin='lower')
    ax.set_title('Rotated data with blank regions')

Now we create a coverage mask and input it into
`~photutils.background.Background2D` to exclude the regions where we
have no data. For this example, we set ``fill_value`` to 0.0, which is
also the default. For real data, one can usually create a coverage mask
from a weight or noise image.

Because the blank regions in this example contain zeros, the coverage
mask must also be input as the ``mask`` keyword of
`~photutils.background.make_morans_i_mask`, so that the blank regions
are excluded from the statistic. Otherwise, every pixel with data would
be flagged as a source. Blank regions that contain NaN
values are excluded automatically. The excluded pixels are not flagged
as sources, so the source mask and the coverage mask are input
separately into `~photutils.background.Background2D`. As in the example
above, we set ``exclude_percentile`` to 50.0, which also keeps more
boxes around the edges of the rotated image:

.. doctest-skip::

    >>> coverage_mask = (data3 == 0)
    >>> mask3 = make_morans_i_mask(data3, mask=coverage_mask)
    >>> bkg3 = Background2D(data3, (10, 10), filter_size=(3, 3),
    ...                     mask=mask3, coverage_mask=coverage_mask,
    ...                     fill_value=0.0, exclude_percentile=50.0)

Let's plot the background image, in which the regions without
coverage are set to ``fill_value``:

.. doctest-skip::

    >>> norm = simple_norm(bkg3.background, 'sqrt', percent=99.5)
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(bkg3.background, norm=norm, origin='lower')

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.visualization import simple_norm
    from photutils.background import Background2D, make_morans_i_mask
    from photutils.datasets import make_100gaussians_image
    from scipy.ndimage import rotate

    data = make_100gaussians_image()
    ny, nx = data.shape
    y, x = np.mgrid[:ny, :nx]
    gradient = x * y / 5000.0
    data2 = data + gradient
    data3 = rotate(data2, -45.0)
    coverage_mask = (data3 == 0)
    mask3 = make_morans_i_mask(data3, mask=coverage_mask)
    bkg3 = Background2D(data3, (10, 10), filter_size=(3, 3),
                        mask=mask3, coverage_mask=coverage_mask,
                        fill_value=0.0, exclude_percentile=50.0)
    norm = simple_norm(bkg3.background, 'sqrt', percent=99.5)
    fig, ax = plt.subplots()
    ax.imshow(bkg3.background, norm=norm, origin='lower')
    ax.set_title('Estimated Background (Rotated Image)')

Finally, let's subtract the background from the image and plot it:

.. doctest-skip::

    >>> data_sub = data3 - bkg3.background
    >>> norm = simple_norm(data_sub, 'sqrt', percent=99.5)
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(data_sub, norm=norm, origin='lower')

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.visualization import simple_norm
    from photutils.background import Background2D, make_morans_i_mask
    from photutils.datasets import make_100gaussians_image
    from scipy.ndimage import rotate

    data = make_100gaussians_image()
    ny, nx = data.shape
    y, x = np.mgrid[:ny, :nx]
    gradient = x * y / 5000.0
    data2 = data + gradient
    data3 = rotate(data2, -45.0)
    coverage_mask = (data3 == 0)
    mask3 = make_morans_i_mask(data3, mask=coverage_mask)
    bkg3 = Background2D(data3, (10, 10), filter_size=(3, 3),
                        mask=mask3, coverage_mask=coverage_mask,
                        fill_value=0.0, exclude_percentile=50.0)
    data_sub = data3 - bkg3.background
    norm = simple_norm(data_sub, 'sqrt', percent=99.5)
    fig, ax = plt.subplots()
    ax.imshow(data_sub, norm=norm, origin='lower')
    ax.set_title('Background-subtracted Data (Rotated Image)')


Plotting Meshes
^^^^^^^^^^^^^^^

The boxes that were used in generating the 2D background can be
plotted on the input image using the
:meth:`~photutils.background.Background2D.plot_meshes` method. With
``outlines=True``, each box is outlined, and a marker shows its center.
Only the boxes that were used are shown, so the excluded boxes appear
as gaps in the grid. Here we zoom in on the lower-left corner of the
rotated image and outline the source mask in orange. The gaps line up
with the masked sources, where more than half of the pixels of a box
are masked. Boxes along the edge of the blank region are excluded for
the same reason, because the pixels without coverage also count as
masked pixels for ``exclude_percentile``.

.. doctest-skip::

    >>> norm = simple_norm(data3, 'sqrt', percent=99.5)
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(data3, norm=norm, origin='lower')
    >>> ax.contour(mask3, levels=[0.5], colors='orange', linewidths=1)
    >>> bkg3.plot_meshes(ax=ax, outlines=True, marker='.', color='white',
    ...                  alpha=0.5)
    >>> ax.set_xlim(0, 250)
    >>> ax.set_ylim(0, 250)

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.visualization import simple_norm
    from photutils.background import Background2D, make_morans_i_mask
    from photutils.datasets import make_100gaussians_image
    from scipy.ndimage import rotate

    data = make_100gaussians_image()
    ny, nx = data.shape
    y, x = np.mgrid[:ny, :nx]
    gradient = x * y / 5000.0
    data2 = data + gradient
    data3 = rotate(data2, -45.0)
    coverage_mask = (data3 == 0)
    mask3 = make_morans_i_mask(data3, mask=coverage_mask)
    bkg3 = Background2D(data3, (10, 10), filter_size=(3, 3),
                        mask=mask3, coverage_mask=coverage_mask,
                        fill_value=0.0, exclude_percentile=50.0)
    norm = simple_norm(data3, 'sqrt', percent=99.5)
    fig, ax = plt.subplots()
    ax.imshow(data3, norm=norm, origin='lower')
    ax.contour(mask3, levels=[0.5], colors='orange', linewidths=1)
    bkg3.plot_meshes(ax=ax, outlines=True, marker='.', color='white',
                     alpha=0.5)
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 250)
    ax.set_title('Background Meshes (Lower-left Corner)')


.. _background_2d_customizing:

Customizing the Estimate
^^^^^^^^^^^^^^^^^^^^^^^^

The pixel values in each box are first sigma clipped. Sigma clipping is
defined by inputting a :class:`astropy.stats.SigmaClip` object to the
``sigma_clip`` keyword. The default is to perform sigma clipping with
``sigma=3`` and ``maxiters=10``. Sigma clipping can be turned off by
setting ``sigma_clip=None``.

The background level in each box is then calculated from the remaining
pixels using the function or callable object (e.g., class instance)
input via the ``bkg_estimator`` keyword. Photutils provides several
background classes that can be used:

* `~photutils.background.MeanBackground`: the mean
* `~photutils.background.MedianBackground`: the median
* `~photutils.background.ModeEstimatorBackground`: a mode estimator of
  the form ``(median_factor * median) - (mean_factor * mean)``, with
  factors of 3 and 2 by default
* `~photutils.background.MMMBackground`: the DAOPHOT MMM mode estimator,
  ``(3 * median) - (2 * mean)``
* `~photutils.background.SExtractorBackground`: the SourceExtractor
  mode estimator (see below)
* `~photutils.background.BiweightLocationBackground`: the biweight
  location

The default is a `~photutils.background.SExtractorBackground` instance.
For this method, the background in each box is calculated as ``(2.5 *
median) - (1.5 * mean)``. However, if ``abs(mean - median) / std >=
0.3`` then the ``median`` is used instead. If the standard deviation is
zero then the ``mean`` is used.

Likewise, the background RMS level in each box is calculated using the
function or callable object input via the ``bkg_rms_estimator`` keyword.
Photutils provides the following classes for this purpose:

* `~photutils.background.StdBackgroundRMS`: the standard deviation
* `~photutils.background.MADStdBackgroundRMS`: the standard deviation
  estimated from the median absolute deviation
* `~photutils.background.BiweightScaleBackgroundRMS`: the biweight
  scale

The default is a `~photutils.background.StdBackgroundRMS` instance. For
even more flexibility, users may input a custom function or callable
object to the ``bkg_estimator`` and/or ``bkg_rms_estimator`` keywords.

Any sigma clipping that is defined on the estimator instances
themselves (e.g., ``MedianBackground(sigma_clip=SigmaClip(sigma=2.0))``)
is ignored. Use the ``sigma_clip`` keyword of
`~photutils.background.Background2D` instead, which applies the same
clipping to both estimators. For example, to use the MMM background,
the MAD-based background RMS, and a 2.5-sigma clip::

    >>> from photutils.background import MADStdBackgroundRMS, MMMBackground
    >>> bkg_mmm = Background2D(data2, (10, 10), mask=mask,
    ...                        exclude_percentile=50.0,
    ...                        sigma_clip=SigmaClip(sigma=2.5),
    ...                        bkg_estimator=MMMBackground(),
    ...                        bkg_rms_estimator=MADStdBackgroundRMS())

With a good source mask, the choice of estimator matters little for
the bias. For the image with the gradient above, the median, MMM, and
SourceExtractor estimators all give a small bias under the sources
(less than 0.03). Their RMS errors of the background range from about
0.25 for the median to about 0.30 for MMM, because the mode estimators
are noisier than the median.

After the background level has been determined in each of the boxes,
the low-resolution background and background RMS images are median
filtered, with a window size of ``filter_size`` (3x3 by default), to
suppress local underestimates or overestimates (e.g., due to bright
galaxies in a particular box). Setting ``filter_size=1`` turns the
filtering off. The median filter can also be applied only to those
boxes where the background level is above a specified threshold
(``filter_threshold``).

The low-resolution background and background RMS images are resized to
the original data size using cubic spline interpolation via
`scipy.ndimage.zoom`.

.. note::
    The ``interpolator`` keyword and the
    :class:`~photutils.background.BkgZoomInterpolator` and
    :class:`~photutils.background.BkgIDWInterpolator` classes, which
    were used to customize the interpolation, were deprecated in
    version 3.0.


Multithreading
^^^^^^^^^^^^^^

The work done by `~photutils.background.Background2D` can be
multithreaded by setting the ``n_threads`` keyword, which parallelizes
both the box statistics and the interpolation of the low-resolution
meshes to the full-size maps returned by the ``background`` and
``background_rms`` properties. The box statistics are identical to the
single-threaded computation and the full-size maps are identical up to
floating-point rounding. The parallel interpolation requires scipy 1.16
or later. With older versions, only the box statistics are computed in
parallel. If a custom ``bkg_estimator`` or ``bkg_rms_estimator``
callable is input, it must be thread-safe. For example:

.. doctest-skip::

    >>> bkg = Background2D(data2, (10, 10), mask=mask,
    ...                    exclude_percentile=50.0, n_threads=4)

Multithreading is most beneficial for large images. For the small
example image above, the gain is negligible. For a 4096x4096 image
with 64x64 boxes, four threads make both the creation of the object
and the calculation of the background map about three times faster.

Each access to the ``background`` or ``background_rms`` property
recalculates its map, which can take longer than creating the object.
Store the map in a variable if it is needed more than once:

.. doctest-skip::

    >>> bkg_image = bkg.background

When both maps are needed, they can also be computed concurrently,
because the underlying interpolation releases the Python global
interpreter lock (GIL). This roughly halves the time to compute both
maps, also with ``n_threads > 1`` as long as there are enough CPU cores
for both sets of threads (``2 * n_threads``):

.. doctest-skip::

    >>> from concurrent.futures import ThreadPoolExecutor
    >>> with ThreadPoolExecutor(max_workers=2) as executor:
    ...     bkg_future = executor.submit(getattr, bkg, 'background')
    ...     rms_future = executor.submit(getattr, bkg, 'background_rms')
    ...     bkg_image = bkg_future.result()
    ...     rms_image = rms_future.result()

`~photutils.background.Background2D` instances are not modified after
construction, except for cached properties, which are safe to compute
from several threads. A single instance can therefore be shared across
threads, and independent images can be processed concurrently, e.g.,
with a `~concurrent.futures.ThreadPoolExecutor`. The underlying
sigma-clipping, statistics, and interpolation kernels release the GIL,
so the threads can effectively run in parallel. For example, to
estimate the background of each image in a list of 2D arrays called
``images``:

.. doctest-skip::

    >>> from photutils.background import Background2D, make_morans_i_mask
    >>> def estimate_background(data):
    ...     mask = make_morans_i_mask(data)
    ...     return Background2D(data, (25, 25), mask=mask,
    ...                         exclude_percentile=50.0).background
    >>> with ThreadPoolExecutor() as executor:
    ...     backgrounds = list(executor.map(estimate_background, images))

For four 2048x2048 images, processing the images in parallel roughly
doubles the throughput of this example.

.. note::
    `~astropy.stats.SigmaClip` instances store internal state while
    clipping. `~photutils.background.Background2D` gives each of its
    own threads a copy, but a single `~astropy.stats.SigmaClip`
    instance should not be shared by
    `~photutils.background.Background2D` objects that are created
    concurrently in different threads. Likewise, a background or
    background RMS estimator instance with sigma clipping enabled
    should not be called concurrently from multiple threads.


API Reference
-------------

:doc:`../reference/background_api`
