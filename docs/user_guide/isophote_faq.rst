.. _isophote-faq:

Isophote Frequently Asked Questions
-----------------------------------

.. _harmonic_ampl:

1. What are the basic equations relating harmonic amplitudes to geometrical parameter updates?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The basic elliptical isophote fitting algorithm, as described in
`Jedrzejewski (1987, MNRAS 226, 747)
<https://ui.adsabs.harvard.edu/abs/1987MNRAS.226..747J/abstract>`_,
computes corrections for the current ellipse's geometrical parameters
by essentially "projecting" the fitted harmonic amplitudes onto the
image plane:

.. math::

    {\delta}_{X0} = \frac {-B_{1}} {I'}

.. math::

    {\delta}_{Y0} = \frac {-A_{1} (1 - {\epsilon})} {I'}

.. math::

    {\delta}_{\epsilon} = \frac {-2 B_{2} (1 - {\epsilon})} {I' a_0}

.. math::

    {\delta}_{\Theta} = \frac {2 A_{2} (1 - {\epsilon})} {I' a_0 [(1 - {\epsilon}) ^ 2 - 1]}

where :math:`{\delta}_{X0}` and :math:`{\delta}_{Y0}` are the center
corrections along the major and minor axes of the ellipse,
respectively, :math:`\epsilon` is the ellipticity, :math:`\Theta` is
the position angle, :math:`A_i` and :math:`B_i` are the coefficients
of :math:`\sin(iE)` and :math:`\cos(iE)` in the harmonic fit to the
sampled intensities as a function of the angle :math:`E` from the
major axis, and :math:`I'` is the derivative of the intensity along
the major axis direction evaluated at a semimajor axis length of
:math:`a_0`. The center corrections are then rotated by the position
angle to give the corrections along the image x and y axes.


2. Why use "ellipticity" instead of the canonical ellipse eccentricity?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main reason is that ellipticity, defined as

.. math::

    \epsilon =  1  -  \frac{b}{a}

relates better to the visual flattening of an ellipse. By looking
at a flattened circle it is easy to guess its ellipticity as, say, 0.1.
The same ellipse has an eccentricity of 0.44, which is not obvious
from visual inspection. The quantities relate as

.. math::

    Ecc = \sqrt{1 - (1 - {\epsilon})^2}


3. How is the radial gradient estimated?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The radial intensity gradient is the most critical quantity computed
by the fitting algorithm. As can be seen from the above formulae,
small :math:`I'` values lead to large values for the correction terms.
Thus, :math:`I'` errors may lead to large fluctuations in these terms,
when :math:`I'` itself is small. This usually happens at the fainter,
outer regions of galaxy images. `Busko (1996, ASPC 101, 139)
<https://ui.adsabs.harvard.edu/abs/1996ASPC..101..139B/abstract>`_
found by numerical experiments that the precision to which a given
ellipse can be fitted is related to the relative error in the local
radial gradient.

Because of the gradient's critical role, the algorithm has a number of
features to allow its estimation even under difficult conditions. The
default gradient computation, the one used by the algorithm when it
first starts to fit a new isophote, is based on the extraction of two
intensity samples: #1 at the current ellipse position, and #2 at a
similar ellipse with a 10% larger semimajor axis.

If the gradient so estimated is not meaningful, the algorithm extracts
another #2 sample, this time using a 20% larger semimajor axis. In this
context, a meaningful gradient is negative and is either steeper than
one third of a reference gradient or measured with a relative error
smaller than 20%. The reference is the gradient of the previous
iteration of the fit of the same isophote. If no reference is
available, which is the case for the first iteration of each
isophote, the reference is the current gradient plus its error, so
any negative gradient is meaningful.

If still no meaningful gradient can be measured, the algorithm uses
the reference gradient multiplied by a factor of 0.8, which makes it
shallower. The current gradient error then cannot be computed and is
NaN.

The factor of 0.8 is roughly what is expected between two isophotes
from semimajor-axis geometrical-sampling steps of 10% to 20% and a de
Vaucouleurs law or the inner region of an exponential disk (within
about 5 effective radii). These rules follow the STSDAS ``ellipse``
task, with two differences. First, the STSDAS task keeps the reference
from one isophote to the next, starting from a fixed guess. Second,
the STSDAS task never accepts a gradient that is shallower than one
third of the reference. Here such a gradient is accepted if its
relative error is smaller than 20%. Otherwise, a single gradient that
is much steeper than the following ones would replace them for
several iterations.

The use of approximate gradient estimators may seem in contradiction
with the fact that isophote fitting errors depend on gradient error,
as well as with the fact that the algorithm itself is so sensitive to
the gradient value. The rationale behind the use of approximate
estimators, however, is based on the fact that the gradient value is
used only to compute increments, not the ellipse parameters
themselves. Approximate estimators are useful along the first steps in
the iteration sequence, in particular when local image contamination
(stars, defects, etc.) might make it difficult to find the correct
path toward the solution. However, if the gradient is still not well
determined at convergence, the subsequent error computations, and the
algorithm's behavior from that point on, will take the fact into
account properly. For instance, the 3rd and 4th harmonic amplitude
errors depend on the gradient relative error, and if this is not
computable at the current isophote, the algorithm uses a fixed
relative gradient error of 0.8 in order to generate sensible estimates
for those harmonic errors.


4. How are the errors estimated?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most parameters computed directly at each isophote have their errors
computed by standard error propagation. Errors in the ellipse geometry
parameters, on the other hand, cannot be estimated in the same way,
since these parameters are not computed directly but result from a
number of updates from a starting guess value. An error analysis based
on numerical experiments (`Busko 1996, ASPC 101, 139
<https://ui.adsabs.harvard.edu/abs/1996ASPC..101..139B/abstract>`_)
showed that the best error estimators for these geometrical parameters
can be found by simply "projecting" the harmonic amplitude errors that
come from the least-squares covariance matrix by the same formulae in
:ref:`Question 1 <harmonic_ampl>` above used to "project" the
associated parameter updates. In other words, errors for the ellipse
center, ellipticity, and orientation are computed by the same
formulae as in :ref:`Question 1 <harmonic_ampl>`, but replacing the
least-squares amplitudes by their errors. This is empirical and
difficult to justify in terms of any theoretical error analysis, but
it produces sensible error estimators in practice.


5. How is the image sampled?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When sampling is done using elliptical sectors (mean or median modes),
the algorithm described in `Jedrzejewski (1987, MNRAS 226, 747)
<https://ui.adsabs.harvard.edu/abs/1987MNRAS.226..747J/abstract>`_
uses an elaborate, high-precision scheme to take into account partial
pixels that lie along elliptical sector boundaries. In the current
implementation of the `~photutils.isophote.Ellipse` algorithm, this
method was not implemented. Instead, pixels at sector boundaries are
either fully included or discarded, depending on the precise position
of their centers in relation to the elliptical geometric locus
corresponding to the current ellipse. This design decision is based on
two arguments: (i) it would be difficult to include partial pixels in
median computation, and (ii) speed.

In the ``'mean'`` and ``'median'`` modes, a sector with 6 or fewer
valid pixels is sampled with bilinear interpolation instead, and the
whole ellipse is sampled with bilinear interpolation if its sectors
are smaller than one pixel. It was found that bilinear mode gives
smoother samples in those cases.

Tests performed with artificial images showed that cosmic rays and
defective pixels can be very effectively removed from the fit by a
combination of median sampling and sigma clipping.


6. How reliable are the fluxes computed by the `~photutils.isophote.Ellipse` algorithm?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The integrated fluxes and areas computed by
`~photutils.isophote.Ellipse` were checked against results produced by
the IRAF ``noao.digiphot.apphot`` tasks ``phot`` and ``polyphot``,
using artificial images. Quantities computed by
`~photutils.isophote.Ellipse` match the reference ones to within 0.1%
in all tested cases.


7. How does the object centerer work?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `~photutils.isophote.EllipseGeometry` class has a
:meth:`~photutils.isophote.EllipseGeometry.find_center` method that
runs an "object centerer" around the input object coordinates.

This routine performs a scan over an 11x11 pixel window centered on the
input object coordinates. At each scan position, it extracts two
concentric, roughly circular samples, an inner disk with a radius of
about 3.5 pixels and an outer ring with a radius of about 8 pixels. It
then computes a signal-to-noise-like criterion using the intensity
averages and standard deviations in each region:

.. math::

    c = \frac{f_{1} - f_{2}}{{\sqrt{\sigma_{1}^{2} + \sigma_{2}^{2}}}}

and locates the pixel inside the scanned window where this criterion
is a maximum. If the criterion so computed exceeds a given threshold,
it assumes that a suitable object was detected at that position.

The default threshold value is set to 0.1. This value and the annuli
and window sizes currently used were found by trial and error using a
number of both artificial and real galaxy images. It was found that
very flattened galaxy images (ellipticity ~ 0.7) cannot be detected by
such a simple algorithm. By increasing the threshold value the object
centerer becomes stricter, in the sense that it will not detect faint
objects. The criterion has no upper limit, so only a threshold value
of `numpy.inf` guarantees that the center coordinates are not
modified.

The `~photutils.isophote.Ellipse` class does not run the object
centerer. To use it, call the
:meth:`~photutils.isophote.EllipseGeometry.find_center` method of the
`~photutils.isophote.EllipseGeometry` instance before it is input to
`~photutils.isophote.Ellipse`.


8. How can I limit how far the ellipse center moves?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the center of each isophote is fitted freely. In the
outer, low surface-brightness regions of a galaxy, a nearby star or
companion galaxy can pull the fitted center away from the galaxy
nucleus. There are two ways to control this in
:meth:`~photutils.isophote.Ellipse.fit_image`.

Setting ``fix_center=True`` holds the center of every isophote at the
center of the input `~photutils.isophote.EllipseGeometry`.

Setting ``max_center_shift`` to a distance in pixels lets the center
move freely within a circle of that radius around the center of the
input `~photutils.isophote.EllipseGeometry`. If the fit of an isophote
would move the center outside this circle, the center of that isophote
is placed on the circle and held fixed for the rest of its fit, while
the ellipticity and orientation continue to be fitted. The next
isophote starts from that position and may move its center back
inside the circle.

The ``wander`` parameter of the STSDAS ``ellipse`` task is different.
It limits the x and y sizes of each individual center correction
during the iterations, not the total distance from the input center.
When a single correction exceeds ``wander``, STSDAS stops iterating
and extracts the following isophotes in the same direction (outward or
inward) without fitting them.


9. Why does the fit fail for images with a low signal-to-noise ratio?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The fitting algorithm corrects the ellipse parameters by dividing the
harmonic amplitudes by the local radial gradient, as described above.
The gradient is measured from the mean intensities of two ellipses
whose semimajor axis lengths differ by only 10%. For a low surface
brightness galaxy, or any image where the noise of a pixel is
comparable to the galaxy intensity, the difference between these two
means can be smaller than its noise. The corrections are then
dominated by noise. The ellipse wanders or diverges, and the fit
stops after a few isophotes. If the first isophote cannot be
fit, :meth:`~photutils.isophote.Ellipse.fit_image` emits a "No
meaningful fit was possible" warning and returns an empty list.

No setting of the fit parameters solves this reliably. In particular,
the ``max_gradient_rel_err`` parameter only controls where the outward
growth stops, so it does not help the first isophote. Multiplying the
image by a constant does not help either, because the algorithm does
not depend on the intensity scale.

Whether the first isophote can be fit depends on the noise along one
ellipse. A fit that fails therefore often succeeds from a slightly
different starting semimajor axis length (the ``sma_start`` parameter),
e.g., one that is 10% larger or smaller.

The opposite failure is also possible.
:meth:`~photutils.isophote.Ellipse.fit_image` is not a detection
test. If there is no detectable galaxy at the starting position, the
fit can follow the noise and return isophotes without a warning.
Check that the galaxy is detected before fitting it, e.g., with the
tools described in :ref:`image_segmentation`.

The most effective solution is to increase the signal-to-noise ratio
of the pixels by smoothing the image before fitting it, e.g., with a
Gaussian kernel with a standard deviation of 2 to 3 pixels:

.. doctest-skip::

    >>> from astropy.convolution import Gaussian2DKernel, convolve
    >>> from photutils.isophote import Ellipse
    >>> smoothed = convolve(data, Gaussian2DKernel(2.0))
    >>> ellipse = Ellipse(smoothed, geometry=geometry)
    >>> isolist = ellipse.fit_image()

Fits of the smoothed image are much more likely to succeed, and they
reach larger semimajor axis lengths.

Smoothing lowers the central intensities and makes the isophotes
rounder at semimajor axis lengths comparable to the kernel size. To
measure intensities that are not affected by the smoothing, use the
:meth:`~photutils.isophote.Ellipse.measure_isophotes` method to
measure the original image along the ellipses that were fit to the
smoothed image (see :ref:`isophote_fixed_ellipses`):

.. doctest-skip::

    >>> isolist_data = Ellipse(data).measure_isophotes(isolist)

If the ellipse geometry of the galaxy is already known, the isophotes
can also be extracted without fitting them. With
``max_iterative_sma=0``, all the isophotes have the geometry of the
input `~photutils.isophote.EllipseGeometry`. The ``max_sma`` parameter
must then be set, because otherwise the outward growth stops where the
gradient error exceeds ``max_gradient_rel_err``:

.. doctest-skip::

    >>> ellipse = Ellipse(data, geometry=geometry)
    >>> isolist = ellipse.fit_image(max_iterative_sma=0.0, max_sma=60.0)

The ``'mean'`` and ``'median'`` integration modes with a larger
``step`` (e.g., 0.3) average more pixels for each sample point. They
can help for moderately noisy images, but they are less reliable than
smoothing the image.
