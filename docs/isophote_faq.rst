.. _isophote-faq:

Isophote Frequently Asked Questions
-----------------------------------

.. _harmonic_ampl:

1. What are the basic equations relating harmonic amplitudes to geometrical parameter updates?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The basic elliptical isophote fitting algorithm, as described in
`Jedrzejewski (1987; MNRAS 226, 747)
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

    {\delta}_{\Theta} = \frac {2 A_{2} (1 - {\epsilon})} {I' a_0 [(1 - {\epsilon}) ^ 2 - 1 ]}

where :math:`\epsilon` is the ellipticity, :math:`\Theta` is the
position angle, :math:`A_i` and :math:`B_i` are the harmonic
coefficients, and :math:`I'` is the derivative of the intensity along
the major axis direction evaluated at a semimajor axis length of
:math:`a_0`.


2. Why use "ellipticity" instead of the canonical ellipse eccentricity?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main reason is that ellipticity, defined as

.. math::

    \epsilon =  1  -  \frac{b}{a}

better relates with the visual "flattening" of an ellipse.  By looking
at a flattened circle it is easy to guess its ellipticity, as say 0.1.
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
when :math:`I'` itself is small.  This usually happens at the fainter,
outer regions of galaxy images.  `Busko (1996; ASPC 101, 139)
<https://ui.adsabs.harvard.edu/abs/1996ASPC..101..139B/abstract>`_
found by numerical experiments that the precision to which a given
ellipse can be fitted is related to the relative error in the local
radial gradient.

Because of the gradient's critical role, the algorithm has a number of
features to allow its estimation even under difficult conditions. The
default gradient computation, the one used by the algorithm when it
first starts to fit a new isophote, is based on the extraction of two
intensity samples:  #1 at the current ellipse position, and #2 at a
similar ellipse with a 10% larger semimajor axis.

If the gradient so estimated is not meaningful, the algorithm extracts
another #2 sample, this time using a 20% larger radius. In this
context, a meaningful gradient means "shallower", but still close to
within a factor 3 from the previous isophote's gradient estimate.

If still no meaningful gradient can be measured, the algorithm uses
the value measured at the last fitted isophote, but decreased (in
absolute value) by a factor 0.8. This factor is roughly what is
expected from semimajor-axis geometrical-sampling steps of 10 - 20%
and a deVaucouleurs law or an exponential disk in its inner region (r
<~ 5 req). When using the last isophote's gradient as estimator for
the current one, the current gradient error cannot be computed and is
set to `None`.

As a last resort, if no previous gradient estimate is available, the
algorithm just guesses the current value by setting it to be (minus)
10% of the mean intensity at sample #1. This case usually happens only
at the first isophote fitted by the algorithm.

The use of approximate gradient estimators may seem in contradiction
with the fact that isophote fitting errors depend on gradient error,
as well as with the fact that the algorithm itself is so sensitive to
the gradient value. The rationale behind the use of approximate
estimators, however, is based on the fact that the gradient value is
used only to compute increments, not the ellipse parameters
themselves. Approximate estimators are useful along the first steps in
the iteration sequence, in particular when local image contamination
(stars, defects, etc.) might make it difficult to find the correct
path towards the solution. However, if the gradient is still not well
determined at convergence, the subsequent error computations, and the
algorithm's behavior from that point on, will take the fact into
account properly. For instance, the 3rd and 4th harmonic amplitude
errors depend on the gradient relative error, and if this is not
computable at the current isophote, the algorithm uses a reasonable
estimate (80% of the value at the last successful isophote) in order
to generate sensible estimates for those harmonic errors.


4. How are the errors estimated?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most parameters computed directly at each isophote have their errors
computed by standard error propagation. Errors in the ellipse geometry
parameters, on the other hand, cannot be estimated in the same way,
since these parameters are not computed directly but result from a
number of updates from a starting guess value. An error analysis based
on numerical experiments (`Busko 1996; ASPC 101, 139
<https://ui.adsabs.harvard.edu/abs/1996ASPC..101..139B/abstract>`_)
showed that the best error estimators for these geometrical parameters
can be found by simply "projecting" the harmonic amplitude errors that
come from the least-squares covariance matrix by the same formulae in
:ref:`Question 1 <harmonic_ampl>` above used to "project" the
associated parameter updates. In other words, errors for the ellipse
center, ellipticity, and position angle are computed by the same
formulae as in :ref:`Question 1 <harmonic_ampl>`, but replacing the
least-squares amplitudes by their errors. This is empirical and
difficult to justify in terms of any theoretical error analysis, but
it produces sensible error estimators in practice.


5. How is the image sampled?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When sampling is done using elliptical sectors (mean or median modes),
the algorithm described in `Jedrzejewski (1987; MNRAS 226, 747)
<https://ui.adsabs.harvard.edu/abs/1987MNRAS.226..747J/abstract>`_
uses an elaborate, high-precision scheme to take into account partial
pixels that lie along elliptical sector boundaries. In the current
implementation of the `~photutils.isophote.Ellipse` algorithm, this
method was not implemented.  Instead, pixels at sector boundaries are
either fully included or discarded, depending on the precise position
of their centers in relation to the elliptical geometric locus
corresponding to the current ellipse. This design decision is based on
two arguments: (i) it would be difficult to include partial pixels in
median computation, and (ii) speed.

Even when the chosen integration mode is not bilinear, the sampling
algorithm resorts to it in case the number of sampled pixels inside
any given sector is less than 5. It was found that bilinear mode gives
smoother samples in those cases.

Tests performed with artificial images showed that cosmic rays and
defective pixels can be very effectively removed from the fit by a
combination of median sampling and sigma-clipping.


6. How reliable are the fluxes computed by the `~photutils.isophote.Ellipse` algorithm?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The integrated fluxes and areas computed by
`~photutils.isophote.Ellipse` were checked against results produced by
the IRAF ``noao.digiphot.apphot`` tasks ``phot`` and ``polyphot``,
using artificial images. Quantities computed by
`~photutils.isophote.Ellipse` match the reference ones within < 0.1%
in all tested cases.


7. How does the object centerer work?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `~photutils.isophote.EllipseGeometry` class has a
:meth:`~photutils.isophote.EllipseGeometry.find_center` method that
runs an "object locator" around the input object coordinates.

This routine performs a scan over a 10x10 pixel window centered on the
input object coordinates. At each scan position, it extracts two
concentric, roughly circular samples with radii 4 and 8 pixels. It
then computes a signal-to-noise-like criterion using the intensity
averages and standard deviations at each annulus:

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
locator becomes stricter, in the sense that it will not detect faint
objects. To turn off the object locator, set the threshold to a value
>> 1 in `~photutils.isophote.Ellipse`. This will prevent it from
modifying whatever values for the center coordinates were given to the
`~photutils.isophote.Ellipse` algorithm.
