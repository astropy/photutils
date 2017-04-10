Elliptical isophote analysis (`photutils.isophote`)
===================================================

Introduction
------------
The `isophote` package replaces the analysis/isophote package formerly found in
the STSDAS software.

The core of the package is the `ellipse` analysis algorithm. It is designed to
fit elliptical isophotes to galaxy images.

The image is measured using an iterative method described in [1]_. Each isophote
is fitted at a pre-defined, fixed semi-major axis length. The algorithm starts from
a first guess ellipse. The image is sampled along that elliptical path, producing a
1-dimensional function that describes the dependency of the intensity (pixel value)
with the polar angle. The harmonic content of this function is decomposed by
least-squares fitting to an harmonic function that includes first and second harmonics.

Each one of the harmonic amplitudes that result from this fit is related to a specific
ellipse geometric parameter, in the sense that it conveys information regarding how
much the current parameter value deviates from the "true" one. At each iteration,
the largest amplitude among the fitted values is selected and used to compute the
corresponding increment in the associated ellipse parameter. That parameter is updated,
and the image is resampled. The process is repeated until certain criteria are met.

See the documentation for the Ellipse class for details.

Refer to the examples in the notebooks for how to start using the package.

Refer to the API documentation below for the detailed description of each class,
method, and parameter.

You can also look for the test code in directory photutils/isophote/tests/ for
several examples on how the API is used in practice.


Reference/API
-------------

.. automodapi:: photutils.isophote.ellipse
.. automodapi:: photutils.isophote.ellipse.ellipse
.. automodapi:: photutils.isophote.ellipse.isophote
.. automodapi:: photutils.isophote.ellipse.sample
.. automodapi:: photutils.isophote.ellipse.fitter
.. automodapi:: photutils.isophote.ellipse.geometry
.. automodapi:: photutils.isophote.ellipse.integrator
.. automodapi:: photutils.isophote.ellipse.harmonics
.. automodapi:: photutils.isophote.ellipse.centerer
.. automodapi:: photutils.isophote.model


Frequently Asked Questions
--------------------------


**1 - What are the basic equations relating harmonic amplitudes to geometrical parameter updates?**

The basic elliptical isophote fitting algorithm, as described in reference [1]_, computes
corrections for the current ellipse's geometrical parameters by essentially "projecting"
the fitted harmonic amplitudes onto the image plane:

.. math::

    {\delta}_{X0} = \frac {-B_{1}} {I'}

.. math::

    {\delta}_{Y0} = \frac {-A_{1} (1 - {\epsilon})} {I'}

.. math::

    {\delta}_{\epsilon} = \frac {-2 B_{2} (1 - {\epsilon})} {I' a}

.. math::

    {\delta}_{\Theta} = \frac {2 A_{2} (1 - {\epsilon})} {I' a [(1 - {\epsilon}) ^ 2 - 1 ]}


**2 - Why use "ellipticity" instead of the canonical ellipse eccentricity?**

The main reason is that ellipticity, defined as

.. math::

      {\epsilon} =  1  -  \frac{b}{a}

better relates with the visual "flattening" of an ellipse. It is easy, by looking to a
flattened circle, to guess its ellipticity as, say, 0.1. The same ellipse has, however,
an eccentricity of 0.44, which is not obvious from its visual aspect. The quantities
relate as

.. math::

      Ecc  =  sqrt [ 1 -  (1 - {\epsilon})^2 ]


**3 - How the radial gradient is estimated?**

The radial intensity gradient is the most critical quantity computed
by the fitting algorithm. As can be seen from the above formulae, small
I' values lead to large values for the correction terms. Thus, I' errors
may lead to large fluctuations in these terms, when I' itself is small.
This happens usually at the fainter, outer regions of galaxy images.
It was found by numerical experiments [2]_ that the precision to which a
given ellipse can be fitted is related to the relative error in the local
radial gradient.

Because of the gradient's critical role, the algorithm has a number of
features to allow its estimation even under difficult conditions. The default
gradient computation, the one used at first by the algorithm when it starts to
fit a new isophote, is based on the extraction of two intensity samples: #1 at
the current ellipse position, and #2 at a similar ellipse with a 10% larger
semi-major axis.

If the gradient so estimated is not meaningful, the algorithm extracts another
#2 sample, this time using a 20% larger radius. In this context, meaningful
gradient means "shallower", but still close to within a factor 3 from the
previous isophote's gradient estimate.

If still no meaningful gradient can be measured, the algorithm uses the value
measured at the last fitted isophote, but decreased (in absolute value) by a
factor 0.8. This factor is roughly what is expected from semi-major axis
geometrical sampling steps of 10 - 20 % and a deVaucouleurs law or an
exponential disk in its inner region (r <~ 5 req). When using the last
isophote's gradient as estimator for the current one, the current gradient
error cannot be computed and is set to None.

As a last resort, if no previous gradient estimate is available, the
algorithm just guesses the current value by setting it to be (minus) 10%
of the mean intensity at sample #1. This case usually happens only at
the first isophote fitted by the algorithm.

The use of approximate gradient estimators may seem in contradiction with
the fact that isophote fitting errors depend on gradient error, as well as
with the fact that the algorithm itself is so sensitive to the gradient
value. The rationale behind the use of approximate estimators, however, is
based on the fact that the gradient value is used only to compute increments,
not the ellipse parameters themselves. Approximate estimators are useful
along the first steps in the iteration sequence, in particular when local
image contamination (stars, defects, etc.) might make it difficult to find
the correct path towards the solution. At convergency, however, if the
gradient is still not well determined, the subsequent error computations,
and the algorithm's behavior from that point on, will take the fact into account
properly. For instance, the 3rd and 4th harmonic amplitude errors depend
on the gradient relative error, and if this is not computable at the
current isophote, the algorithm uses a reasonable estimate (80% of the value at
the last successful isophote) in order to generate sensible estimates for
those harmonic errors.


**4 - How are errors estimated?**

Most parameters computed directly at each isophote have their errors computed
by standard error propagation. Errors in the ellipse geometry parameters, on
the other hand, cannot be estimated in the same way, since these parameters
are not computed directly but result from a number of updates from a starting
guess value. An error analysis based on numerical experiments [2]_ showed that
the best error estimators for these geometrical parameters can be found by
simply "projecting" the harmonic amplitude errors that come from the least-squares
covariance matrix by the same formulae in **Question 1** above used to "project"
the associated parameter updates. In other words, errors for ellipse center,
ellipticity and position angle are computed by the same formulae as in
**Question 1**, but replacing the least-squares amplitudes by their errors. This
is empirical and difficult to justify in terms of any theoretical error analysis,
but showed in practice to produce sensible error estimators.


**5 - How is the image sampled?**

When sampling is done using elliptical sectors (mean or median modes), the
algorithm described in [1]_ uses an elaborate, high-precision scheme to take into
account partial pixels that lie along elliptical sector boundaries. In the
current implementation of the `ellipse` algorithm, this method was not implemented.
Instead, pixels at sector boundaries are either fully included or discarded, depending
on the precise position of their centers in relation to the elliptical geometric locus
corresponding to the current ellipse. This design decision is based on two arguments:
(i) it would be difficult to include partial pixels in median computation, and (ii)
speed.

Even when the chosen integration mode is not bi-linear, the sampling algorithm resorts
to it in case the number of sampled pixels inside any given sector is less than 5. It
was found that bi-linear mode gives smoother samples in those cases.

Tests performed with artificial images showed that cosmic rays and defective pixels can
be very effectively removed from the fit by a combination of median sampling and
sigma-clipping.


**6 - How reliable are the fluxes computed by the `ellipse` algorithm?**

The integrated fluxes and areas computed by `ellipse` where checked against results
produced by the `noao.digiphot.apphot` tasks `phot` and `polyphot`, using artificial
images. Quantities computed by `ellipse` match the reference ones within < 0.1 % in
all tested cases.


**7 - How does the object locator works?**

Before starting the main fitting loop, the algorithm runs an "object locator" routine
around the specified or assumed object coordinates, to check if minimal conditions for
starting a reasonable fit are met. This routine performs a scan over a 10 X 10 pixel
window centered on the input object coordinates. At each scan position, it extracts
two concentric, roughly circular samples with radii 4 and 8 pixels. It computes a
signal-to-noise-like criterion using the intensity averages and standard deviations
at each annulus

.. math::

    c = \frac{f_{1} - f_{2}}{{\sqrt{\sigma_{1}^{2} + \sigma_{2}^{2}}}}


and locates the pixel inside the scanned window where this criterion is a maximum. If the
criterion so computed exceeds a given threshold, it assumes that a suitable object was
detected at that position.

The default threshold value is set to 0.1. This value, and the annuli and window sizes
currently used, were found by trial and error using a number of both artificial and real galaxy
images. It was found that very flattened galaxy images (ellipticity ~ 0.7) cannot be detected
by such a simple algorithm. By increasing the threshold value the object locator becomes more
strict, in the sense that it will not detect faint objects. To turn the object locator, set
the threshold to a value >> 1. This will prevent it from modifying whatever values for the
center coordinates were given to the `ellipse` algorithm.


References
----------

.. [1] JEDRZEJEWSKI, R., 1987, Mon. Not. R. Astr. Soc., 226, 747.

.. [2] BUSKO, I., 1996, Proceedings of the Fifth Astronomical Data Analysis Software and Systems
   Conference, Tucson, PASP Conference Series v.101, ed. G.H. Jacoby and J. Barnes, p.139-142.


