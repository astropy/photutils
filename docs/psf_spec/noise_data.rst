NoiseData
==========

NoiseData handles the determination and use of data uncertainty in the fitting
of PSF models to image arrays, either by calculating the uncertainty of each
image array value or by allowing the passing of a previously determined
uncertainty array along with the array containing the image counts. This method
may be subject to API changes and should not currently be considered the final
API specification.

NoiseData is currently handled as an additional input parameter for
`~photutils.psf.BasicPSFPhotometry`, and by extension more complex
PSF fitting routines, such as 
`~photutils.psf.IterativelySubtractedPSFPhotometry`. It is either a function
that calculates the corresponding uncertainty value of a given data value,
a value such that the uncertainty array is ignored and all data values
effectively given unity weighting, or set to indicate that the uncertainties
for the image array were pre-computed and passed into the routine.

``noise_calc`` should therefore be provided as a ``callable`` function (or `None` in 
which case it is ignored) returning the uncertainty (in standard deviations)
of each pixel, the same shape as the input ``image`` array.  The data and 
uncertainty arrays are then wrapped in an `~astropy.nddata.NDData`
instance. Alternatively, if ``image`` is provided as an 
`~astropy.nddata.NDData` instance with both ``data`` and ``uncertainty``
attributes then ``noise_calc`` is ignored and the provided uncertainty array
in ``image`` is used directly when calling the chosen ``fitter`` in the PSF
fitting routine.

Parameters
----------

image : array-like
    The input image for which the uncertainty of each pixel is to be calculated.

Returns
-------

uncertainty : array-like
    The uncertainties of the input ``image`` values, as determined by the callable
    function ``noise_calc``. Must return an array of the same shape as ``image``.

Example Usage
-------------

The simplest, and perhaps most common, uncertainty used in photometry is that
derived from Poissonian statistics, where the standard deviation is the square
root of the photon counts. We can then simply pass the `~numpy.sqrt` function
as ``noise_calc``::

    import numpy as np
    from astropy.modeling.fitting import LevMarLSQFitter

    from photutils.psf import (DAOGroup, IntegratedGaussianPRF,
                               IterativelySubtractedPSFPhotometry)
    from photutils.background import MMMBackground
    from photutils.detection import IRAFStarFinder

    daogroup = DAOGroup(crit_separation=8)
    mmm_bkg = MMMBackground()
    iraffind = IRAFStarFinder(threshold=2.5*self.mmm_bkg(image), fwhm=4.5)
    fitter = LevMarLSQFitter()
    psf_model = IntegratedGaussianPRF(sigma=2.05)
    psf_model.sigma.fixed = False
    iter_phot_psf = IterativelySubtractedPSFPhotometry(finder=iraffind,
                                                       group_maker=daogroup,
                                                       bkg_estimator=mmm_bkg,
                                                       psf_model=psf_model,
                                                       fitter=fitter,
                                                       fitshape=(11, 11),
                                                       niters=2,
                                                       noise_calc=np.sqrt)

However, if we wished, we could instead derive our own, more complex function
to determine the corresponding uncertainty of a given pixel value. For instance,
if -- for whatever reason -- the uncertainty of our data points was determined
to be the logarithm of the absolute value of the counts, we could write such a
function, remembering that the returned array must match in shape the input
image::

    def new_uncert_func(image):
        return np.log10(np.abs(image) + 5)
    iter_phot_psf = IterativelySubtractedPSFPhotometry(finder=iraffind,
                                                       group_maker=daogroup,
                                                       bkg_estimator=mmm_bkg,
                                                       psf_model=psf_model,
                                                       fitter=fitter,
                                                       fitshape=(11, 11),
                                                       niters=2,
                                                       noise_calc=new_uncert_func)
