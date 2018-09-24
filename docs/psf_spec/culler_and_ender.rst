CullerAndEnder
==============

CullerAndEnder determines the fit quality of IterativelySubtractedPSFPhotometry 
sources, removing any sources from the input `~astropy.table.Table` below a 
minimum fit criterion, and subsequently determines whether the loop can be ended 
prematurely, most likely when all found sources are of good quality and no 
additional sources have been found. This method may be subject to API changes in
future versions and should not be considered final.

Parameters
----------

data : `~astropy.table.Table`
    Table containing the sources.
psf_model : `astropy.modeling.Fittable2DModel` instance
    PSF/PRF model to which the data are being fit.
new_sources : `~astropy.table.Table`
    Newly found list of sources to compare with the sources
    in `data`, when deciding whether to end iteration.

Returns
-------

culled_data : `~astropy.table.Table`
        ``data`` with poor fits removed.
end_flag : boolean
    Flag indicating whether to end iterative PSF fitting
    before the maximum number of loops.

Example Usage
-------------

In the simplest example, a list of sources ``found_sources`` with parameters ``x_0``, ``y_0``
and ``flux`` are compared to the expected PSF model (a FittableImageModel such as 
`photutils.psf.IntegratedGaussianPRF`) and any fits above some minimum goodness-of-fit
criterion are removed. If len(new_sources) == 0 then end_flag would return True, otherwise
it will be set to False.
::
    from photutils.psf.funcs import CullerAndEnder
    culler_and_ender = CullerAndEnder()
    good_sources, end_flag = culler_and_ender(found_sources, psf_model, new_sources)