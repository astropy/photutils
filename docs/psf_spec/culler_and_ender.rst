CullerAndEnder
==============

CullerAndEnder determines the fit quality of any sources found in PSF fitting
routines that build off `~photutils.psf.BasicPSFPhotometry` to include an 
iterative element, such as `~photutils.psf.IterativelySubtractedPSFPhotometry`. It
removes any sources from the input `~astropy.table.Table` below a 
minimum fit criterion, and subsequently determines whether the loop can be ended 
prematurely, most likely when all found sources are of good quality and no 
additional sources have been found. This method may be subject to API changes in
future versions and should not be considered final.

This block is an extra step at the end of each iterative PSF fitting loop, set up
to consider whether the fitting process is complete, or if any sources can have
their fits improved or be added to the output table. As such it requires three
inputs: ``data``, the `~astropy.table.Table` of sources found in the N-1 previous
iterations of the routine with their corresponding positions, fluxes, etc.;
``psf_model``, the `~astropy.modeling.Fittable2DModel` describing the PSF; and
``new_sources``, a `~astropy.table.Table` list of those sources found in the
residual image once the ``data`` sources are removed (i.e., additional sources
with no previously derived parameters).

The two halves of the block are further split into "cull" and "end": one function
determining whether any sources in ``data`` are sufficiently poorly explained by
``psf_model`` that they should not be considered sources at all (e.g., they are
cosmic ray hits picked up as being above some local background); and a second
function, which considers whether we have reached the end of the finding of new 
sources prematurely (whether finding all sources before the loop counter reaches
``niters`` or simply because there are no new sources if ``niters = None``). These
functions (``cull_data`` and ``end_loop``) return a ``new_data`` table, with culled
sources (those whose goodness-of-fit criteria are sufficiently low to be deemed
non-stellar, e.g.), and ``end_flag``, a boolean for whether the iterations can be
ceased (prematurely or not), respectively.

Each subclass of ``CullerAndEnderBase`` can then provide its own definitions of
``cull_data`` and ``end_loop`` depending on the specific user cases. For example,
``end_loop`` could simply be ``return len(new_sources) == 0``, while ``cull_data``
would likely involve the removal of sources with a chi-squared fit to ``psf_model``
above some threshold, or equivalent.

Parameters
----------

data : `~astropy.table.Table`
    Table containing the sources.
psf_model : `astropy.modeling.Fittable2DModel` instance
    PSF/PRF model to which the data are being fit.
new_sources : `~astropy.table.Table`
    Newly found list of sources to compare with the sources
    in ``data``, when deciding whether to end iteration.

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
and ``flux`` are compared to the expected PSF model (a `~astropy.modeling.Fittable2DModel`
such as `photutils.psf.IntegratedGaussianPRF`) and any fits above some minimum goodness-of-fit
criterion are removed. If len(new_sources) == 0 then end_flag would return True, otherwise
it will be set to False.::

    from photutils.psf import CullerAndEnder
    culler_and_ender = CullerAndEnder()
    good_sources, end_flag = culler_and_ender(found_sources, psf_model, new_sources)