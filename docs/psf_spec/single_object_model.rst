SingleObjectModel
=================

SingleObjectModel is an additional step between the PSF (or PRF) model
and the `~astropy.modeling.fitting.Fitter` instance, to allow for cases
where the images being fit contain sources other than point sources. In
these instances a combined PRF, combining the PSF resulting from
telescope optics, CCD, etc. with that of the intrinsic source as it would
appear with infinite resolution.

Parameters
----------

psf_model : `astropy.modeling.Fittable2DModel` instance
    The model used to fit individual point source objects.
object_type : string
    The extended source type used to determine the innate
    light distribution of the source.

Returns
-------

convolve_psf_model : `astropy.modeling.Fittable2DModel` instance
    The new, combined PRF of the extended source, combining the
    intrinsic light distribution and PSF effects.


Example Usage
-------------

This class simply takes the `~astropy.modeling.Fittable2Dmodel` instance and 
convolves it with the appropriate model describing the, e.g., galaxy light
distribution.::
    from photutils.funcs import SingleObjectModel
    single_object_model = SingleObjectModel()
    new_composite_psf = single_object_model(psf_to_add, star['object_type'])
