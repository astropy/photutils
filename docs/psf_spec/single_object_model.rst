SingleObjectModel
=================

SingleObjectModel handles the creation of the model image of an
astrophysical object, combining physical on-sky flux distributions with the
effects of the PSF, extending the ``psf_model`` block of the PSF
fitting routine to include non-point sources. This method may be subject
to API changes in future versions and should not be considered final.

This block is an additional step between the PSF (or PRF) 
`~astropy.modeling.Fittable2DModel` and the
`~astropy.modeling.fitting.Fitter` instance, to allow for cases
where the images being fit contain sources other than point sources. In
these instances a combined PRF, combining the PSF resulting from
telescope optics, CCD, etc. with that of the intrinsic source as it would
appear with infinite resolution is required. If the source is a simple
point source, then the block simply returns the given PSF, avoiding the
lengthy computational time of image convolution.

The single object model -- the model determining what type of object
(single star, binary star, galaxy, supernova, etc.) this source is assumed
to be -- is described by an additional column of the ``star`` source list
created during the PSF fitting routine. For a given fit the object is
determined by ``star['object_type']`` which is accepted as the ``string``
argument ``object_type`` by ``SingleObjectModel``. Additionally,
``psf_model``, the `~astropy.modeling.Fittable2DModel` describing the PSF,
accounting for quantized detector pixels, is required. The block then returns
``convolve_psf_model``, the convolution of ``psf_model`` with the flux
distribution intrinsic to the source with given internal parameters
(such as galaxy distance to correct for apparent size, time since
supernova explosion to produce the correct lightcurve, etc.). The class must
be a subclass of ``SingleObjectModelBase``, with the individual class provided
to the PSF fitting routine accepting a specific set of sources to be passed
in ``object_type``, with each object being handled on a case-by-case basis.

This block is also related to ``SceneMaker``, the future extension to the
standard "source grouping" aspect of the PSF fitting process, which additionally
allows for the grouping of several objects, otherwise assumed to be single point
sources, into a composite extended source, and vice versa. It is
``SceneMaker`` which therefore dictates the ``object_type`` of a given source
in the created ``star`` catalogue. ``SceneMaker`` must be set up such that it
will group and merge ``star`` sources into groups such that all object types
determined are accepted by ``SingleObjectModel`` -- the two blocks must therefore
work closely in parallel with one another.

Parameters
----------

psf_model : `~astropy.modeling.Fittable2DModel` instance
    The model describing the PSF/PRF of the image, used to fit individual point
    source objects.
object_type : string
    The extended source type used to determine the innate
    light distribution of the source.

Returns
-------

convolve_psf_model : `~astropy.modeling.Fittable2DModel` instance
    The new, combined PRF of the extended source, combining the
    intrinsic light distribution and PSF effects.


Example Usage
-------------

This class simply takes the `~astropy.modeling.Fittable2DModel` instance and 
convolves it with the appropriate model describing the, e.g., galaxy light
distribution::

    from photutils.psf import SingleObjectModel
    single_object_model = SingleObjectModel()
    new_composite_psf = single_object_model(psf_to_add, star['object_type'])
