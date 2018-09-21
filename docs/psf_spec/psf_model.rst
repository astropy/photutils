PsfModel
========

The PSF models in the current ``photutils.psf`` are not explicitly
defined as a specific class, but any 2D model (inputs: x,y, output: flux) can
be considered a PSF Model.  The `~photutils.psf.PRFAdapter` class is the
clearest application specific example, however, and demonstrates the required
convention for *names* of the PSF model's parameters.  Hopefully that class can
be used *directly* as this block, as it is meant to wrap any arbitrary other
models to make it compatible with the machinery.

The model for the PSF -- integrated over discrete pixels (see `discussion 
<https://github.com/astropy/photutils/blob/master/docs/psf.rst#terminology>'_ 
on terminology for more details) -- on which we should evaluate the parameters
of the source. The main input is a 2D image, with the model being represented
by a ``FittableImageModel``, returning ``flux``, ``x_0``, and ``y_0``, the
overall source flux and centroid position. Individual models will require 
additional parameters, such as the ``IntegratedGaussianPRF`` which additionally
requiring ``sigma`` as a parameter that describes the underlying Gaussian PSF.

Parameters
----------

data : np.ndarray
    Array containing the 2D image representing the source(s) that should be
    evaluated for centroiding and flux.

Returns
-------

flux : float
    The flux of the source; if any normalization has been applied then ``flux``
    will be set to 1.
x_0 : float
    The position of the source in the x axis, to sub-pixel resolution.


Methods
-------

Not all blocks will have these, but if desired some blocks can have methods that
let you do something other than just running the block.  E.g::

    some_block = BlockClassName()
    output = some_block(input1, input2, ...)  # this is what is documented above
    result = some_block.method_name(...)  #this is documented here

evaluate
^^^^^^^^^^^

Function that returns the values of the pixels of a model with ``flux``, ``x_0`` and
``y_0`` as parameters.

Parameters
""""""""""

x : float
    Central values of the pixels in the x axis at which to evaluate the PSF.
y : float
    Central values of the pixels in the y axis at which to evaluate the PSF.
flux : float
    Overall flux level of source, setting the scaling of each pixel.
x_0 : float
    The centroid position of the source in the x axis.
y_0 : float
    The centroid position of the source in the y axis.

Returns
"""""""

The evaluation of the PSF, integrated over discrete pixels, at the centroid position
and with the flux level specified.


Example Usage
-------------
Here we create a discrete Gaussian PSF with standard deviation of 2 pixels, and 
fit for centroid and overall flux level of a source.
::
    from photutils.psf import IntegratedGaussianPRF
    psf_model = IntegratedGaussianPRF(sigma=2)
    model_image = psf_model.evaluate(x, y, flux, x_0, y_0, sigma=2)
