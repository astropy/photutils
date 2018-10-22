PsfModel
========

The model for the PSF -- integrated over discrete pixels (see `discussion 
<https://github.com/astropy/photutils/blob/master/docs/psf.rst#terminology>`_ 
on terminology for more details) -- on which we should evaluate the parameters
of the source. The main input is a 2D image, with the model returning ``flux``,
``x_0``, and ``y_0``, the overall source flux and centroid position. PSF models
in the current ``photutils.psf`` are not explicitly defined as a specific class,
but any 2D model (inputs: x,y, output: flux) can be considered a PSF Model.

This block is primarily an external block, defined as a
`~astropy.modeling.Fittable2DModel`, and should have a well documented subclass.
The requirements for use within the PSF model block are that the
`~astropy.modeling.Fittable2DModel` have parameters ``flux``, ``x_0``, and ``y_0``,
as well as the ``evaluate`` function. When the model is combined with a minimization
routine the best fitting flux and position for the input ``data`` image array
containing the source are returned.

The `~photutils.psf.PRFAdapter` class allows for the wrapping of any arbitrary 
other models to make it compatible with the machinery. Some individual models 
will require additional parameters, such as the ``IntegratedGaussianPRF`` which 
additionally requiring ``sigma`` as a parameter that describes the underlying 
Gaussian PSF.

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
y_0 : float
    The position of the source in the y axis, to sub-pixel resolution.


Methods
-------

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
fit for centroid and overall flux level of a source.::

    from photutils.psf import IntegratedGaussianPRF
    psf_model = IntegratedGaussianPRF(sigma=2)
    model_image = psf_model.evaluate(x, y, flux, x_0, y_0, sigma=2)
