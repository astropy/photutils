BackgroundEstimator
===================

Existing code documented at
https://photutils.readthedocs.io/en/stable/api/photutils.background.BackgroundBase.html
-- while the ``__call__`` function has no docstring, the ``calc_background``
function is the actual block API. This function is used significantly through `photutils`
and is heavily used within the PSF fitting process, so the documentation is summarized
again here.

Routine to estimate the background level of images and provide background subtraction. 
Can either be applied across an entire image or applied in a two-dimensional grid, at
which point background estimation is applied at each grid location locally.

Base class from which all background estimators are defined. Further parameters may be
specified by subclass estimators, such as the ``ModeEstimatorBackground`` in ``photutils``
which takes ``median_factor`` and ``mean_factor`` as additional variables.

Parameters
----------

sigma_clip : `~astropy.stats.SigmaClip` object or None
    The object which defines the level of sigma clipping applied to the data. If `None`
    then no clipping is performed. Passed when creating the class in ``__init__``.

data : array_like or `~numpy.ma.MaskedArray`
    The array for which to calculate the background value. Passed to the ``__call__`` 
    function.

axis : int or `None`, optional
    The array axis along which the background is calculated.  If
    `None`, then the entire array is used. Passed to the ``__call__`` function.

Returns
-------

result : float or `~numpy.ma.MaskedArray`
    The calculated background value.  If ``axis`` is `None` then
    a scalar will be returned, otherwise a
    `~numpy.ma.MaskedArray` will be returned.


Methods
-------

This block requires no methods beyond first initialisation, and ``__call__()``.


Example Usage
-------------

A variety of implementations of this block already exist in ``photutils``. A
canonical example is the mode estimation algorithm ``3 * median - 2 * mean``.
This can be done on an array called  ``image_data`` by using the block like so::

    from photutils.background import ModeEstimatorBackground
    bkg_estimator = ModeEstimatorBackground()
    bkg_value = bkg_estimator(image_data)

The median/mean parameter values can be adjusted as keyword arguments to the
estimator object if desired::

    tweaked_bkg_estimator = ModeEstimatorBackground(median_factor=3.2, mean_factor=1.8)
    new_bkg_value = tweaked_bkg_estimator(image_data)


The estimator will also accept a sigma clipping object that automatically does
sigma clipping before the background is subtracted, like so::

    from astropy.stats import SigmaClip
    clipped_bkg_estimator = ModeEstimatorBackground(sigma_clip=SigmaClip(sigma=3.))
    clipped_bkg_value = clipped_bkg_estimator(image_data)
