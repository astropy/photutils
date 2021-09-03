BackgroundEstimator
===================

EJT: Existing code documented at
https://photutils.readthedocs.io/en/stable/api/photutils.background.Back
groundBase.html - while the ``__call__`` function has no docstring, the
``calc_background`` function is the actual block API. I'm providing
this as an *example* block because it is heavily used in other parts of
`photutils` and therefore probably should not be changed much unless
absolutely necessary.

A single sentence summarizing this block.

A longer description. Can be multiple paragraphs. You can link to other
things like `photutils.background`.


Parameters
----------

data : array_like or `~numpy.ma.MaskedArray`
    The array for which to calculate the background value.

axis : int or `None`, optional
    The array axis along which the background is calculated.  If
    `None`, then the entire array is used.

Returns
-------

result : float or `~numpy.ma.MaskedArray`
    The calculated background value.  If ``axis`` is `None` then
    a scalar will be returned, otherwise a
    `~numpy.ma.MaskedArray` will be returned.


Methods
-------

This block requires no methods beyond ``__call__()``.


Example Usage
-------------

A variety of implementations of this block already exist in ``photutils``. A
canononical example is the mode estimation algorithm ``3 * median - 2 * mean``.
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
