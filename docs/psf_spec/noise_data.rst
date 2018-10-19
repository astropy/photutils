NoiseData
==========

NoiseData is currently handled as an additional input parameter for
BasicPSFPhotometry, and by extension IterativelySubtractedPSFPhotometry.
``Noise_calc`` should be provided as a callable function (or `None` in 
which case it is ignored) returning the uncertainty (in standard deviations)
of each pixel, the same shape as the input ``image`` array. The data and
uncertainty arrays are then wrapped in an `~astropy.nddata.NDData` 
instance.
