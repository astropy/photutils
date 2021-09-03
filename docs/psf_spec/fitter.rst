Fitter
======

The fitter block is unique in that it is a class not implemented as
part of `photutils`. Rather, it is an object that follows the interface
for fitters in the `astropy.modeling` package. Note that implicitly
the fitters for PSF photometry are always fitters appropriate for *2D*
models (i.e., 2 input dimensions for the x and y pixel coordinates and a
"flux" output).
