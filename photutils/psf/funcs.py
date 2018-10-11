# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Models and functions for doing PSF/PRF fitting photometry on image data.
"""

import numpy as np
from astropy.table import Table
from astropy.nddata.utils import add_array, extract_array
import abc

__all__ = ['subtract_psf', 'SingleObjectModel', 'SingleObjectModelBase']


def _extract_psf_fitting_names(psf):
    """
    Determine the names of the x coordinate, y coordinate, and flux from
    a model.  Returns (xname, yname, fluxname)
    """

    if hasattr(psf, 'xname'):
        xname = psf.xname
    elif 'x_0' in psf.param_names:
        xname = 'x_0'
    else:
        raise ValueError('Could not determine x coordinate name for '
                         'psf_photometry.')

    if hasattr(psf, 'yname'):
        yname = psf.yname
    elif 'y_0' in psf.param_names:
        yname = 'y_0'
    else:
        raise ValueError('Could not determine y coordinate name for '
                         'psf_photometry.')

    if hasattr(psf, 'fluxname'):
        fluxname = psf.fluxname
    elif 'flux' in psf.param_names:
        fluxname = 'flux'
    else:
        raise ValueError('Could not determine flux name for psf_photometry.')

    return xname, yname, fluxname


def _call_fitter(fitter, psf, x, y, data, weights):
    """
    Not all fitters have to support a weight array. This function
    includes the weight in the fitter call only if really needed.
    """

    if np.all(weights == 1.):
        return fitter(psf, x, y, data)
    else:
        return fitter(psf, x, y, data, weights=weights)


def subtract_psf(data, psf, posflux, subshape=None):
    """
    Subtract PSF/PRFs from an image.

    Parameters
    ----------
    data : `~astropy.nddata.NDData` or array (must be 2D)
        Image data.
    psf : `astropy.modeling.Fittable2DModel` instance
        PSF/PRF model to be substracted from the data.
    posflux : Array-like of shape (3, N) or `~astropy.table.Table`
        Positions and fluxes for the objects to subtract.  If an array,
        it is interpreted as ``(x, y, flux)``  If a table, the columns
        'x_fit', 'y_fit', and 'flux_fit' must be present.
    subshape : length-2 or None
        The shape of the region around the center of the location to
        subtract the PSF from.  If None, subtract from the whole image.

    Returns
    -------
    subdata : same shape and type as ``data``
        The image with the PSF subtracted
    """

    if data.ndim != 2:
        raise ValueError('{0}-d array not supported. Only 2-d arrays can be '
                         'passed to subtract_psf.'.format(data.ndim))

    #  translate array input into table
    if hasattr(posflux, 'colnames'):
        if 'x_fit' not in posflux.colnames:
            raise ValueError('Input table does not have x_fit')
        if 'y_fit' not in posflux.colnames:
            raise ValueError('Input table does not have y_fit')
        if 'flux_fit' not in posflux.colnames:
            raise ValueError('Input table does not have flux_fit')
    else:
        posflux = Table(names=['x_fit', 'y_fit', 'flux_fit'], data=posflux)

    # Set up contstants across the loop
    psf = psf.copy()
    xname, yname, fluxname = _extract_psf_fitting_names(psf)
    indices = np.indices(data.shape)
    subbeddata = data.copy()

    if subshape is None:
        indicies_reversed = indices[::-1]

        for row in posflux:
            getattr(psf, xname).value = row['x_fit']
            getattr(psf, yname).value = row['y_fit']
            getattr(psf, fluxname).value = row['flux_fit']

            subbeddata -= psf(*indicies_reversed)
    else:
        for row in posflux:
            x_0, y_0 = row['x_fit'], row['y_fit']

            y = extract_array(indices[0], subshape, (y_0, x_0))
            x = extract_array(indices[1], subshape, (y_0, x_0))

            getattr(psf, xname).value = x_0
            getattr(psf, yname).value = y_0
            getattr(psf, fluxname).value = row['flux_fit']

            subbeddata = add_array(subbeddata, -psf(x, y), (y_0, x_0))

    return subbeddata


@abc.abstractmethod
class SingleObjectModelBase:
    """
    Handles the convolution of non-point source objects with the
    telescope PSF (or PRF, for discrete pixels), returning a PRF
    which describes the fraction of light falling in each pixel
    of an extended source.

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
    """
    def __call__(self, psf_model, object_type):
        return self.make_single_object(psf_model, object_type)

    def make_single_object(self, psf_model, object_type):
        raise NotImplementedError('make_single_object must be defined'
                                  'in a subclass.')


class SingleObjectModel(SingleObjectModelBase):
    """
    Simple single object model class which assumes all sources are
    point sources.
    """
    def make_single_object(self, psf_model, object_type):
        return psf_model
