# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Models and functions for doing PSF/PRF fitting photometry on image data.
"""

import numpy as np
from astropy.table import Table
from astropy.nddata.utils import add_array, extract_array
import abc

__all__ = ['subtract_psf', 'CullerAndEnder', 'CullerAndEnderBase']


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
class CullerAndEnderBase:
    """
    Return input table, removing any sources which do not meet
    the quality of fit statistic used to assess fits.
     Parameters
    ----------
    data : `~astropy.table.Table`
        Table containing the sources.
    psf_model : `astropy.modeling.Fittable2DModel` instance
        PSF/PRF model to which the data are being fit.
    new_sources : `~astropy.table.Table`
        Newly found list of sources to compare with the sources
        in `data`, when deciding whether to end iteration.

     Returns
    -------
    culled_data : `~astropy.table.Table`
        ``data`` with poor fits removed.
    end_flag : boolean
        Flag indicating whether to end iterative PSF fitting
        before the maximum number of loops.

    """

    def __call__(self, data, psf_model, new_sources):
        new_data = self.cull_data(data, psf_model)
        end_flag = self.end_loop(new_data, data, new_sources)
        return new_data, end_flag

    def cull_data(self, data, psf_model):
        return NotImplementedError('cull_data should be defined in '
                                   'the subclass.')

    def end_loop(self, new_data, data, new_sources):
        return NotImplementedError('end_loop should be defined in '
                                   'the subclass.')


class CullerAndEnder(CullerAndEnderBase):
    """
    Initial CullerAndEnder which simply ignores culling and ending
    to preserve backwards-compatibility with no implementation.
    """

    def cull_data(self, data, psf_model):
        # Make no attempt to cull any data by quality of fit
        # here; could potentially involve something like sharpness,
        # chi-squared statistics, analysis of residual images, etc.
        return data

    def end_loop(self, new_data, data, new_sources):
        # For now, trivially don't ever dictate the loop
        # end earlier than the maximum number of loops.
        # However, it will likely check something along the lines
        # of whether any poor quality stars were just culled, or
        # whether any new sources were just added; in either case
        # the loop should continue.
        return False
