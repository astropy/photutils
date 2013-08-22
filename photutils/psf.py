# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Functions for performing PSF fitting photometry on 2-D arrays."""

import abc

import numpy as np

from astropy.nddata.convolution.core import Kernel2D
from astropy.nddata.convolution.kernels import *
from astropy.nddata.modeling import fitting
from astropy.nddata.modeling.core import Parametric2DModel

from arrayutils import *

class AnalyticalPSF(Parametric2DModel):
    """
    An abstract base class for an analytical PSF.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, psf_model):
        pass

    def fit(self):
        """
        Fit PSF model to the data.
        """
        pass


class DiscretePSFModel(Parametric2DModel):
    """
    """
    param_names = ['amplitude']

    def __init__(self, array):
        self._array = array
        super(DiscretePSFModel, self).__init__(array=array)

    def eval(self, x, y, amplitude):
        """
        Discrete model evaluation
        """
        return amplitude * self._array

class SpitzerPSF():
    pass

class FermiPSF():
    pass


class GaussianPSF(Gauss2DKernel):
    """
    Gaussian PSF model.
    """
    param_names = ['amplitude']
    def __init__(self, width):
        super
    
    def eval(self):
        """
        
        """
        return self._array * amplitude 
    
    


class LorentzianPSF(Lorentzian2DKernel, PSF):
    """
    Lorentzian PSF model.
    """
    pass


class MoffatPSF(Beta2DKernel, PSF):
    """
    Moffat PSF model.
    """
    pass


def psf_photometry(data, positions, psf, mode='simultaneous'):
    """
    Perform PSF photometry on the data.
    """
    if mode == 'simultaneous':
        pass
    elif mode == 'sequential':
        for position in positions:
            pass    
            
    else:
        raise Exception('Invalid photometry mode.')


def create_psf(data, positions, size, mode='mean'):
    """
    Create PSF from data array.
    
    Parameters
    ----------
    data : array
        Data array
    positions : List or array
        List of positions in pixel coordinates
        where to create the PSF from.
    size : int
        Size of the PSF in pixels.
    mode : string
        One of the following modes to combine
        the extracted PSFs:
            * 'mean' (default)
                Take the mean of the extracted PSFs.
            * 'median'
                Take the median of the extracted PSFs.
            * 'min'
                Take the pixelwise minimum of the extracted PSFs.
            * 'max'
                Take the pixelwise maximum of the extracted PSFs.
     
    """
    # Setup data cube for extracted PSFs
    extracted_psfs = np.ndarray(shape=(0, size, size))
    
    # Extract PSFs a given pixel positions
    for position in positions:
        extracted_psfs = np.append(extract_array_2D(data, 
                                    (size, size), position), axis=0)
    
    # Choose combination mode
    if mode == 'mean':
        psf = np.mean(extracted_psfs, axis=0)
    elif mode == 'median':
        psf = np.median(extracted_psfs, axis=0)
    elif mode == 'max':
        psf = np.max(extracted_psfs, axis=0)
    elif mode == 'min':
        psf = np.min(extracted_psfs, axis=0)
    else:
        raise Exception('Invalid mode to combine PSFs.') 
    return DiscretePSFModel(psf)
        
        
        
        
        
    