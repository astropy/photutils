# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Functions for performing PSF fitting photometry on 2-D arrays."""

import abc

import numpy as np

from astropy.nddata.convolution.core import Kernel2D
from astropy.nddata.convolution.kernels import *
from astropy.nddata.convolution.utils import *
from astropy.modeling import fitting
from astropy.modeling.core import Parametric2DModel
from astropy.table import Table, Column
from .photometryutils import *


class ParametricPSFModel():
    pass


class AnalyticalPSF(Kernel2D, Parametric2DModel):
    """
    An abstract base class for an analytical PSF.
    """
    __metaclass__ = abc.ABCMeta

    param_names = ['amplitude']
    def __init__(self, psf_model):
        Kernel2D.__init__(self, array=array)
        Parametric2DModel.__init__(self, {'amplitude': 1})
        self.normalize()


    def eval(self, x, y, amplitude):
        """
        Analytic PSF model evaluation.
        
        Parameters
        ----------
        x : float
            x in pixel coordinates.
        y : float 
            y in pixel coordinates.
        amplitude : float
            Model amplitude. 
        """
        return amplitude * self._array[y, x]
        

class DiscretePSF(Kernel2D, Parametric2DModel):
    """
    Base class for discrete PSF models.
    
    """
    param_names = ['amplitude', 'x_0', 'y_0']
    linear = True
    
    def __init__(self, array, amplitude):
        Kernel2D.__init__(self, array=array)
        Parametric2DModel.__init__(self, {'amplitude': 1})
        self.normalize()
        self.fitter = fitting.LinearLSQFitter(self)
        
    def eval(self, x, y, amplitude):
        """
        Discrete PSF model evaluation.
        
        Parameters
        ----------
        x : float
            x in pixel coordinates.
        y : float 
            y in pixel coordinates.
        amplitude : float
            Model amplitude. 
        """
        return amplitude * self._array[y, x]
        
    def fit(self, data):
        """
        Fit PSF to data.
        """
        y, x = np.indices(self.shape)
        self.fitter(x, y, data)
        return self.amplitude



#class CompositePSFModel(Parametric2DModel):
#    """
#    """
#    def __init__(self):
#        pass
#    
#    def eval(self, x, y, *amplitudes):
#        """
#        """
#        for psf, amplitude in zip(psf_list, amplitudes):
#            psf.eval(x, y, amplitude)
    

#class SpitzerPSF(DiscretePSFModel):
#    """
#    Spitzer PSF model.
#    """
#    pass
#
#
#class FermiPSF():
#    """
#    Fermi PSF model.
#    """
#    pass
#
#
class GaussianPSF(AnalyticalPSF):
    """
    Gaussian PSF model.
    """
    def __init__(self, width):
        constraints = {'fixed': {'x_stddev': True,
                                 'y_stddev': True, 
                                 'x_mean': True, 
                                 'y_mean': True,
                                 'theta': True}}
        
        constraints = {'fixed': {'x_stddev': True,
                                 'y_stddev': True, 
                                 'theta': True}}
#        
#    
#
#class LorentzianPSF(Lorentzian2DKernel, PSF):
#    """
#    Lorentzian PSF model.
#    """
#    pass
#
#
#class MoffatPSF(Beta2DKernel, PSF):
#    """
#    Moffat PSF model.
#    """
#    pass


def psf_photometry(data, xc, yc, psf, mode='simultaneous'):
    """
    Perform PSF photometry on the data.
    
    Parameters
    ----------
    data : array
        Data array
    positions : List or array
        List of positions in pixel coordinates
        where to fit the PSF.
    """
    # Check input array type and dimension.
    data = np.asarray(data)
    if np.iscomplexobj(data):
        raise TypeError('Complex type not supported')
    if data.ndim != 2:
        raise ValueError('{0}-d array not supported. '
                         'Only 2-d arrays supported.'.format(data.ndim))
    if xc.size != yc.size:
        raise ValueError('xc and yc must have same length')
     
    # Actual photometry
    if mode == 'simultaneous':
        raise NotImplementedError('Simultaneous mode not implemented')
    elif mode == 'sequential':
        for x, y in zip(xc, yc):
            psf.x_0, psf.y_0 = x, y
            flux = psf.fit(data)  
    else:
        raise Exception('Invalid photometry mode.')
    return result


def create_psf(data, positions, size, fluxes=None, mode='mean'):
    """
    Create PSF from data array.
    
    Parameters
    ----------
    data : array
        Data array
    positions : List or array
        List of positions in pixel coordinates
        where to create the PSF from.
    size : odd int
        Size of the quadratic PSF grid in pixels.
    mode : string
        One of the following modes to combine
        the extracted PSFs:
            * 'mean' (default)
                Take the pixelwise mean of the extracted PSFs.
            * 'median'
                Take the pixelwise median of the extracted PSFs.
     
    Returns
    -------
    psf : DiscretePSF
        Discrete PSF model estimated from data.
    """
    # Check input array type and dimension.
    data = np.asarray(data)
    if np.iscomplexobj(data):
        raise TypeError('Complex type not supported')
    if data.ndim != 2:
        raise ValueError('{0}-d array not supported. '
                         'Only 2-d arrays supported.'.format(data.ndim))
    if size % 2 == 0:
        raise TypeError("Size must be odd.")
        
    # Setup data cube for extracted PSFs
    extracted_psfs = np.ndarray(shape=(0, size, size))
    
    # Extract PSFs at given pixel positions
    for x, y in positions:
        extracted_psf = extract_array_2D(data, (size, size), (x, y))
        
        # Check shape to exclude incomplete boundary psfs
        if extracted_psf.shape == (size, size) and extracted_psf.sum() != 0:
            
            # Replace NaN values by mirrored value, with respect to the psf's center
            psf_nan = np.isnan(extracted_psf)
            if psf_nan.any():
                if psf_nan.sum() > 3 or psf_nan[size / 2, size / 2]:
                    continue
                else:
                    y_nan_coords, x_nan_coords = np.where(psf_nan==True)
                    for y_nan, x_nan in zip(y_nan_coords, x_nan_coords):
                        if not np.isnan(extracted_psf[-y_nan - 1, -x_nan - 1]):
                            extracted_psf[y_nan, x_nan] = extracted_psf[-y_nan - 1, -x_nan - 1]
                        elif not np.isnan(extracted_psf[y_nan, -x_nan]):
                            extracted_psf[y_nan, x_nan] = extracted_psf[y_nan, -x_nan - 1]
                        else:
                            extracted_psf[y_nan, x_nan] = extracted_psf[-y_nan - 1, x_nan]
            
            # Add extracted psf to data cube
            extracted_psf.shape = (1, size, size)
            extracted_psfs = np.append(extracted_psfs, extracted_psf / extracted_psf.sum(), axis=0)
        else:
            continue
    
    # Choose combination mode
    if mode == 'mean':
        psf = np.mean(extracted_psfs, axis=0)
    elif mode == 'median':
        psf = np.median(extracted_psfs, axis=0)
        psf /= psf.sum()
    else:
        raise Exception('Invalid mode to combine PSFs.') 
    return psf
        

def create_prf(data, xc, yc, size, mode='mean', subpixels=5):
    """
    Create PSF from data array.
    
    Parameters
    ----------
    data : array
        Data array
    positions : List or array
        List of positions in pixel coordinates
        where to create the PSF from.
    size : odd int
        Size of the quadratic PSF grid in pixels.
    mode : string
        One of the following modes to combine
        the extracted PSFs:
            * 'mean' (default)
                Take the pixelwise mean of the extracted PSFs.
            * 'median'
                Take the pixelwise median of the extracted PSFs.
            * 'min'
                Take the pixelwise minimum of the extracted PSFs.
            * 'max'
                Take the pixelwise maximum of the extracted PSFs.
     
    Returns
    -------
    psf : DiscretePSF
        Discrete PSF model estimated from data.
    """
    # Check input array type and dimension.
    data = np.asarray(data)
    if np.iscomplexobj(data):
        raise TypeError('Complex type not supported')
    if data.ndim != 2:
        raise ValueError('{0}-d array not supported. '
                         'Only 2-d arrays supported.'.format(data.ndim))
        
        
    sub_x = sub_y = subpixels
    
    # Setup data array for extracted PRFs
    extracted_prfs = np.ndarray(shape=(sub_y, sub_x, 0, size, size))
    
    # Extract PRFs at given pixel positions
    for x, y in zip(xc, yc):
        extracted_prf = extract_array_2D(data, (size, size), (x, y))
        extracted_prfs = np.append(extracted_psf / extracted_psf.sum(), axis=0)
        
    
    # Choose combination mode
    if mode == 'mean':
        psf = np.mean(extracted_prfs, axis=0)
    elif mode == 'median':
        psf = np.median(extracted_prfs, axis=0)
    elif mode == 'max':
        psf = np.max(extracted_prfs, axis=0)
    elif mode == 'min':
        psf = np.min(extracted_prfs, axis=0)
    else:
        raise Exception('Invalid mode to combine PRFs.') 
    return DiscretePSFModel(psf, 1, 0, 0)
        
        
class PSFFitter(fitting.Fitter):
    """
    PSF fitter.
    """
    def __init__(self):
        """
        """
        pass
    
    def errorfunc(self, fps, *args):
        self.fitpars = fps
        meas = args[-1]
        if self.weights is None:
            return np.ravel(self.model(*args[: -1]) - meas)
        else:
            return np.ravel(self.weights * (self.model(*args[: -1]) - meas))
    
    
    def errorfunc(self, fps, *args):
        """
        Error function.
        """
        self.fitpars = fps
        meas = args[-1]
        return np.ravel(self.model(*args[: -1]) - meas)
    
    def __call__(self):
        """
        """
        pass
    
    
    
    
    
    
        
        
    