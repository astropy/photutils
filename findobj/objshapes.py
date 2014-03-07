
import findobj
import numpy as np
from astropy.convolution import Kernel2D
from astropy.modeling import models


def centroid_com(data, data_mask=None):
    """
    Calculate the centroid of an array as its center of mass determined
    from image moments.

    Parameters
    ----------
    data : array_like
        The image data.

    data_mask : array_like, boolean, optional
        A boolean mask with the same shape as `data`, where a `True`
        value indicates the corresponding element of data in invalid.

    Returns
    -------
    centroid : tuple
        (x, y) coordinates of the centroid.
    """

    if data_mask is not None:
        if data.shape != data_mask.shape:
            raise ValueError('data and data_mask must have the same shap')
        data[data_mask] = 0.
    m = findobj.moments(data, 1)
    xcen = m[1, 0] / m[0, 0]
    ycen = m[0, 1] / m[0, 0]
    return xcen, ycen


def centroid_2dg(data, fwhm, data_err=None, data_mask=None):
    """
    Calculate the centroid of an array from fitting a symmetrical
    2D Gaussian to the data.

    Parameters
    ----------
    data : array_like
        The image data.

    fwhm : float
        The full width at half maximum of the 2D Gaussian.

    data_err : array_like, optional
        The 1-sigma errors for `data`.

    data_mask : array_like, boolean, optional
        A boolean mask with the same shape as `data`, where a `True`
        value indicates the corresponding element of data in invalid.

    Returns
    -------
    centroid : tuple
        (x, y) coordinates of the centroid.
    """
    size = 5.      # all that is needed for accurate centroiding
    stddev = fwhm / 2. * np.sqrt(2. * np.log(2.))
    gauss = Gaussian2DImg(size, size, x_stddev=stddev, y_stddev=stddev)
    return None


def shape_params(data):
    """
    Calculate the centroid and basic shape parameters for an object.

    Parameters
    ----------
    data : array_like
        The image data.

    linear_eccen : is the distance between the object center and either of
                   its two ellipse foci.
    """

    result = {}
    xcen, ycen = centroid(data)
    m = findobj.moments(data, 1)
    mu = findobj.moments_central(data, xcen, ycen, 2) / m[0, 0]
    result['xcen'] = xcen
    result['ycen'] = ycen
    musum = mu[2, 0] + mu[0, 2]
    mudiff = mu[2, 0] - mu[0, 2]
    pa = 0.5 * np.arctan2(2.0*mu[1, 1], mudiff) * (180.0 / np.pi)
    if pa < 0.0:
        pa += 180.0
    result['pa'] = pa
    covar = np.array([[mu[2,0], mu[1,1]], [mu[1,1], mu[0,2]]])
    result['covar'] = covar
    eigvals, eigvecs = np.linalg.eigh(covar)
    majsq = np.max(eigvals)
    minsq = np.min(eigvals)
    result['major_axis'] = np.sqrt(majsq)
    result['minor_axis'] = np.sqrt(minsq)
    #tmp = np.sqrt(4.0*mu[1,1]**2 + mudiff**2)
    #majsq = 0.5 * (musum + tmp)
    #minsq = 0.5 * (musum - tmp)
    #result['major_axis2'] = np.sqrt(majsq)
    #result['minor_axis2'] = np.sqrt(minsq)
    result['eccen'] = np.sqrt(1.0 - (minsq / majsq))
    result['linear_eccen'] = np.sqrt(majsq - minsq)
    return result


class Gaussian2DImg(Kernel2D):
    _separable = True
    _is_bool = False

    def __init__(self, xsize, ysize, x_stddev=None, y_stddev=None,
                 theta=0.0, cov_matrix=None, **kwargs):
        #amplitude = 1. / (2 * np.pi * width ** 2)
        amplitude = 1.0
        self._model = models.Gaussian2D(amplitude, 0, 0, x_stddev=x_stddev,
                                        y_stddev=y_stddev, theta=theta,
                                        cov_matrix=cov_matrix)

        #self._default_size = _round_up_to_odd_integer(8 * width)
        self._default_size = 101
        super(Gaussian2DImg, self).__init__(**kwargs)
        self._truncation = np.abs(1. - 1 / self._normalization)


