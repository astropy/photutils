# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from scipy import ndimage
from .objshapes import shape_params
from .utils.scale_img import img_stats


def detect_obj(image, snr_threshold, npixels, filter_fwhm=None, image_mask=None, mask_val=None, sig=3.0, iters=None):
    """
    Detect sources in a 2D image above a specified signal-to-noise ratio
    threshold and return a 2D segmentation image.

    This routine does not yet deblend overlapping sources.

    Parameters
    ----------
    image : array_like
        The 2D array of the image.

    snr_threshold : float
        The signal-to-noise ratio threshold above which to detect
        sources.  The background rms noise level is computed using
        sigma-clipped statistics, which can be controlled via the
        ``sig`` and ``iters`` keywords.

    npixels : float
        The number of connected pixels an object must have above the
        threshold level to be detected.

    filter_fwhm : float, optional
        The FWHM of the circular 2D Gaussian filter that is applied to
        the input image before it is thresholded.  Filtering the image
        will maximize detectability of objects with a FWHM similar to
        ``filter_fwhm``.  Set to `None` (the default) to turn off image
        filtering.

    image_mask : array_like, boolean, optional
        A boolean mask with the same shape as ``image``, where a `True`
        value indicates the corresponding element of ``image`` is
        invalid.  Masked pixels are ignored when computing the image
        statistics.

    mask_val : float, optional
        An image data value (e.g., ``0.0``) that is ignored when
        computing the image statistics.  ``mask_val`` will be ignored if
        ``image_mask`` is input.

    sig : float, optional
        The number of standard deviations to use as the clipping limit.

    iters : float, optional
       The number of iterations to perform clipping, or `None` to clip
       until convergence is achieved (i.e. continue until the last
       iteration clips nothing).

    Returns
    -------
    segment_image:  array_like
        A 2D segmentation image of integers indicating segment labels.
    """

    bkgrd, median, bkgrd_rms = img_stats(image, image_mask=image_mask,
                                         mask_val=mask_val, sig=sig,
                                         iters=iters)
    if filter_fhwm is not None:
        img_smooth = ndimage.gaussian_filter(image, filter_fwhm)

    # threshold the smoothed image
    level = bkgrd + (bkgrd_rms * snr_threshold)
    img_thresh = img_smooth >= level

    struct = ndimage.generate_binary_structure(2, 1)
    objlabels, nobj = ndimage.label(img_thresh, structure=struct)
    objslices = ndimage.find_objects(objlabels)

    # remove objects smaller than npixels size
    for objslice in objslices:
        objlabel = objlabels[objslice]
        obj_npix = len(np.where(objlabel.ravel() != 0)[0])
        if obj_npix < npixels:
            objlabels[objslice] = 0

    # relabel (labeled indices must be consecutive)
    objlabels, nobj = ndimage.label(objlabels, structure=struct)
    return objlabels


def find_peaks(img, min_distance=5., threshold_abs=0.03, threshold_rel=0.0, indices=True):



    from skimage.feature import peak_local_max


    idx = peak_local_max(img, min_distance=min_distance, threshold_abs=threshold_abs, threshold_rel=threshold_rel, indices=True)
    #idx = peak_local_max(img, min_distance=5, threshold_abs=0.03, threshold_rel=0, indices=True)
    #print idx.shape
    return idx

#def outline_segments(img, objlabels):
#    x = np.arange(1000)
#    y = np.arange(1000)
#    X, Y = np.meshgrid(x, y)
#    plt.imshow(findobj.scale_sqrt (img, per=99.), cmap=cm.Greys)
#    plt.contour(X, Y, objlabels>0)





def kron_apers(img, objlabels):
    objslices = ndimage.find_objects(objlabels)
    xcens = []
    ycens = []
    majs = []
    mins = []
    thetas = []
    for objslice in objslices:
        tobj = img[objslice]
        sp = shape_params(tobj)
        xcens.append(sp['xcen'] + objslice[1].start)
        ycens.append(sp['ycen'] + objslice[0].start)
        majs.append(sp['major_axis'])
        mins.append(sp['minor_axis'])
        thetas.append(sp['pa'])
    return xcens, ycens, majs, mins, thetas


