
#from astropy.io import fits
#from scale_img import *
#from matplotlib import cm
import numpy as np
from scipy import ndimage
from astropy.stats import sigma_clip
from .objshapes import shape_params
from skimage.feature import peak_local_max


def imgstats(img, sig=3.0, iters=None):
    """
    Perform sigma-clipped statistics on the image.
    """

    idx = (img != 0.0).nonzero()    # remove padding zeros (crude)
    img_clip = sigma_clip(img[idx], sig=sig, iters=iters)
    vals = img_clip.data[~img_clip.mask]    # only good values
    return np.mean(vals), np.median(vals), np.std(vals)


def segment(img, snr_threshold, npixels, filter_fwhm=2.5):
    # background
    # bkgrd, rms =

    bkgrd, img_median, rms = imgstats(img, sig=3.0)

    # filtering (ideally a "matched filter" will maximize detectability)
    # Smooth with a 2.5 pixel gaussian
    img_smooth = ndimage.gaussian_filter(img, filter_fwhm)

    # threshold the smoothed image
    nsigma = snr_threshold
    level = bkgrd + nsigma*rms
    #print 'threshold level', level
    img_thresh = img_smooth > level

    # segment the thresholded image
    shape = ndimage.generate_binary_structure(2, 2)
    objlabels, nobj = ndimage.label(img_thresh, structure=shape)
    #print 'nobj', nobj

    # slice the segments
    objslices = ndimage.find_objects(objlabels)

    #objslices_new = []
    #objlables_new = []
    idx = []
    for (i, objslice) in enumerate(objslices):
        # extract the object from the unconvolved image, centered on
        # the brightest pixel in the thresholded segment and with the
        # same size of the kernel
        tseg = objlabels[objslice]
        npix = len(np.where(tseg.ravel() != 0)[0])
        #if npix >= npixels:
        if npix < npixels:
            objlabels[objslice] = 0
            idx.append(i)
    #print 'Removed nobj:', len(idx)
    #return objlabels[idx]
    # relabel (indices must be consecutive for find_objects, etc.)
    objlabels, nobj = ndimage.label(objlabels, structure=shape)
    return objlabels


def find_peaks(img, min_distance=5., threshold_abs=0.03, threshold_rel=0.0, indices=True):
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


