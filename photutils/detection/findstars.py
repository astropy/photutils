# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions for detecting stars in an astronomical image."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import warnings
import math
import numpy as np
from astropy.table import Column, Table
from astropy.utils.exceptions import AstropyUserWarning
from astropy.stats import gaussian_fwhm_to_sigma
from .core import _convolve_data, find_peaks


__all__ = ['daofind', 'irafstarfind']


def daofind(data, threshold, fwhm, ratio=1.0, theta=0.0, sigma_radius=1.5,
            sharplo=0.2, sharphi=1.0, roundlo=-1.0, roundhi=1.0, sky=0.0,
            exclude_border=False):
    """
    Detect stars in an image using the DAOFIND algorithm.

    `DAOFIND`_ (`Stetson 1987; PASP 99, 191
    <http://adsabs.harvard.edu/abs/1987PASP...99..191S>`_) searches
    images for local density maxima that have a peak amplitude greater
    than ``threshold`` (approximately; ``threshold`` is applied to a
    convolved image) and have a size and shape similar to the defined 2D
    Gaussian kernel.  The Gaussian kernel is defined by the ``fwhm``,
    ``ratio``, ``theta``, and ``sigma_radius`` input parameters.

    .. _DAOFIND: http://iraf.net/irafhelp.php?val=daofind&help=Help+Page

    ``daofind`` finds the object centroid by fitting the the marginal x
    and y 1D distributions of the Gaussian kernel to the marginal x and
    y distributions of the input (unconvolved) ``data`` image.

    ``daofind`` calculates the object roundness using two methods.  The
    ``roundlo`` and ``roundhi`` bounds are applied to both measures of
    roundness.  The first method (``roundness1``; called ``SROUND`` in
    `DAOFIND`_) is based on the source symmetry and is the ratio of a
    measure of the object's bilateral (2-fold) to four-fold symmetry.
    The second roundness statistic (``roundness2``; called ``GROUND`` in
    `DAOFIND`_) measures the ratio of the difference in the height of
    the best fitting Gaussian function in x minus the best fitting
    Gaussian function in y, divided by the average of the best fitting
    Gaussian functions in x and y.  A circular source will have a zero
    roundness.  An source extended in x or y will have a negative or
    positive roundness, respectively.

    The sharpness statistic measures the ratio of the difference between
    the height of the central pixel and the mean of the surrounding
    non-bad pixels in the convolved image, to the height of the best
    fitting Gaussian function at that point.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    threshold : float
        The absolute image value above which to select sources.

    fwhm : float
        The full-width half-maximum (FWHM) of the major axis of the
        Gaussian kernel in units of pixels.

    ratio : float, optional
        The ratio of the minor to major axis standard deviations of the
        Gaussian kernel.  ``ratio`` must be strictly positive and less
        than or equal to 1.0.  The default is 1.0 (i.e., a circular
        Gaussian kernel).

    theta : float, optional
        The position angle (in degrees) of the major axis of the
        Gaussian kernel measured counter-clockwise from the positive x
        axis.

    sigma_radius : float, optional
        The truncation radius of the Gaussian kernel in units of sigma
        (standard deviation) [``1 sigma = FWHM /
        (2.0*sqrt(2.0*log(2.0)))``].

    sharplo : float, optional
        The lower bound on sharpness for object detection.

    sharphi : float, optional
        The upper bound on sharpness for object detection.

    roundlo : float, optional
        The lower bound on roundess for object detection.

    roundhi : float, optional
        The upper bound on roundess for object detection.

    sky : float, optional
        The background sky level of the image.  Setting ``sky`` affects
        only the output values of the object ``peak``, ``flux``, and
        ``mag`` values.  The default is 0.0, which should be used to
        replicate the results from `DAOFIND`_.

    exclude_border : bool, optional
        Set to `True` to exclude sources found within half the size of
        the convolution kernel from the image borders.  The default is
        `False`, which is the mode used by `DAOFIND`_.

    Returns
    -------
    table : `~astropy.table.Table`

        A table of found objects with the following parameters:

        * ``id``: unique object identification number.
        * ``xcentroid, ycentroid``: object centroid.
        * ``sharpness``: object sharpness.
        * ``roundness1``: object roundness based on symmetry.
        * ``roundness2``: object roundness based on marginal Gaussian
          fits.
        * ``npix``: number of pixels in the Gaussian kernel.
        * ``sky``: the input ``sky`` parameter.
        * ``peak``: the peak, sky-subtracted, pixel value of the object.
        * ``flux``: the object flux calculated as the peak density in
          the convolved image divided by the detection threshold.  This
          derivation matches that of `DAOFIND`_ if ``sky`` is 0.0.
        * ``mag``: the object instrumental magnitude calculated as
          ``-2.5 * log10(flux)``.  The derivation matches that of
          `DAOFIND`_ if ``sky`` is 0.0.

    See Also
    --------
    irafstarfind

    Notes
    -----
    For the convolution step, this routine sets pixels beyond the image
    borders to 0.0.  The equivalent parameters in `DAOFIND`_ are
    ``boundary='constant'`` and ``constant=0.0``.

    References
    ----------

    .. [1] Stetson, P. 1987; PASP 99, 191 (http://adsabs.harvard.edu/abs/1987PASP...99..191S)
    .. [2] http://iraf.net/irafhelp.php?val=daofind&help=Help+Page
    .. [3] http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?daofind
    """

    daofind_kernel = _FindObjKernel(fwhm, ratio, theta, sigma_radius)
    threshold *= daofind_kernel.relerr
    objs = _findobjs(data, threshold, daofind_kernel,
                     exclude_border=exclude_border)
    tbl = _daofind_properties(objs, threshold, daofind_kernel, sky)
    if len(objs) == 0:
        warnings.warn('No sources were found.', AstropyUserWarning)
        return tbl     # empty table
    table_mask = ((tbl['sharpness'] > sharplo) &
                  (tbl['sharpness'] < sharphi) &
                  (tbl['roundness1'] > roundlo) &
                  (tbl['roundness1'] < roundhi) &
                  (tbl['roundness2'] > roundlo) &
                  (tbl['roundness2'] < roundhi))
    tbl = tbl[table_mask]
    idcol = Column(name='id', data=np.arange(len(tbl)) + 1)
    tbl.add_column(idcol, 0)
    if len(tbl) == 0:
        warnings.warn('Sources were found, but none pass the sharpness and '
                      'roundness criteria.', AstropyUserWarning)
    return tbl


def irafstarfind(data, threshold, fwhm, sigma_radius=1.5, minsep_fwhm=2.5,
                 sharplo=0.5, sharphi=2.0, roundlo=0.0, roundhi=0.2,
                 sky=None, exclude_border=False):
    """
    Detect stars in an image using IRAF's "starfind" algorithm.

    `starfind`_ searches images for local density maxima that have a
    peak amplitude greater than ``threshold`` above the local background
    and have a PSF full-width half-maximum similar to the input
    ``fwhm``.  The objects' centroid, roundness (ellipticity), and
    sharpness are calculated using image moments.

    .. _starfind: http://iraf.net/irafhelp.php?val=starfind&help=Help+Page

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    threshold : float
        The absolute image value above which to select sources.

    fwhm : float
        The full-width half-maximum (FWHM) of the 2D circular Gaussian
        kernel in units of pixels.

    minsep_fwhm : float, optional
        The minimum separation for detected objects in units of
        ``fwhm``.

    sigma_radius : float, optional
        The truncation radius of the Gaussian kernel in units of sigma
        (standard deviation) [``1 sigma = FWHM /
        2.0*sqrt(2.0*log(2.0))``].

    sharplo : float, optional
        The lower bound on sharpness for object detection.

    sharphi : float, optional
        The upper bound on sharpness for object detection.

    roundlo : float, optional
        The lower bound on roundess for object detection.

    roundhi : float, optional
        The upper bound on roundess for object detection.

    sky : float, optional
        The background sky level of the image.  Inputing a ``sky`` value
        will override the background sky estimate.  Setting ``sky``
        affects only the output values of the object ``peak``, ``flux``,
        and ``mag`` values.  The default is ``None``, which means the
        sky value will be estimated using the `starfind`_ method.

    exclude_border : bool, optional
        Set to `True` to exclude sources found within half the size of
        the convolution kernel from the image borders.  The default is
        `False`, which is the mode used by `starfind`_.

    Returns
    -------
    table : `~astropy.table.Table`

        A table of found objects with the following parameters:

        * ``id``: unique object identification number.
        * ``xcentroid, ycentroid``: object centroid (zero-based origin).
        * ``fwhm``: estimate of object FWHM from image moments.
        * ``sharpness``: object sharpness calculated from image moments.
        * ``roundness``: object ellipticity calculated from image moments.
        * ``pa``:  object position angle in degrees from the positive x
          axis calculated from image moments.
        * ``npix``: number of pixels in the object used to calculate
          ``flux``.
        * ``sky``: the derived background sky value, unless ``sky`` was
          input.  If ``sky`` was input, then that value overrides the
          background sky estimation.
        * ``peak``: the peak, sky-subtracted, pixel value of the object.
        * ``flux``: the object sky-subtracted flux, calculated by
          summing object pixels over the Gaussian kernel.  The
          derivation matches that of `starfind`_ if ``sky`` is ``None``.
        * ``mag``: the object instrumental magnitude calculated as
          ``-2.5 * log10(flux)``.  The derivation matches that of
          `starfind`_ if ``sky`` is ``None``.

    See Also
    --------
    daofind

    Notes
    -----
    For the convolution step, this routine sets pixels beyond the image
    borders to 0.0.  The equivalent parameters in `starfind`_ are
    ``boundary='constant'`` and ``constant=0.0``.

    IRAF's `starfind`_ uses ``hwhmpsf``, ``fradius``, and ``sepmin`` as
    input parameters.  The equivalent input values for ``irafstarfind``
    are:

    * ``fwhm = hwhmpsf * 2``
    * ``sigma_radius = fradius * sqrt(2.0*log(2.0))``
    * ``minsep_fwhm = 0.5 * sepmin``

    The main differences between ``daofind`` and ``irafstarfind`` are:

    * ``irafstarfind`` always uses a 2D circular Gaussian kernel,
      while ``daofind`` can use an elliptical Gaussian kernel.

    * ``irafstarfind`` calculates the objects' centroid, roundness,
      and sharpness using image moments.

    References
    ----------
    .. [1] http://iraf.net/irafhelp.php?val=starfind&help=Help+Page
    .. [2] http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?starfind
    """

    starfind_kernel = _FindObjKernel(fwhm, ratio=1.0, theta=0.0,
                                     sigma_radius=sigma_radius)
    min_separation = max(2, int((fwhm * minsep_fwhm) + 0.5))
    objs = _findobjs(data, threshold, starfind_kernel,
                     min_separation=min_separation,
                     exclude_border=exclude_border)
    tbl = _irafstarfind_properties(objs, starfind_kernel, sky)
    if len(objs) == 0:
        warnings.warn('No sources were found.', AstropyUserWarning)
        return tbl     # empty table
    table_mask = ((tbl['sharpness'] > sharplo) &
                  (tbl['sharpness'] < sharphi) &
                  (tbl['roundness'] > roundlo) &
                  (tbl['roundness'] < roundhi))
    tbl = tbl[table_mask]
    idcol = Column(name='id', data=np.arange(len(tbl)) + 1)
    tbl.add_column(idcol, 0)
    if len(tbl) == 0:
        warnings.warn('Sources were found, but none pass the sharpness and '
                      'roundness criteria.', AstropyUserWarning)
    return tbl


def _findobjs(data, threshold, kernel, min_separation=None,
              exclude_border=False, local_peaks=True):
    """
    Find sources in an image by convolving the image with the input
    kernel and selecting connected pixels above a given threshold.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    threshold : float
        The absolute image value above which to select sources.  Note
        that this threshold is not the same threshold input to
        ``daofind`` or ``irafstarfind``.  It should be multiplied by the
        kernel relerr.

    kernel : `_FindObjKernel`
        The convolution kernel.  The dimensions should match those of
        the cutouts.  The kernel should be normalized to zero sum.

    exclude_border : bool, optional
        Set to `True` to exclude sources found within half the size of
        the convolution kernel from the image borders.  The default is
        `False`, which is the mode used by `DAOFIND`_ and `starfind`_.

    local_peaks : bool, optional
        Set to `True` to exactly match the `DAOFIND`_ method of finding
        local peaks.  If `False`, then only one peak per thresholded
        segment will be used.

    Returns
    -------
    objects : list of `_ImgCutout`
        A list of `_ImgCutout` objects containing the image cutout for
        each source.
    """

    from scipy import ndimage

    x_kernradius = kernel.kern.shape[1] // 2
    y_kernradius = kernel.kern.shape[0] // 2

    if not exclude_border:
        # create a larger image padded by zeros
        ysize = int(data.shape[0] + (2. * y_kernradius))
        xsize = int(data.shape[1] + (2. * x_kernradius))
        data_padded = np.zeros((ysize, xsize))
        data_padded[y_kernradius:y_kernradius + data.shape[0],
                    x_kernradius:x_kernradius + data.shape[1]] = data
        data = data_padded

    convolved_data = _convolve_data(data, kernel.kern, mode='constant',
                                    fill_value=0.0, check_normalization=False)

    if not exclude_border:
        # keep border=0 in convolved data
        convolved_data[:y_kernradius, :] = 0.
        convolved_data[-y_kernradius:, :] = 0.
        convolved_data[:, :x_kernradius] = 0.
        convolved_data[:, -x_kernradius:] = 0.

    selem = ndimage.generate_binary_structure(2, 2)
    object_labels, nobjects = ndimage.label(convolved_data > threshold,
                                            structure=selem)
    objects = []
    if nobjects == 0:
        return objects

    # find object peaks in the convolved data
    if local_peaks:
        # footprint overrides min_separation in find_peaks
        if min_separation is None:   # daofind
            footprint = kernel.mask.astype(np.bool)
        else:
            from skimage.morphology import disk
            footprint = disk(min_separation)
        tbl = find_peaks(convolved_data, threshold, footprint=footprint)
        coords = np.transpose([tbl['y_peak'], tbl['x_peak']])
    else:
        object_slices = ndimage.find_objects(object_labels)
        coords = []
        for object_slice in object_slices:
            # thresholded_object is not the same size as the kernel
            thresholded_object = convolved_data[object_slice]
            ypeak, xpeak = np.unravel_index(thresholded_object.argmax(),
                                            thresholded_object.shape)
            xpeak += object_slice[1].start
            ypeak += object_slice[0].start
            coords.append((ypeak, xpeak))

    for (ypeak, xpeak) in coords:
        # now extract the object from the data, centered on the peak
        # pixel in the convolved image, with the same size as the kernel
        x0 = xpeak - x_kernradius
        x1 = xpeak + x_kernradius + 1
        y0 = ypeak - y_kernradius
        y1 = ypeak + y_kernradius + 1
        if x0 < 0 or x1 > data.shape[1]:
            continue    # pragma: no cover (isolated continue is never tested)
        if y0 < 0 or y1 > data.shape[0]:
            continue    # pragma: no cover (isolated continue is never tested)
        object_data = data[y0:y1, x0:x1]
        object_convolved_data = convolved_data[y0:y1, x0:x1].copy()
        if not exclude_border:
            # correct for image padding
            x0 -= x_kernradius
            y0 -= y_kernradius
        imgcutout = _ImgCutout(object_data, object_convolved_data, x0, y0)
        objects.append(imgcutout)
    return objects


def _irafstarfind_properties(imgcutouts, kernel, sky=None):
    """
    Find the properties of each detected source, as defined by IRAF's
    ``starfind``.

    Parameters
    ----------
    imgcutouts : list of `_ImgCutout`
        A list of `_ImgCutout` objects containing the image cutout for
        each source.

    kernel : `_FindObjKernel`
        The convolution kernel.  The dimensions should match those of
        the cutouts.  ``kernel.gkernel`` should have a peak pixel value
        of 1.0 and not contain any masked pixels.

    sky : float, optional
        The absolute sky level.  If sky is ``None``, then a local sky
        level will be estimated (in a crude fashion).

    Returns
    -------
    table : `~astropy.table.Table`
        A table of the objects' properties.
    """

    result = defaultdict(list)
    for imgcutout in imgcutouts:
        if sky is None:
            skymask = ~kernel.mask.astype(np.bool)   # 1=sky, 0=obj
            nsky = np.count_nonzero(skymask)
            if nsky == 0:
                meansky = imgcutout.data.max() - imgcutout.convdata.max()
            else:
                meansky = (imgcutout.data * skymask).sum() / nsky
        else:
            meansky = sky
        objvals = _irafstarfind_moments(imgcutout, kernel, meansky)
        for key, val in objvals.items():
            result[key].append(val)
    names = ['xcentroid', 'ycentroid', 'fwhm', 'sharpness', 'roundness',
             'pa', 'npix', 'sky', 'peak', 'flux', 'mag']
    if len(result) == 0:
        for name in names:
            result[name] = []
    table = Table(result, names=names)
    return table


def _irafstarfind_moments(imgcutout, kernel, sky):
    """
    Find the properties of each detected source, as defined by IRAF's
    ``starfind``.

    Parameters
    ----------
    imgcutout : `_ImgCutout`
        The image cutout for a single detected source.

    kernel : `_FindObjKernel`
        The convolution kernel.  The dimensions should match those of
        ``imgcutout``.  ``kernel.gkernel`` should have a peak pixel
        value of 1.0 and not contain any masked pixels.

    sky : float
        The local sky level around the source.

    Returns
    -------
    result : dict
        A dictionary of the object parameters.
    """

    from skimage.measure import moments, moments_central

    result = defaultdict(list)
    img = np.array((imgcutout.data - sky) * kernel.mask)
    img = np.where(img > 0, img, 0)    # starfind discards negative pixels
    if np.count_nonzero(img) <= 1:
        return {}
    m = moments(img, 1)
    result['xcentroid'] = m[1, 0] / m[0, 0]
    result['ycentroid'] = m[0, 1] / m[0, 0]
    result['npix'] = float(np.count_nonzero(img))   # float for easier testing
    result['sky'] = sky
    result['peak'] = np.max(img)
    flux = img.sum()
    result['flux'] = flux
    result['mag'] = -2.5 * np.log10(flux)
    mu = moments_central(
        img, result['ycentroid'], result['xcentroid'], 2) / m[0, 0]
    musum = mu[2, 0] + mu[0, 2]
    mudiff = mu[2, 0] - mu[0, 2]
    result['fwhm'] = 2.0 * np.sqrt(np.log(2.0) * musum)
    result['sharpness'] = result['fwhm'] / kernel.fwhm
    result['roundness'] = np.sqrt(mudiff**2 + 4.0*mu[1, 1]**2) / musum
    pa = 0.5 * np.arctan2(2.0 * mu[1, 1], mudiff) * (180.0 / np.pi)
    if pa < 0.0:
        pa += 180.0
    result['pa'] = pa
    result['xcentroid'] += imgcutout.x0
    result['ycentroid'] += imgcutout.y0
    return result


def _daofind_properties(imgcutouts, threshold, kernel, sky=0.0):
    """
    Find the properties of each detected source, as defined by
    `DAOFIND`_.

    Parameters
    ----------
    imgcutouts : list of `_ImgCutout`
        A list of `_ImgCutout` objects containing the image cutout for
        each source.

    threshold : float
        The absolute image value above which to select sources.

    kernel : `_FindObjKernel`
        The convolution kernel.  The dimensions should match those of
        the objects in ``imgcutouts``.  ``kernel.gkernel`` should have a
        peak pixel value of 1.0 and not contain any masked pixels.

    sky : float, optional
        The local sky level around the source.  ``sky`` is used only to
        calculate the source peak value and flux.  The default is 0.0.

    Returns
    -------
    table : `~astropy.table.Table`
        A table of the object parameters.
    """

    result = defaultdict(list)
    ykcen, xkcen = kernel.center
    for imgcutout in imgcutouts:
        convobj = imgcutout.convdata.copy()
        convobj[ykcen, xkcen] = 0.0
        q1 = convobj[0:ykcen+1, xkcen+1:]
        q2 = convobj[0:ykcen, 0:xkcen+1]
        q3 = convobj[ykcen:, 0:xkcen]
        q4 = convobj[ykcen+1:, xkcen:]
        sum2 = -q1.sum() + q2.sum() - q3.sum() + q4.sum()
        sum4 = np.abs(convobj).sum()
        result['roundness1'].append(2.0 * sum2 / sum4)

        obj = imgcutout.data
        objpeak = obj[ykcen, xkcen]
        convpeak = imgcutout.convdata[ykcen, xkcen]
        npts = kernel.mask.sum()
        obj_masked = obj * kernel.mask
        objmean = (obj_masked.sum() - objpeak) / (npts - 1)   # exclude peak
        sharp = (objpeak - objmean) / convpeak
        result['sharpness'].append(sharp)

        dx, dy, g_roundness = _daofind_centroid_roundness(obj, kernel)
        yc, xc = imgcutout.center
        result['xcentroid'].append(xc + dx)
        result['ycentroid'].append(yc + dy)
        result['roundness2'].append(g_roundness)
        result['sky'].append(sky)      # DAOFIND uses sky=0
        result['npix'].append(float(obj.size))
        result['peak'].append(objpeak - sky)
        flux = (convpeak / threshold) - (sky * obj.size)
        result['flux'].append(flux)
        if flux <= 0:
            mag = np.nan
        else:
            mag = -2.5 * np.log10(flux)
        result['mag'].append(mag)

    names = ['xcentroid', 'ycentroid', 'sharpness', 'roundness1', 'roundness2',
             'npix', 'sky', 'peak', 'flux', 'mag']
    if len(result) == 0:
        for name in names:
            result[name] = []
    table = Table(result, names=names)
    return table


def _daofind_centroid_roundness(obj, kernel):
    """
    Calculate the source (x, y) centroid and `DAOFIND`_ "GROUND"
    roundness statistic.

    `DAOFIND`_ finds the centroid by fitting 1D Gaussians (marginal x/y
    distributions of the kernel) to the marginal x/y distributions of
    the original (unconvolved) image.

    The roundness statistic measures the ratio of the difference in the
    height of the best fitting Gaussian function in x minus the best
    fitting Gaussian function in y, divided by the average of the best
    fitting Gaussian functions in x and y.  A circular source will have
    a zero roundness.  An source extended in x (y) will have a negative
    (positive) roundness.

    Parameters
    ----------
    obj : array_like
        The 2D array of the source cutout.

    kernel : `_FindObjKernel`
        The convolution kernel.  The dimensions should match those of
        ``obj``.  ``kernel.gkernel`` should have a peak pixel value of
        1.0 and not contain any masked pixels.

    Returns
    -------
    dx, dy : float
        Fractional shift in x and y of the image centroid relative to
        the maximum pixel.

    g_roundness : float
        `DAOFIND`_ roundness (GROUND) statistic.
    """

    dx, hx = _daofind_centroidfit(obj, kernel, axis=0)
    dy, hy = _daofind_centroidfit(obj, kernel, axis=1)
    g_roundness = 2.0 * (hx - hy) / (hx + hy)
    return dx, dy, g_roundness


def _daofind_centroidfit(obj, kernel, axis):
    """
    Find the source centroid along one axis by fitting a 1D Gaussian to
    the marginal x or y distribution of the unconvolved source data.

    Parameters
    ----------
    obj : array_like
        The 2D array of the source cutout.

    kernel : `_FindObjKernel`
        The convolution kernel.  The dimensions should match those of
        ``obj``.  ``kernel.gkernel`` should have a peak pixel value of
        1.0 and not contain any masked pixels.

    axis : {0, 1}
        The axis for which the centroid is computed:

        * 0: for the x axis
        * 1: for the y axis

    Returns
    -------
    dx : float
        Fractional shift in x or y (depending on ``axis`` value) of the
        image centroid relative to the maximum pixel.

    hx : float
        Height of the best-fitting Gaussian to the marginal x or y
        (depending on ``axis`` value) distribution of the unconvolved
        source data.
    """

    # define a triangular weighting function, peaked in the middle
    # and equal to one at the edge
    nyk, nxk = kernel.shape
    ykrad, xkrad = kernel.center
    ywtd, xwtd = np.mgrid[0:nyk, 0:nxk]
    xwt = xkrad - abs(xwtd - xkrad) + 1.0
    ywt = ykrad - abs(ywtd - ykrad) + 1.0
    if axis == 0:
        wt = xwt[0]
        wts = ywt
        ksize = nxk
        kernel_sigma = kernel.xsigma
        krad = ksize // 2
        sumdx_vec = krad - np.arange(ksize)
    elif axis == 1:
        wt = ywt.T[0]
        wts = xwt
        ksize = nyk
        kernel_sigma = kernel.ysigma
        krad = ksize // 2
        sumdx_vec = np.arange(ksize) - krad
    n = wt.sum()

    sg = (kernel.gkernel * wts).sum(axis)
    sumg = (wt * sg).sum()
    sumg2 = (wt * sg**2).sum()
    vec = krad - np.arange(ksize)
    dgdx = sg * vec
    sdgdx = (wt * dgdx).sum()
    sdgdx2 = (wt * dgdx**2).sum()
    sgdgdx = (wt * sg * dgdx).sum()
    sd = (obj * wts).sum(axis)
    sumd = (wt * sd).sum()
    sumgd = (wt * sg * sd).sum()
    sddgdx = (wt * sd * dgdx).sum()
    sumdx = (wt * sd * sumdx_vec).sum()
    # linear least-squares fit (data = sky + hx*gkernel) to find amplitudes
    denom = (n*sumg2 - sumg**2)
    hx = (n*sumgd - sumg*sumd) / denom
    # sky = (sumg2*sumd - sumg*sumgd) / denom
    dx = (sgdgdx - (sddgdx - sdgdx*sumd)) / (hx * sdgdx2 / kernel_sigma**2)

    hsize = (ksize / 2.)
    if abs(dx) > hsize:
        dx = 0
        if sumd == 0:
            dx = 0.0
        else:
            dx = float(sumdx / sumd)
            if abs(dx) > hsize:
                dx = 0.0
    return dx, hx


class _ImgCutout(object):
    """Class to hold image cutouts."""
    def __init__(self, data, convdata, x0, y0):
        """
        Parameters
        ----------
        data : array_like
            The cutout 2D image from the input unconvolved 2D image.

        convdata : array_like
            The cutout 2D image from the convolved 2D image.

        x0, y0 : float
            Image coordinates of the lower left pixel of the cutout region.
            The pixel origin is (0, 0).
        """

        self.data = data
        self.convdata = convdata
        self.x0 = x0
        self.y0 = y0

    @property
    def radius(self):
        return [size // 2 for size in self.data.shape]

    @property
    def center(self):
        yr, xr = self.radius
        return yr + self.y0, xr + self.x0


class _FindObjKernel(object):
    """
    Calculate a 2D Gaussian density enhancement kernel.  This kernel has
    negative wings and sums to zero.  It is used by both `daofind` and
    `irafstarfind`.

    Parameters
    ----------
    fwhm : float
        The full-width half-maximum (FWHM) of the major axis of the
        Gaussian kernel in units of pixels.

    ratio : float, optional
        The ratio of the minor to major axis standard deviations of the
        Gaussian kernel.  ``ratio`` must be strictly positive and less
        than or equal to 1.0.  The default is 1.0 (i.e., a circular
        Gaussian kernel).

    theta : float, optional
        The position angle (in degrees) of the major axis of the
        Gaussian kernel measured counter-clockwise from the positive x
        axis.

    sigma_radius : float, optional
        The truncation radius of the Gaussian kernel in units of sigma
        (standard deviation) [``1 sigma = FWHM /
        2.0*sqrt(2.0*log(2.0))``].  The default is 1.5.

    Notes
    -----
    The object attributes include the dimensions of the elliptical
    kernel and the coefficients of a 2D elliptical Gaussian function
    expressed as:

        ``f(x,y) = A * exp(-g(x,y))``

        where

        ``g(x,y) = a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2``

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Gaussian_function
    """

    def __init__(self, fwhm, ratio=1.0, theta=0.0, sigma_radius=1.5):
        assert fwhm > 0, 'FWHM must be positive'
        assert ((ratio > 0) & (ratio <= 1)), \
            'ratio must be positive and less than 1'
        assert sigma_radius > 0, 'sigma_radius must be positive'
        self.fwhm = fwhm
        self.sigma_radius = sigma_radius
        self.ratio = ratio
        self.theta = theta
        self.theta_radians = np.deg2rad(self.theta)
        self.xsigma = self.fwhm * gaussian_fwhm_to_sigma
        self.ysigma = self.xsigma * self.ratio
        self.a = None
        self.b = None
        self.c = None
        self.f = None
        self.nx = None
        self.ny = None
        self.xc = None
        self.yc = None
        self.circrad = None
        self.ellrad = None
        self.gkernel = None
        self.mask = None
        self.npts = None
        self.kern = None
        self.relerr = None
        self.set_gausspars()
        self.mk_kern()

    @property
    def shape(self):
        return self.kern.shape

    @property
    def center(self):
        """Index of the kernel center."""
        return [size // 2 for size in self.kern.shape]

    def set_gausspars(self):
        xsigma2 = self.xsigma**2
        ysigma2 = self.ysigma**2
        cost = np.cos(self.theta_radians)
        sint = np.sin(self.theta_radians)
        self.a = (cost**2 / (2.0 * xsigma2)) + (sint**2 / (2.0 * ysigma2))
        self.b = 0.5 * cost * sint * (1.0/xsigma2 - 1.0/ysigma2)    # CCW
        self.c = (sint**2 / (2.0 * xsigma2)) + (cost**2 / (2.0 * ysigma2))
        # find the extent of an ellipse with radius = sigma_radius*sigma;
        # solve for the horizontal and vertical tangents of an ellipse
        # defined by g(x,y) = f
        self.f = self.sigma_radius**2 / 2.0
        denom = self.a*self.c - self.b**2
        self.nx = 2 * int(max(2, math.sqrt(self.c*self.f / denom))) + 1
        self.ny = 2 * int(max(2, math.sqrt(self.a*self.f / denom))) + 1
        return

    def mk_kern(self):
        yy, xx = np.mgrid[0:self.ny, 0:self.nx]
        self.xc = self.nx // 2
        self.yc = self.ny // 2
        self.circrad = np.sqrt((xx-self.xc)**2 + (yy-self.yc)**2)
        self.ellrad = (self.a*(xx-self.xc)**2 +
                       2.0*self.b*(xx-self.xc)*(yy-self.yc) +
                       self.c*(yy-self.yc)**2)
        self.gkernel = np.exp(-self.ellrad)
        self.mask = np.where((self.ellrad <= self.f) |
                             (self.circrad <= 2.0), 1, 0).astype(np.int16)
        self.npts = self.mask.sum()
        self.kern = self.gkernel * self.mask
        # normalize the kernel to zero sum (denom = variance * npts)
        denom = ((self.kern**2).sum() - (self.kern.sum()**2 / self.npts))
        self.relerr = 1.0 / np.sqrt(denom)
        self.kern = (((self.kern - (self.kern.sum() / self.npts)) / denom) *
                     self.mask)
        return
