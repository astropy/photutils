# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from astropy import log


__all__ = ['Centerer']


IN_MASK = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

OUT_MASK = [
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
]


class Centerer(object):
    """
    Class to find the center of a galaxy.

    The isophote fit algorithm requires an initial guess for the galaxy
    center (x, y) coordinates and these coordinates must be close to the
    actual galaxy center for the isophote fit to work.  This class
    provides an object centerer function to determine an initial guess
    for the the galaxy center coordinates.  See the **Notes** section
    below for more details.

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        The image array.  Masked arrays are not recognized here. This
        assumes that centering should always be done on valid pixels.
    geometry : `~photutils.isophote.Geometry` instance
        The `~photutils.isophote.Geometry` object that directs the
        centerer to look at its (x, y) coordinates.  These coordinates
        are modified by the centerer algorithm.
    verbose : bool, optional
        Whether to print object centering information.  The default is
        `True`.

    Attributes
    ----------
    threshold : float
        The centerer threshold value.

    Notes
    -----
    The centerer function scans a 10x10 window centered on the (x, y)
    coordinates in the `~photutils.isophote.Geometry` instance passed to
    the constructor of the `~photutils.isophote.Ellipse` class.  If any
    of the `~photutils.isophote.Geometry` (x, y) coordinates are `None`,
    the center of the input image frame is used.  If the center
    acquisition is successful, the `~photutils.isophote.Geometry`
    instance is modified in place to reflect the solution of the object
    centerer algorithm.

    In some cases the object centerer algorithm may fail even though
    there is enough signal-to-noise to start a fit (e.g. objects with
    very high ellipticity).  In those cases the sensitivity of the
    algorithm can be decreased by decreasing the value of the object
    centerer threshold parameter.  The centerer works by looking where a
    quantity akin to a signal-to-noise ratio is maximized within the
    10x10 window.  The centerer can thus be shut off entirely by setting
    the threshold to a large value (i.e. >> 1; meaning no location
    inside the search window will achieve that signal-to-noise ratio).
    """

    def __init__(self, image, geometry, verbose=True):
        self._image = image
        self._geometry = geometry
        self._verbose = verbose
        self._mask_half_size = len(IN_MASK) / 2
        self.threshold = None

        # number of pixels in each mask
        sz = len(IN_MASK)
        self._ones_in = np.ma.masked_array(np.ones(shape=(sz, sz)),
                                           mask=IN_MASK)
        self._ones_out = np.ma.masked_array(np.ones(shape=(sz, sz)),
                                            mask=OUT_MASK)

        self._in_mask_npix = np.sum(self._ones_in)
        self._out_mask_npix = np.sum(self._ones_out)

    def center(self, threshold=0.1):
        """
        Calculate the object center.

        The ``._geometry`` attribute position will be modified
        with the object center.

        Parameters
        ----------
        threshold : float, optional
            The object centerer threshold.  To turn off the centerer,
            set this to a large value (i.e. >> 1).  The default is 0.1.
        """

        self.threshold = threshold

        # Check if center coordinates point to somewhere inside the frame.
        # If not, set then to frame center.
        _x0 = self._geometry.x0
        _y0 = self._geometry.y0
        if (_x0 is None or _x0 < 0 or _x0 >= self._image.shape[0] or
                _y0 is None or _y0 < 0 or _y0 >= self._image.shape[1]):
            _x0 = self._image.shape[0] / 2
            _y0 = self._image.shape[1] / 2

        max_fom = 0.
        max_i = 0
        max_j = 0

        # scan all positions inside window
        window_half_size = 5
        for i in range(int(_x0 - window_half_size),
                       int(_x0 + window_half_size) + 1):
            for j in range(int(_y0 - window_half_size),
                           int(_y0 + window_half_size) + 1):

                # ensure that it stays inside image frame
                i1 = int(max(0, i - self._mask_half_size))
                j1 = int(max(0, j - self._mask_half_size))
                i2 = int(min(self._image.shape[0] - 1,
                             i + self._mask_half_size))
                j2 = int(min(self._image.shape[1] - 1,
                             j + self._mask_half_size))

                window = self._image[j1:j2, i1:i2]

                # averages in inner and outer regions.
                inner = np.ma.masked_array(window, mask=IN_MASK)
                outer = np.ma.masked_array(window, mask=OUT_MASK)
                inner_avg = np.sum(inner) / self._in_mask_npix
                outer_avg = np.sum(outer) / self._out_mask_npix

                # standard deviation and figure of merit
                inner_std = np.std(inner)
                outer_std = np.std(outer)
                stddev = np.sqrt(inner_std**2 + outer_std**2)

                fom = (inner_avg - outer_avg) / stddev

                if fom > max_fom:
                    max_fom = fom
                    max_i = i
                    max_j = j

        # figure of merit > threshold: update geometry with new coordinates.
        if max_fom > threshold:
            self._geometry.x0 = float(max_i)
            self._geometry.y0 = float(max_j)

            if self._verbose:
                log.info("Found center at x0 = {0:5.1f}, y0 = {1:5.1f}"
                         .format(self._geometry.x0, self._geometry.y0))
        else:
            if self._verbose:
                log.info('Result is below the threshold -- keeping the '
                         'original coordinates.')
