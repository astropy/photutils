# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools to build an ePSF.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import copy
import warnings

import numpy as np
from astropy.stats import SigmaClip
from astropy.table import Table

from .centroid import find_peak
from .epsf_fitter import EPSFFitter, compute_residuals
from .models import NonNormalizable, PSF2DModel
from .utils import (py2round, interpolate_missing_data, _pixstat, _smoothPSF,
                    _parse_tuple_pars)


__all__ = ['EPSFBuilder']


class EPSFBuilder(object):
    def __init__(self, peak_fit_box=5, peak_search_box='fitbox',
                 recenter_accuracy=1.0e-4, recenter_max_iters=1000,
                 ignore_badfit_stars=True, stat='median',
                 sigma_clip=SigmaClip(sigma=3., iters=10),
                 smoothing_kernel='quar', fitter=EPSFFitter(residuals=True),
                 max_iters=50, accuracy=1e-4, epsf=None):

        self.peak_fit_box = peak_fit_box
        self.peak_search_box = peak_search_box

        recenter_accuracy = float(recenter_accuracy)
        if recenter_accuracy <= 0.0:
            raise ValueError('recenter_accuracy must be a strictly positive '
                             'number.')
        self.recenter_accuracy = recenter_accuracy

        recenter_max_iters = int(recenter_max_iters)
        if recenter_max_iters <= 0:
            raise ValueError('recenter_max_iters must be a positive integer.')
        self.recenter_max_iters = recenter_max_iters

        self.ignore_badfit_stars = ignore_badfit_stars
        self.stat = stat
        self.sigma_clip = sigma_clip
        self.smoothing_kernel = smoothing_kernel
        self.fitter = fitter

        max_iters = int(max_iters)
        if max_iters <= 0:
            raise ValueError('max_iters must be a positive number.')
        self.max_iters = max_iters

        if accuracy <= 0.0:
            raise ValueError('accuracy must be a positive number.')
        self.accuracy2 = accuracy**2

        self.epsf = epsf

    def __call__(self, stars):
        return self.build_psf(stars)

    def _build_psf_step(self, stars, psf=None):
        if len(stars) < 1:
            raise ValueError('stars must be a list containing at least '
                             'one Star object.')

        if psf is None:
            psf = init_psf(stars)
        elif isinstance(psf, type):
            psf = init_psf(stars, psf_cls=psf)
        else:
            psf = copy.deepcopy(psf)

        cx, cy = psf.origin
        ny, nx = psf.shape
        pscale_x, pscale_y = psf.pixel_scale

        # get all stars including linked stars as a flat list
        all_stars = []
        for s in stars:
            all_stars += s.get_linked_list()

        # allocate "accumulator" array (to store transformed PSFs):
        apsf = [[[] for k in range(nx)] for k in range(ny)]

        norm_psf_data = psf.normalized_data

        for s in all_stars:
            if s.ignore:
                continue

            if (self.ignore_badfit_stars and
                    s.fit_error_status is not None and
                    s.fit_error_status > 0):
                continue

            pixlist = s.centered_plist(normalized=True)

            # evaluate previous PSF model at star pixel location in
            # the PSF grid
            ovx = s.pixel_scale[0] / pscale_x
            ovy = s.pixel_scale[1] / pscale_y
            x = ovx * (pixlist[:, 0])
            y = ovy * (pixlist[:, 1])
            old_model_vals = psf.evaluate(x=x, y=y, flux=1.0, x_0=0.0,
                                          y_0=0.0)

            # find integer location of star pixels in the PSF grid
            # and compute residuals
            ix = py2round(x + cx).astype(np.int)
            iy = py2round(y + cy).astype(np.int)
            pv = pixlist[:, 2] / (ovx * ovy) - old_model_vals
            m = np.logical_and(np.logical_and(ix >= 0, ix < nx),
                               np.logical_and(iy >= 0, iy < ny))

            # add all pixel values to the corresponding accumulator
            for i, j, v in zip(ix[m], iy[m], pv[m]):
                apsf[j][i].append(v)

        psfdata = np.empty((ny, nx), dtype=np.float)
        psfdata.fill(np.nan)

        for i in range(nx):
            for j in range(ny):
                psfdata[j, i] = _pixstat(apsf[j][i], stat=self.stat,
                                         sigma_clip=self.sigma_clip,
                                         default=np.nan)

        mask = np.isfinite(psfdata)
        if not np.all(mask):
            # fill in the "holes" (=np.nan) using interpolation
            # I. using cubic spline for inner holes
            psfdata = interpolate_missing_data(psfdata, method='cubic',
                                               mask=mask)

            # II. we fill outer points with zeros
            mask = np.isfinite(psfdata)
            psfdata[np.logical_not(mask)] = 0.0

        # add residuals to old PSF data:
        psfdata += norm_psf_data

        # apply a smoothing kernel to the PSF:
        psfdata = _smoothPSF(psfdata, self.smoothing_kernel)

        shift_x = 0
        shift_y = 0
        peak_eps2 = self.recenter_accuracy**2
        eps2_prev = None
        y, x = np.indices(psfdata.shape, dtype=np.float)
        ePSF = psf.make_similar_from_data(psfdata)
        ePSF.fill_value = 0.0

        for iteration in range(self.recenter_max_niters):
            # find peak location:
            peak_x, peak_y = find_peak(psfdata, xmax=cx, ymax=cy,
                                       peak_fit_box=self.peak_fit_box,
                                       peak_search_box=self.peak_search_box,
                                       mask=None)

            dx = cx - peak_x
            dy = cy - peak_y

            eps2 = dx**2 + dy**2
            if ((eps2_prev is not None and eps2 > eps2_prev)
                    or eps2 < peak_eps2):
                break
            eps2_prev = eps2

            shift_x += dx
            shift_y += dy

            # Resample PSF data to a shifted grid such that the pick of
            # the PSF is at expected position
            psfdata = ePSF.evaluate(x=x, y=y, flux=1.0,
                                    x_0=shift_x + cx, y_0=shift_y + cy)

        # apply final shifts and fill in any missing data
        if shift_x != 0.0 or shift_y != 0.0:
            ePSF.fill_value = np.nan
            psfdata = ePSF.evaluate(x=x, y=y, flux=1.0,
                                    x_0=shift_x + cx, y_0=shift_y + cy)

            # fill in the "holes" (=np.nan) using 0 (no contribution to
            # the flux)
            mask = np.isfinite(psfdata)
            psfdata[np.logical_not(mask)] = 0.0

        norm = np.abs(np.sum(psfdata, dtype=np.float64))
        psfdata /= norm

        # Create ePSF model and return:
        ePSF = psf.make_similar_from_data(psfdata)

        return ePSF

    def build_psf(self, stars, psf=None):
        """
        Iteratively build the psf.
        """

        #if isinstance(stars, Table):
        #    self.stars = self._extract_stars(
        #        data, stars, common_catalog=None, extract_size=11,
        #        recenter=False, peak_fit_box=5, peak_search_box='fitbox',
        #        catmap={'x': 'x', 'y': 'y', 'lon': 'lon', 'lat': 'lat',
        #                'weight': 'weight', 'id': 'id'}, cat_name_kwd='name',
        #        image_name_kwd='name')
        #else:
        #    # TODO: check if stars is a list of Stars
        #    self.stars = stars

        self.stars = stars

        # get all stars (including linked stars) as a flat list
        all_stars = []
        for s in self.stars:
            all_stars += s.get_linked_list()
        nstars = len(all_stars)

        # create an array of star centers
        prev_centers = np.array([s.center for s in stars], dtype=np.float)

        # initialize array for detection of oscillatory behavior
        oscillatory = np.zeros(nstars, dtype=bool)
        prev_failed = np.zeros(nstars, dtype=bool)
        dxy = np.zeros((nstars, 2), dtype=np.float)

        iter_num = 0
        eps2 = 2.0 * self.accuracy2
        while (iter_num < self.max_iters and np.amax(eps2) >= self.accuracy2
               and not np.all(oscillatory)):
            iter_num += 1

            # build/improve the PSF
            psf = self._build_psf_step(stars, psf=psf)

            # fit the new PSF to the stars to find improved centers
            stars = self.fitter(stars, psf)

            # get all stars (including linked stars) as a flat list
            all_stars = []
            for s in stars:
                all_stars += s.get_linked_list()

            # create an array of star centers at this iteration
            centers = np.array([s.center for s in stars], dtype=np.float)

            # detect oscillatory behavior
            failed = np.array([s.fit_error_status > 0 for s in stars],
                              dtype=np.bool)

            # exclude oscillatory stars after 3 iterations
            if iter_num > 3:
                oscillatory = np.logical_and(prev_failed,
                                             np.logical_not(failed))
                for s, osc in zip(stars, oscillatory):
                    s.ignore = bool(osc)
                prev_failed = failed

            # check termination criterion
            good_mask = np.logical_not(np.logical_or(failed, oscillatory))
            dxy = centers - prev_centers
            mdxy = dxy[good_mask]
            eps2 = np.sum(mdxy * mdxy, axis=1, dtype=np.float64)
            prev_centers = centers

        # TODO: make compute_residuals a method of Stars
        # compute residuals
        #if residuals:
        #    res = compute_residuals(psf, all_stars)
        #else:
        #    res = len(all_stars) * [None]
        #
        #for s, r in zip(all_stars, res):
        #    s.fit_residual = r

        # assign coordinate residuals of the iterative process
        for s, (dx, dy) in zip(all_stars, dxy):
            s.iter_fit_eps = (float(dx), float(dy))

        # TODO: return iter_num as Stars attribute
        return psf, stars


def init_psf(stars, shape=None, oversampling=4.0, pixel_scale=None,
             psf_cls=PSF2DModel, **kwargs):
    """
    Parameters
    ----------
    stars : Star, list of Star
        A list of :py:class:`~psfutils.catalogs.Star` objects
        containing image data of star cutouts that are used to "build" a PSF.

    psf : PSF2DModel, type, None, optional
        An existing approximation to the PSF which needs to be recomputed
        using new ``stars`` (with new parameters for center and flux)
        or a class indicating the type of the ``psf`` object to be created
        (preferebly a subclass of :py:class:`PSF2DModel`).
        If `psf` is `None`, a new ``psf`` will be computed using
        :py:class:`PSF2DModel`.

    shape : tuple, optional
        Numpy-style shape of the output PSF. If shape is not specified (i.e.,
        it is set to `None`), the shape will be derived from the sizes of the
        input star models. This is ignored if `psf` is not `None`.

    oversampling : float, tuple of float, list of (float, tuple of float), \
optional
        Oversampling factor of the PSF relative to star. It indicates how many
        times a pixel in the star's image should be sampled when creating PSF.
        If a single number is provided, that value will be used for both
        ``X`` and ``Y`` axes and for all stars. When a tuple is provided,
        first number will indicate oversampling along the ``X`` axis and the
        second number will indicate oversampling along the ``Y`` axis. It is
        also possible to have individualized oversampling factors for each star
        by providing a list of integers or tuples of integers.

    """
    # check parameters:
    if pixel_scale is None and oversampling is None:
        raise ValueError(
            "'oversampling' and 'pixel_scale' cannot be 'None' together. "
            "At least one of these two parameters must be provided."
        )

    # get all stars including linked stars as a flat list:
    all_stars = []
    for s in stars:
        all_stars += s.get_linked_list()

    # find pixel scale:
    if pixel_scale is None:
        ovx, ovy = _parse_tuple_pars(oversampling, name='oversampling',
                                     dtype=float)

        # compute PSF's pixel scale as the smallest scale of the stars
        # divided by the requested oversampling factor:
        pscale_x, pscale_y = all_stars[0].pixel_scale
        for s in all_stars[1:]:
            px, py = s.pixel_scale
            if px < pscale_x:
                pscale_x = px
            if py < pscale_y:
                pscale_y = py

        pscale_x /= ovx
        pscale_y /= ovy

    else:
        pscale_x, pscale_y = _parse_tuple_pars(pixel_scale, name='pixel_scale',
                                               dtype=float)

    # if shape is None, find the minimal shape that will include input star's
    # data:
    if shape is None:
        w = np.array([((s.x_center + 0.5) * s.pixel_scale[0] / pscale_x,
                       (s.nx - s.x_center - 0.5) * s.pixel_scale[0] / pscale_x)
                      for s in all_stars])
        h = np.array([((s.y_center + 0.5) * s.pixel_scale[1] / pscale_y,
                       (s.ny - s.y_center - 0.5) * s.pixel_scale[1] / pscale_y)
                      for s in all_stars])

        # size of the PSF in the input image pixels
        # (the image with min(pixel_scale)):
        nx = int(np.ceil(np.amax(w / ovx + 0.5)))
        ny = int(np.ceil(np.amax(h / ovy + 0.5)))

        # account for a maximum error of 1 pix in the initial star coordinates:
        nx += 2
        ny += 2

        # convert to oversampled pixels:
        nx = int(np.ceil(np.amax(nx * ovx + 0.5)))
        ny = int(np.ceil(np.amax(ny * ovy + 0.5)))

        # we prefer odd sized images
        nx += 1 - nx % 2
        ny += 1 - ny % 2
        shape = (ny, nx)

    else:
        (ny, nx) = _parse_tuple_pars(shape, name='shape', dtype=int)

    # center of the output grid:
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0

    with warnings.catch_warnings(record=False):
        warnings.simplefilter("ignore", NonNormalizable)

        # filter out parameters set by us:
        kwargs = copy.deepcopy(kwargs)
        for kw in ['data', 'origin', 'normalize', 'pixel_scale']:
            if kw in kwargs:
                del kwargs[kw]

        data = np.zeros((ny, nx), dtype=np.float)
        psf = psf_cls(
            data=data, origin=(cx, cy), normalize=True,
            pixel_scale=(pscale_x, pscale_y), **kwargs
        )

    return psf
