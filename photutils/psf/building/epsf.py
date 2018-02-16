# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools to build an ePSF.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import SigmaClip


__all__ = ['EPSFFitter', 'EPSFBuilder']


class EPSFFitter(object):
    def __init__(self, psf, psf_fit_box=5, fitter=LevMarLSQFitter(),
                 residuals=False, **kwargs):
        self.psf = psf
        self.psf_fit_box = psf_fit_box
        self.fitter = fitter
        self.residuals = residuals
        self.fitter_kwargs = kwargs

    def __call__(self, data, psf, star_table):
        return self.fit_psf(data, psf, star_table)

    def fit_psf(self, data, psf, star_table):
        pass


class EPSFBuilder(object):
    def __init__(self, peak_fit_box=5, peak_search_box='fitbox',
                 recenter_accuracy=1.0e-4, recenter_max_iters=1000,
                 ignore_badfit_stars=True, stat='median',
                 sigma_clip=SigmaClip(sigma=3., iters=10),
                 smoothing_kernel='quar', fitter=EPSFFitter(residuals=True),
                 max_iters=50, accuracy=1e-4, epsf=None):

        self.peak_fit_box = peak_fit_box
        self.peak_search_box = peak_search_box
        self.recenter_accuracy = recenter_accuracy
        self.recenter_max_iters = recenter_max_iters
        self.ignore_badfit_stars = ignore_badfit_stars
        self.stat = stat
        self.sigma_clip = sigma_clip
        self.smoothing_kernel = smoothing_kernel
        self.fitter = fitter

        max_iters = int(max_iters)
        if max_iters < 0:
            raise ValueErro('max_iters must be non-negative.')
        self.max_iters = max_iters

        if accuracy <= 0.0:
            raise ValueError('accuracy must be a positive number.')
        self.accuracy = accuracy

        self.epsf = epsf

    def __call__(self, data, star_table):
        return self.build_psf(data, star_table)

    def _build_psf_step(self, psf=None):
        if len(stars) < 1:
            raise ValueError("'stars' must be a list containing at least one "
                            "'Star' object.")

        if psf is None:
            psf = init_psf(stars)
        elif isinstance(psf, type):
            psf = init_psf(stars, psf_cls=psf)
        else:
            psf = copy.deepcopy(psf)

        recenter_accuracy = float(recenter_accuracy)
        if recenter_accuracy <= 0.0:
            raise ValueError("Re-center accuracy must be a strictly positive "
                            "number.")

        recenter_nmax = int(recenter_nmax)
        if recenter_nmax < 0:
            raise ValueError("Maximum number of re-recntering iterations must be "
                            "a non-negative integer number.")

        cx, cy = psf.origin
        ny, nx = psf.shape
        pscale_x, pscale_y = psf.pixel_scale

        # get all stars including linked stars as a flat list:
        all_stars = []
        for s in stars:
            all_stars += s.get_linked_list()

        # allocate "accumulator" array (to store transformed PSFs):
        apsf = [[[] for k in range(nx)] for k in range(ny)]

        norm_psf_data = psf.normalized_data

        for s in all_stars:
            if s.ignore:
                continue

            if ignore_badfit_stars and s.fit_error_status is not None and \
            s.fit_error_status > 0:
                continue

            pixlist = s.centered_plist(normalized=True)

            # evaluate previous PSF model at star pixel location in the PSF grid:
            ovx = s.pixel_scale[0] / pscale_x
            ovy = s.pixel_scale[1] / pscale_y
            x = ovx * (pixlist[:, 0])
            y = ovy * (pixlist[:, 1])
            old_model_vals = psf.evaluate(x=x, y=y, flux=1.0, x_0=0.0, y_0=0.0)

            # find integer location of star pixels in the PSF grid
            # and compute residuals
            ix = py2round(x + cx).astype(np.int)
            iy = py2round(y + cy).astype(np.int)
            pv = pixlist[:, 2] / (ovx * ovy) - old_model_vals
            m = np.logical_and(
                np.logical_and(ix >= 0, ix < nx),
                np.logical_and(iy >= 0, iy < ny)
            )

            # add all pixel values to the corresponding accumulator:
            for i, j, v in zip(ix[m], iy[m], pv[m]):
                apsf[j][i].append(v)

        psfdata = np.empty((ny, nx), dtype=np.float)
        psfdata.fill(np.nan)

        for i in range(nx):
            for j in range(ny):
                psfdata[j, i] = _pixstat(
                    apsf[j][i], stat=stat, nclip=nclip, lsig=lsig, usig=usig,
                    default=np.nan
                )

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
        psfdata = _smoothPSF(psfdata, ker)

        shift_x = 0
        shift_y = 0
        peak_eps2 = recenter_accuracy**2
        eps2_prev = None
        y, x = np.indices(psfdata.shape, dtype=np.float)
        ePSF = psf.make_similar_from_data(psfdata)
        ePSF.fill_value = 0.0

        for iteration in range(recenter_nmax):
            # find peak location:
            peak_x, peak_y = find_peak(
                psfdata, xmax=cx, ymax=cy, peak_fit_box=peak_fit_box,
                peak_search_box=peak_search_box, mask=None
            )

            dx = cx - peak_x
            dy = cy - peak_y

            eps2 = dx**2 + dy**2
            if (eps2_prev is not None and eps2 > eps2_prev) or eps2 < peak_eps2:
                break
            eps2_prev = eps2

            shift_x += dx
            shift_y += dy

            # Resample PSF data to a shifted grid such that the pick of the PSF is
            # at expected position.
            psfdata = ePSF.evaluate(x=x, y=y, flux=1.0,
                                    x_0=shift_x + cx, y_0=shift_y + cy)

        # apply final shifts and fill in any missing data:
        if shift_x != 0.0 or shift_y != 0.0:
            ePSF.fill_value = np.nan
            psfdata = ePSF.evaluate(x=x, y=y, flux=1.0,
                                    x_0=shift_x + cx, y_0=shift_y + cy)

            # fill in the "holes" (=np.nan) using 0 (no contribution to the flux):
            mask = np.isfinite(psfdata)
            psfdata[np.logical_not(mask)] = 0.0

        norm = np.abs(np.sum(psfdata, dtype=np.float64))
        psfdata /= norm

        # Create ePSF model and return:
        ePSF = psf.make_similar_from_data(psfdata)

        return ePSF


    def build_psf(self, data, star_table):
        """
        Iteratively build the psf.
        """

        self._extract_stars(data, star_table, common_catalog=None,
                            extract_size=11, recenter=False,
                            peak_fit_box=5, peak_search_box='fitbox',
                            catmap={'x': 'x', 'y': 'y', 'lon': 'lon',
                                    'lat': 'lat', 'weight': 'weight',
                                    'id': 'id'}, cat_name_kwd='name',
                            image_name_kwd='name')


        # get all stars including linked stars as a flat list:
        all_stars = []
        for s in stars:
            all_stars += s.get_linked_list()

        # create an array of star centers:
        prev_centers = np.asarray([s.center for s in stars], dtype=np.float)

        # initialize array for detection of scillatory behavior:
        oscillatory = np.zeros((len(all_stars), ), dtype=np.bool)
        prev_failed = np.zeros((len(all_stars), ), dtype=np.bool)

        niter = -1
        eps2 = 2.0 * acc2
        dxy = np.zeros((len(all_stars), 2), dtype=np.float)


        while niter < nmax and np.amax(eps2) >= acc2 and not np.all(oscillatory):
            niter += 1

            # improved PSF:
            psf = build_psf(
                stars=stars,
                psf=psf,
                peak_fit_box=peak_fit_box,
                peak_search_box=peak_search_box,
                recenter_accuracy=recenter_accuracy,
                recenter_nmax=recenter_nmax,
                ignore_badfit_stars=ignore_badfit_stars,
                stat=stat,
                nclip=nclip,
                lsig=lsig,
                usig=usig,
                ker=ker
            )

            # improved fit of the PSF to stars:
            stars = fit_stars(
                stars=stars,
                psf=psf,
                psf_fit_box=psf_fit_box,
                fitter=fitter,
                fitter_kwargs=fitter_kwargs,
                residuals=False
            )

            # get all stars including linked stars as a flat list:
            all_stars = []
            for s in stars:
                all_stars += s.get_linked_list()

            # create an array of star centers at this iteration:
            centers = np.asarray([s.center for s in stars], dtype=np.float)

            # detect oscillatory behavior
            failed = np.array([s.fit_error_status > 0 for s in stars],
                            dtype=np.bool)
            if niter > 2:  # allow a few iterations at the beginning
                oscillatory = np.logical_and(prev_failed, np.logical_not(failed))
                for s, osc in zip(stars, oscillatory):
                    s.ignore = bool(osc)
                prev_failed = failed

            # check termination criterion:
            good_mask = np.logical_not(np.logical_or(failed, oscillatory))
            dxy = centers - prev_centers
            mdxy = dxy[good_mask]
            eps2 = np.sum(mdxy * mdxy, axis=1, dtype=np.float64)
            prev_centers = centers

        # compute residuals:
        if residuals:
            res = compute_residuals(psf, all_stars)
        else:
            res = len(all_stars) * [None]

        for s, r in zip(all_stars, res):
            s.fit_residual = r

        # assign coordinate residuals of the iterative process:
        for s, (dx, dy) in zip(all_stars, dxy):
            s.iter_fit_eps = (float(dx), float(dy))

        return (psf, stars, niter)
