# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for generating radial profiles and curves of
growth.
"""

import math
import warnings

import numpy as np
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning

from photutils.utils._quantity_helpers import process_quantities

__all__ = ['ProfileBase', 'CurveOfGrowth', 'RadialProfile']


class ProfileBase:
    def __init__(self, data, xycen, min_radius, max_radius, radius_step, *,
                 error=None, mask=None, method='exact', subpixels=5):

        (data, error), _ = process_quantities((data, error), ('data', 'error'))
        self.data = data
        self.xycen = xycen

        if min_radius < 0 or max_radius < 0:
            raise ValueError('min_radius and max_radius must be >= 0')
        if min_radius >= max_radius:
            raise ValueError('max_radius must be greater than min_radius')
        if radius_step <= 0:
            raise ValueError('radius_step must be > 0')
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.radius_step = radius_step

        if error is not None and error.shape != data.shape:
            raise ValueError('error must have the same same as data')
        self.error = error
        if mask is not None and mask.shape != data.shape:
            raise ValueError('mask must have the same same as data')
        self.mask = mask

        self.method = method
        self.subpixels = subpixels

        self._nradii = int(math.ceil(self.max_radius - self.min_radius)
                           / self.radius_step) + 1  # inclusive
        self._circular_radii = np.linspace(min_radius, max_radius,
                                           self._nradii)
        self._annulus_radii = (self._circular_radii[1:]
                               + self._circular_radii[:-1]) / 2

    @lazyproperty
    def _annulus_apertures(self):
        from photutils.aperture import CircularAnnulus, CircularAperture

        # circular annulus apertures (circular aperture if min_radius = 0)
        apertures = []
        for i in range(self._nradii - 1):
            try:
                aperture = CircularAnnulus(self.xycen, self._nradii[i],
                                           self._circular_radii[i + 1])
            except ValueError:  # zero radius
                aperture = CircularAperture(self.xycen,
                                            self._circular_radii[i + 1])
            apertures.append(aperture)

        return apertures

    @lazyproperty
    def _circular_apertures(self):
        from photutils.aperture import CircularAperture

        # only circular apertures
        apertures = []
        for radius in self._circular_radii:
            if radius == 0.0:
                aper = None
            else:
                aper = CircularAperture(self.xycen, radius)
            apertures.append(aper)
        return apertures

    @lazyproperty
    def _photometry(self):
        fluxes = []
        fluxerrs = []
        areas = []
        for aperture in self._circular_apertures:
            if aperture is None:
                flux, fluxerr = [0.0], [0.0]
                area = 0.0
            else:
                flux, fluxerr = aperture.do_photometry(
                    self.data, error=self.error, mask=self.mask,
                    method=self.method, subpixels=self.subpixels)
                area = aperture.area_overlap(self.data, mask=self.mask,
                                             method=self.method,
                                             subpixels=self.subpixels)
            fluxes.append(flux[0])
            if self.error is not None:
                fluxerrs.append(fluxerr[0])
            areas.append(area)

        fluxes = np.array(fluxes)
        fluxerrs = np.array(fluxerrs)
        areas = np.array(areas)

        return fluxes, fluxerrs, areas

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        ax.plot(self.radius, self.profile, **kwargs)
        ax.set_xlabel('Radius (pixels)')
        ax.set_ylabel('Profile')

        return ax


class CurveOfGrowth(ProfileBase):
    @lazyproperty
    def _aperphot(self):
        # remove first element of flux, fluxerr, and area arrays
        aperphot = self._photometry
        if self._circular_apertures[0] is None:
            aperphot = tuple(phot[1:] for phot in aperphot)
        return aperphot

    @lazyproperty
    def radius(self):
        radius = self._circular_radii
        if self._circular_apertures[0] is None:
            radius = radius[1:]
        return radius

    @lazyproperty
    def profile(self):
        return self._aperphot[0]

    @lazyproperty
    def profile_err(self):
        return self._aperphot[1]

    @lazyproperty
    def area(self):
        return self._aperphot[2]

    def normalize(self, method='max'):
        if method == 'max':
            normalization = self.profile.max()
        elif method == 'sum':
            normalization = self.profile.sum()
        else:
            raise ValueError('invalid method, must be "peak" or "integral"')

        if normalization == 0:
            warnings.warn('The profile cannot be normalized because the '
                          'max or sum is zero.', AstropyUserWarning)
        else:
            self.__dict__['profile'] = self.profile / normalization
            self.__dict__['profile_err'] = self.profile_err / normalization


class RadialProfile(ProfileBase):
    @lazyproperty
    def radius(self):
        return self._annulus_radii

    @lazyproperty
    def _flux(self):
        return np.diff(self._photometry[0])

    @lazyproperty
    def _fluxerr(self):
        return np.sqrt(np.diff(self._photometry[1] ** 2))

    @lazyproperty
    def area(self):
        return np.diff(self._photometry[2])

    @lazyproperty
    def profile(self):
        return self._flux / self.area

    @lazyproperty
    def profile_err(self):
        return self._fluxerr / self.area
