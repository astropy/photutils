# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define tools for integrating over pixels.
"""

import math

import numpy as np

__all__ = ['BILINEAR', 'INTEGRATORS', 'MEAN', 'MEDIAN', 'NEAREST_NEIGHBOR']


# integration modes
NEAREST_NEIGHBOR = 'nearest_neighbor'
BILINEAR = 'bilinear'
MEAN = 'mean'
MEDIAN = 'median'


class _Integrator:
    """
    Base class that supports different kinds of pixel integration
    methods.

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
         The image array.

    geometry : `~photutils.isophote.EllipseGeometry` instance
        Object that encapsulates geometry information about current
        ellipse.

    angles : list
        Output list; contains the angle values along the elliptical
        path.

    radii : list
        Output list; contains the radius values along the elliptical
        path.

    intensities : list
        Output list; contains the extracted intensity values along the
        elliptical path.
    """

    def __init__(self, image, geometry, angles, radii, intensities):
        self._image = image
        self._geometry = geometry

        self._angles = angles
        self._radii = radii
        self._intensities = intensities

        # for bounds checking
        self._i_range = range(self._image.shape[1] - 1)
        self._j_range = range(self._image.shape[0] - 1)

    def integrate(self, radius, phi):
        """
        The three input lists (angles, radii, intensities) are appended
        with one sample point taken from the image by a chosen
        integration method.

        Subclasses should implement the actual integration method.

        Parameters
        ----------
        radius : float
            The length of the radius vector in pixels.

        phi : float
            The polar angle of radius vector.
        """
        raise NotImplementedError

    def _reset(self):
        """
        Reset the lists containing results.

        This method is for internal use and shouldn't be used by
        external callers.
        """
        self._angles = []
        self._radii = []
        self._intensities = []

    def _store_results(self, phi, radius, sample):
        self._angles.append(phi)
        self._radii.append(radius)
        self._intensities.append(sample)

    def get_polar_angle_step(self):
        """
        Return the polar angle step used to walk over the elliptical
        path.

        The polar angle step is defined by the actual integrator
        subclass.

        Returns
        -------
        result : float
            The polar angle step.
        """
        raise NotImplementedError

    def get_sector_area(self):
        """
        Return the area of elliptical sectors where the integration
        takes place.

        This area is defined and managed by the actual integrator
        subclass. Depending on the integrator, the area may be a
        fixed constant, or may change along the elliptical path, so
        it's up to the caller to use this information in a correct way.

        Returns
        -------
        result : float
            The sector area.
        """
        raise NotImplementedError

    def is_area(self):
        """
        Return the type of the integrator.

        An area integrator gets its value from operating over a (generally
        variable) number of pixels that define a finite area that lies
        around the elliptical path, at a certain point on the image defined
        by a polar angle and radius values. A pixel integrator, by contrast,
        integrates over a fixed and normally small area related to a single
        pixel on the image. An example is the bilinear integrator, which
        integrates over a small, fixed, 5-pixel area. This method checks if
        the integrator is of the first type or not.

        Returns
        -------
        result : boolean
            True if this is an area integrator, False otherwise.
        """
        raise NotImplementedError


class _NearestNeighborIntegrator(_Integrator):
    def integrate(self, radius, phi):
        self._r = radius

        # Get image coordinates of (radius, phi) pixel
        i = int(radius * math.cos(phi + self._geometry.pa)
                + self._geometry.x0)
        j = int(radius * math.sin(phi + self._geometry.pa)
                + self._geometry.y0)

        # ignore data point if outside image boundaries
        if (i in self._i_range) and (j in self._j_range):
            sample = self._image[j][i]

            if sample is not np.ma.masked:
                self._store_results(phi, radius, sample)

    def get_polar_angle_step(self):
        return 1.0 / self._r

    def get_sector_area(self):
        return 1.0

    def is_area(self):
        return False


class _BiLinearIntegrator(_Integrator):
    def integrate(self, radius, phi):
        self._r = radius

        # Get image coordinates of (radius, phi) pixel
        x_ = radius * math.cos(phi + self._geometry.pa) + self._geometry.x0
        y_ = radius * math.sin(phi + self._geometry.pa) + self._geometry.y0
        i = int(x_)
        j = int(y_)
        fx = x_ - i
        fy = y_ - j

        # ignore data point if outside image boundaries
        if (i in self._i_range) and (j in self._j_range):
            # in the future, will need to handle masked pixels here
            qx = 1.0 - fx
            qy = 1.0 - fy

            if (self._image[j][i] is not np.ma.masked
                    and self._image[j + 1][i] is not np.ma.masked
                    and self._image[j][i + 1] is not np.ma.masked
                    and self._image[j + 1][i + 1] is not np.ma.masked):

                sample = (self._image[j][i] * qx * qy
                          + self._image[j + 1][i] * qx * fy
                          + self._image[j][i + 1] * fx * qy
                          + self._image[j + 1][i + 1] * fy * fx)

                self._store_results(phi, radius, sample)

    def get_polar_angle_step(self):
        return 1.0 / self._r

    def get_sector_area(self):
        return 2.0

    def is_area(self):
        return False


class _AreaIntegrator(_Integrator):
    def __init__(self, image, geometry, angles, radii, intensities):
        super().__init__(image, geometry, angles, radii, intensities)

        # build auxiliary bilinear integrator to be used when
        # sector areas contain a too small number of valid pixels.
        self._bilinear_integrator = INTEGRATORS[BILINEAR](image, geometry,
                                                          angles, radii,
                                                          intensities)

    def integrate(self, radius, phi):
        self._phi = phi

        # Get image coordinates of the four vertices of the elliptical sector.
        vertex_x, vertex_y = self._geometry.initialize_sector_geometry(phi)

        self._sector_area = self._geometry.sector_area

        # step in polar angle to be used by caller next time
        # when updating the current polar angle `phi` to point
        # to the next sector.
        self._phistep = self._geometry.sector_angular_width

        # define rectangular image area that encompasses the elliptical
        # sector. We have to account for rounding of pixel indices.
        i1 = int(min(vertex_x)) - 1
        j1 = int(min(vertex_y)) - 1
        i2 = int(max(vertex_x)) + 1
        j2 = int(max(vertex_y)) + 1

        # polar angle limits for this sector
        phi1, phi2 = self._geometry.polar_angle_sector_limits()

        # ignore data point if the elliptical sector lies
        # partially, or totally, outside image boundaries
        if (i1 in self._i_range) and (j1 in self._j_range) and \
           (i2 in self._i_range) and (j2 in self._j_range):

            # Scan rectangular image area, compute sample value.
            npix = 0
            accumulator = self.initialize_accumulator()
            for j in range(j1, j2):
                for i in range(i1, i2):
                    # Check if polar coordinates of each pixel
                    # put it inside elliptical sector.
                    rp, phip = self._geometry.to_polar(i, j)

                    # check if inside angular limits
                    if phip < phi2 and phip >= phi1:

                        # check if radius is inside bounding ellipses
                        sma1, sma2 = self._geometry.bounding_ellipses()
                        aux = ((1.0 - self._geometry.eps)
                               / math.sqrt(((1.0 - self._geometry.eps)
                                            * math.cos(phip))**2
                                           + (math.sin(phip))**2))

                        r1 = sma1 * aux
                        r2 = sma2 * aux

                        if rp < r2 and rp >= r1:
                            # update accumulator with pixel value
                            pix_value = self._image[j][i]
                            if pix_value is not np.ma.masked:
                                accumulator, npix = self.accumulate(
                                    pix_value, accumulator)

            # If 6 or less pixels were sampled, get the bilinear
            # interpolated value instead.
            if npix in range(7):
                # must reset integrator to remove older samples.
                self._bilinear_integrator._reset()
                self._bilinear_integrator.integrate(radius, phi)
                # because it was reset, current value is the only one stored
                # internally in the bilinear integrator instance. Move it
                # from the internal integrator to this instance.
                if len(self._bilinear_integrator._intensities) > 0:
                    sample_value = self._bilinear_integrator._intensities[0]
                    self._store_results(phi, radius, sample_value)

            elif npix > 6:
                sample_value = self.compute_sample_value(accumulator)
                self._store_results(phi, radius, sample_value)

    def get_polar_angle_step(self):
        _, phi2 = self._geometry.polar_angle_sector_limits()
        return self._geometry.sector_angular_width / 2.0 + phi2 - self._phi

    def get_sector_area(self):
        return self._sector_area

    def is_area(self):
        return True

    def initialize_accumulator(self):
        raise NotImplementedError

    def accumulate(self, pixel_value, accumulator):
        raise NotImplementedError

    def compute_sample_value(self, accumulator):
        raise NotImplementedError


class _MeanIntegrator(_AreaIntegrator):
    def initialize_accumulator(self):
        accumulator = 0.0
        self._npix = 0
        return accumulator

    def accumulate(self, pixel_value, accumulator):
        accumulator += pixel_value
        self._npix += 1
        return accumulator, self._npix

    def compute_sample_value(self, accumulator):
        return accumulator / self._npix


class _MedianIntegrator(_AreaIntegrator):
    def initialize_accumulator(self):
        accumulator = []
        self._npix = 0
        return accumulator

    def accumulate(self, pixel_value, accumulator):
        accumulator.append(pixel_value)
        self._npix += 1
        return accumulator, self._npix

    def compute_sample_value(self, accumulator):
        accumulator.sort()
        return accumulator[int(self._npix / 2)]


# Specific integrator subclasses can be instantiated from here.
INTEGRATORS = {
    NEAREST_NEIGHBOR: _NearestNeighborIntegrator,
    BILINEAR: _BiLinearIntegrator,
    MEAN: _MeanIntegrator,
    MEDIAN: _MedianIntegrator,
}
