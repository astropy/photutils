# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for integrating over pixels.
"""

import abc

import numpy as np

__all__ = ['BILINEAR', 'INTEGRATORS', 'MEAN', 'MEDIAN', 'NEAREST_NEIGHBOR']


# Integration modes
NEAREST_NEIGHBOR = 'nearest_neighbor'
BILINEAR = 'bilinear'
MEAN = 'mean'
MEDIAN = 'median'


class _Integrator(abc.ABC):
    """
    Base class that supports different kinds of pixel integration
    methods.

    Parameters
    ----------
    image : 2D `~numpy.ndarray` or `~numpy.ma.MaskedArray`
        The image array. Masked pixels are not used.

    geometry : `~photutils.isophote.EllipseGeometry` instance
        Object that encapsulates geometry information about current
        ellipse.

    angles : list
        Output list containing the angle values along the elliptical
        path.

    radii : list
        Output list containing the radius values along the elliptical
        path.

    intensities : list
        Output list containing the extracted intensity values along the
        elliptical path.
    """

    def __init__(self, image, geometry, angles, radii, intensities):
        self._image = image
        self._geometry = geometry

        # Split a masked image into its data and mask arrays once,
        # because indexing a MaskedArray pixel by pixel is slow. The
        # mask is None if no pixels can be masked.
        self._data = np.ma.getdata(image)
        mask = np.ma.getmask(image)
        self._mask = None if mask is np.ma.nomask else mask

        self._angles = angles
        self._radii = radii
        self._intensities = intensities

        # For bounds checking
        self._i_range = range(self._image.shape[1] - 1)
        self._j_range = range(self._image.shape[0] - 1)

    @abc.abstractmethod
    def integrate(self, radius, phi):
        """
        Sample the image at one point and append the polar angle,
        radius, and sampled value to the three lists.

        Nothing is appended if the point cannot be sampled.

        Subclasses should implement the actual integration method.

        Parameters
        ----------
        radius : float
            The length of the radius vector in pixels.

        phi : float
            The polar angle (radians) of the radius vector.
        """

    def _store_results(self, phi, radius, sample):
        self._angles.append(phi)
        self._radii.append(radius)
        self._intensities.append(sample)

    @abc.abstractmethod
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

    @abc.abstractmethod
    def is_area(self):
        """
        Return the type of the integrator.

        An area integrator gets its value from operating over a (generally
        variable) number of pixels that define a finite area that lies
        around the elliptical path, at a certain point on the image defined
        by a polar angle and radius values. A pixel integrator, by contrast,
        integrates over a fixed and normally small area related to a single
        pixel on the image. An example is the bilinear integrator, which
        interpolates over a small, fixed, 2x2-pixel area. This method
        checks if the integrator is of the first type or not.

        Returns
        -------
        result : bool
            True if this is an area integrator, False otherwise.
        """


class _PixelIntegrator(_Integrator):
    """
    Base class for the integrators that sample single pixel positions.

    The path along the ellipse is sampled at once by the
    ``integrate_path`` method. The ``integrate`` method samples a single
    point.
    """

    def integrate(self, radius, phi):
        angles, radii, values = self.integrate_path(np.array([radius]),
                                                    np.array([phi]))
        if len(values) > 0:
            self._store_results(angles[0], radii[0], values[0])

    @abc.abstractmethod
    def integrate_path(self, radii, phis):
        """
        Sample the image at the input polar coordinates.

        Points outside of the image, or whose sample includes masked or
        nonfinite pixels, are skipped. The bilinear integrator also
        skips points in the outer half-pixel border of the image, where
        the 2x2 block of pixels needed for the interpolation is not
        complete.

        Parameters
        ----------
        radii : 1D `~numpy.ndarray`
            The polar radii (pixels).

        phis : 1D `~numpy.ndarray`
            The polar angles (radians).

        Returns
        -------
        angles, radii, values : 1D `~numpy.ndarray`
            The polar angles, polar radii, and sampled values of the
            valid points.
        """

    def is_area(self):
        return False


class _NearestNeighborIntegrator(_PixelIntegrator):
    def integrate_path(self, radii, phis):
        ny, nx = self._data.shape

        # Get image coordinates of the (radius, phi) points. Pixel
        # centers are at integer coordinates, so the nearest pixel is
        # found by rounding.
        angle = phis + self._geometry.pa
        x = np.floor(radii * np.cos(angle) + self._geometry.x0 + 0.5)
        y = np.floor(radii * np.sin(angle) + self._geometry.y0 + 0.5)

        # Ignore data points outside image boundaries. The floating
        # point coordinates are checked before they are converted to
        # integers because nonfinite and very large coordinates do not
        # have a defined integer value.
        good = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
        i = x[good].astype(int)
        j = y[good].astype(int)
        radii = radii[good]
        phis = phis[good]

        # Skip masked and nonfinite pixels.
        values = self._data[j, i]
        good = np.isfinite(values)
        if self._mask is not None:
            good &= ~self._mask[j, i]

        return phis[good], radii[good], values[good]

    def get_sector_area(self):
        return 1.0


class _BiLinearIntegrator(_PixelIntegrator):
    def integrate_path(self, radii, phis):
        ny, nx = self._data.shape

        # Get image coordinates of the (radius, phi) points.
        angle = phis + self._geometry.pa
        x = radii * np.cos(angle) + self._geometry.x0
        y = radii * np.sin(angle) + self._geometry.y0

        # Use floor so that points before the first pixel center are
        # skipped instead of extrapolated.
        x_floor = np.floor(x)
        y_floor = np.floor(y)

        # Ignore data points outside image boundaries. The floating
        # point coordinates are checked before they are converted to
        # integers because nonfinite and very large coordinates do not
        # have a defined integer value.
        good = ((x_floor >= 0) & (x_floor < nx - 1)
                & (y_floor >= 0) & (y_floor < ny - 1))
        i = x_floor[good].astype(int)
        j = y_floor[good].astype(int)
        fx = x[good] - i
        fy = y[good] - j
        radii = radii[good]
        phis = phis[good]

        # Nonfinite pixels can give invalid operations (e.g., inf - inf).
        # The resulting nonfinite samples are skipped below.
        qx = 1.0 - fx
        qy = 1.0 - fy
        data = self._data
        with np.errstate(invalid='ignore'):
            values = (data[j, i] * qx * qy
                      + data[j + 1, i] * qx * fy
                      + data[j, i + 1] * fx * qy
                      + data[j + 1, i + 1] * fy * fx)

        # Skip samples that include masked or nonfinite pixels.
        good = np.isfinite(values)
        if self._mask is not None:
            mask = self._mask
            good &= ~(mask[j, i] | mask[j + 1, i] | mask[j, i + 1]
                      | mask[j + 1, i + 1])

        return phis[good], radii[good], values[good]

    def get_sector_area(self):
        return 2.0


class _AreaIntegrator(_Integrator):
    def __init__(self, image, geometry, angles, radii, intensities):
        super().__init__(image, geometry, angles, radii, intensities)

        # Build auxiliary bilinear integrator to be used when
        # sector areas contain a too small number of valid pixels.
        self._bilinear_integrator = INTEGRATORS[BILINEAR](image, geometry,
                                                          angles, radii,
                                                          intensities)

    def integrate(self, radius, phi):
        self._phi = phi

        # Get image coordinates of the four vertices of the elliptical sector.
        vertex_x, vertex_y = self._geometry.initialize_sector_geometry(phi)

        self._sector_area = self._geometry.sector_area

        # Step in polar angle to be used by caller next time
        # when updating the current polar angle `phi` to point
        # to the next sector.
        self._phistep = self._geometry.sector_angular_width

        # Define rectangular image area that encompasses the elliptical
        # sector. We have to account for rounding of pixel indices.
        i1 = int(min(vertex_x)) - 1
        j1 = int(min(vertex_y)) - 1
        i2 = int(max(vertex_x)) + 1
        j2 = int(max(vertex_y)) + 1

        # Polar angle limits for this sector
        phi1, phi2 = self._geometry.polar_angle_sector_limits()

        # Ignore data point if the elliptical sector lies
        # partially, or totally, outside image boundaries.
        if ((i1 in self._i_range) and (j1 in self._j_range)
                and (i2 in self._i_range) and (j2 in self._j_range)):
            # Scan rectangular image area. Find the pixels whose polar
            # coordinates put them inside the elliptical sector, i.e.,
            # inside the angular limits and between the bounding
            # ellipses.
            geometry = self._geometry
            dx = np.arange(i1, i2) - geometry.x0
            dy = (np.arange(j1, j2) - geometry.y0)[:, np.newaxis]
            rp = np.hypot(dx, dy)
            phip = np.mod(np.arctan2(dy, dx) - geometry.pa, 2.0 * np.pi)
            sma1, sma2 = geometry.bounding_ellipses()
            eps_ = 1.0 - geometry.eps
            aux = eps_ / np.sqrt((eps_ * np.cos(phip))**2
                                 + np.sin(phip)**2)
            # A sector can extend past a polar angle of 2 pi, so the
            # pixel angles are measured from the start of the sector.
            dphi = np.mod(phip - phi1, 2.0 * np.pi)
            inside = ((dphi < phi2 - phi1)
                      & (rp < sma2 * aux) & (rp >= sma1 * aux))

            # Skip masked and nonfinite pixels.
            values = self._data[j1:j2, i1:i2][inside]
            good = np.isfinite(values)
            if self._mask is not None:
                good &= ~self._mask[j1:j2, i1:i2][inside]
            values = values[good]

            # If 6 or less pixels were sampled, get the bilinear
            # interpolated value instead.
            if values.size <= 6:
                _, _, values = self._bilinear_integrator.integrate_path(
                    np.array([radius]), np.array([phi]))
                if values.size > 0:
                    self._store_results(phi, radius, values[0])
            else:
                sample_value = self.compute_sample_value(values)
                self._store_results(phi, radius, sample_value)

    def get_polar_angle_step(self):
        """
        Return the polar angle step used to walk over the elliptical
        path.

        Returns
        -------
        result : float
            The polar angle step.
        """
        _, phi2 = self._geometry.polar_angle_sector_limits()
        return self._geometry.sector_angular_width / 2.0 + phi2 - self._phi

    def get_sector_area(self):
        return self._sector_area

    def is_area(self):
        return True

    @abc.abstractmethod
    def compute_sample_value(self, values):
        """
        Compute the sample value from the values of the pixels inside
        the elliptical sector.

        Parameters
        ----------
        values : 1D `~numpy.ndarray`
            The pixel values, in row-major order.

        Returns
        -------
        result : float
            The sample value.
        """


class _MeanIntegrator(_AreaIntegrator):
    def compute_sample_value(self, values):
        return values.mean(dtype=float)


class _MedianIntegrator(_AreaIntegrator):
    def compute_sample_value(self, values):
        # As in the STSDAS ellipse task, the median of an even number of
        # values is the average of the two middle values.
        return np.median(values)


# Specific integrator subclasses can be instantiated from here.
INTEGRATORS = {
    NEAREST_NEIGHBOR: _NearestNeighborIntegrator,
    BILINEAR: _BiLinearIntegrator,
    MEAN: _MeanIntegrator,
    MEDIAN: _MedianIntegrator,
}
