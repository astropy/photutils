# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Window (tapering) functions for matching PSFs using Fourier methods.
"""

import warnings

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning

__all__ = [
    'CosineBellWindow',
    'HanningWindow',
    'SplitCosineBellWindow',
    'TopHatWindow',
    'TukeyWindow',
]


def _distance_grid(shape):
    """
    Return an array where each value is the Euclidean distance from the
    array center.

    Parameters
    ----------
    shape : tuple of int
        The size of the output array along each axis. Must have only 2
        elements. To have a well defined array center, the size along
        each axis should be an odd integer, but this is not enforced.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        An array containing the Euclidean radial distances from the
        array center.
    """
    if len(shape) != 2:
        msg = 'shape must have only 2 elements'
        raise ValueError(msg)

    y_cen, x_cen = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    y_vals, x_vals = np.ogrid[:shape[0], :shape[1]]

    return np.hypot(x_vals - x_cen, y_vals - y_cen)


class SplitCosineBellWindow:
    """
    Class to define a 2D split cosine bell taper function.

    This is the base class for window functions, providing full control
    over both the inner flat region (``beta``) and the taper width
    (``alpha``). The window equals 1.0 in the inner region, smoothly
    transitions to 0.0 using a cosine taper, and remains 0.0 outside.

    This window is useful when you need precise control over both the
    preserved central region and the taper characteristics.

    Parameters
    ----------
    alpha : float, optional
        The percentage of array values that are tapered. ``alpha`` must
        be between 0.0 and 1.0, inclusive.

    beta : float, optional
        The inner diameter as a fraction of the array size beyond
        which the taper begins. ``beta`` must be between 0.0 and 1.0,
        inclusive.

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils.psf_matching import SplitCosineBellWindow

        taper = SplitCosineBellWindow(alpha=0.4, beta=0.3)
        data = taper((101, 101))
        plt.imshow(data, origin='lower')
        plt.colorbar()

    A 1D cut across the image center:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils.psf_matching import SplitCosineBellWindow

        taper = SplitCosineBellWindow(alpha=0.4, beta=0.3)
        data = taper((101, 101))
        plt.plot(data[50, :])
    """

    def __init__(self, alpha, beta):
        if not (0.0 <= alpha <= 1.0):
            msg = ('alpha must be between 0.0 and 1.0, inclusive. '
                   f'Got: {alpha}')
            raise ValueError(msg)
        if not (0.0 <= beta <= 1.0):
            msg = ('beta must be between 0.0 and 1.0, inclusive. '
                   f'Got: {beta}')
            raise ValueError(msg)

        if alpha + beta > 1.0:
            msg = ('alpha + beta > 1.0; the taper region will be '
                   'clipped to the array boundary.')
            warnings.warn(msg, AstropyUserWarning)

        self.alpha = float(alpha)
        self.beta = float(beta)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'alpha={self.alpha!r}, beta={self.beta!r})')

    def __str__(self):
        return self.__repr__()

    def __call__(self, shape):
        """
        Generate the window function for the given shape.

        Parameters
        ----------
        shape : tuple of int
            The size of the output array along each axis.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The window function as a 2D array.
        """
        dist = _distance_grid(shape)

        # Define geometry based on the smallest shape dimension
        max_r = (min(shape) - 1.0) / 2.0
        r_inner = self.beta * max_r
        taper_width = self.alpha * max_r
        r_outer = r_inner + taper_width

        if taper_width > 0:
            r = dist - r_inner
            result = 0.5 * (1.0 + np.cos(np.pi * r / taper_width))
        else:
            result = np.ones(shape, dtype=float)

        result[dist < r_inner] = 1.0
        result[dist > r_outer] = 0.0

        return result


class HanningWindow(SplitCosineBellWindow):
    """
    Class to define a 2D `Hanning (or Hann) window
    <https://en.wikipedia.org/wiki/Hann_function>`_ function.

    The Hann window is a taper formed by using a raised cosine with ends
    that touch zero. The taper begins at the center and smoothly
    decreases to zero at the edges. This window equals 1.0 only at the
    exact center point.

    This is a classic general-purpose window function widely used in
    signal processing. It provides good sidelobe suppression in Fourier
    space, reducing ringing artifacts at the cost of tapering the entire
    image. For PSF matching, use this window when edge effects and
    ringing artifacts are a primary concern and you can accept tapering
    most of the data. If you want to preserve more of the central
    region, consider using `TukeyWindow` instead.

    Notes
    -----
    Equivalent to ``SplitCosineBellWindow(alpha=1.0, beta=0.0)``.

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils.psf_matching import HanningWindow

        taper = HanningWindow()
        data = taper((101, 101))
        plt.imshow(data, origin='lower')
        plt.colorbar()

    A 1D cut across the image center:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils.psf_matching import HanningWindow

        taper = HanningWindow()
        data = taper((101, 101))
        plt.plot(data[50, :])
    """

    def __init__(self):
        # alpha=1.0 (full taper), beta=0.0 (taper starts at center)
        super().__init__(alpha=1.0, beta=0.0)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __str__(self):
        return self.__repr__()


class TukeyWindow(SplitCosineBellWindow):
    """
    Class to define a 2D `Tukey window
    <https://en.wikipedia.org/wiki/Window_function#Tukey_window>`_
    function.

    The Tukey window features a flat inner plateau equal to 1.0,
    surrounded by a smooth cosine taper that transitions to 0.0 at the
    edges. This provides an excellent balance between preserving data in
    the central region and suppressing edge artifacts.

    The ``alpha`` parameter controls the trade-off: smaller values
    preserve more data but create stronger edge effects, while larger
    values reduce artifacts but taper more of the image.

    Compared to `HanningWindow`, Tukey preserves a larger central
    region. Compared to `TopHatWindow`, it provides much better artifact
    suppression at the cost of tapering the outer regions.

    Parameters
    ----------
    alpha : float, optional
        The percentage of array values that are tapered. Must be
        between 0.0 and 1.0, inclusive. When ``alpha=0``, this
        becomes a `TopHatWindow`. When ``alpha=1``, this becomes a
        `HanningWindow`.

    Notes
    -----
    Equivalent to ``SplitCosineBellWindow(alpha=alpha, beta=1.0 -
    alpha)``.

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils.psf_matching import TukeyWindow

        taper = TukeyWindow(alpha=0.4)
        data = taper((101, 101))
        plt.imshow(data, origin='lower')
        plt.colorbar()

    A 1D cut across the image center:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils.psf_matching import TukeyWindow

        taper = TukeyWindow(alpha=0.4)
        data = taper((101, 101))
        plt.plot(data[50, :])
    """

    def __init__(self, alpha):
        super().__init__(alpha=alpha, beta=1.0 - alpha)

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(alpha={self.alpha!r})')

    def __str__(self):
        return self.__repr__()


class CosineBellWindow(SplitCosineBellWindow):
    """
    Class to define a 2D cosine bell window function.

    This window equals 1.0 only at the exact center point and smoothly
    tapers to 0.0. The taper begins immediately from the center (no
    inner plateau) and extends outward over a fraction ``alpha`` of the
    maximum radius.

    Use this window when you want to preserve the very center of an
    image while applying a gentle taper that starts relatively far from
    the edges. It provides less artifact suppression than `TukeyWindow`
    for the same ``alpha`` value because the taper region is positioned
    differently.

    Parameters
    ----------
    alpha : float, optional
        The percentage of array values that are tapered. Must be between
        0.0 and 1.0, inclusive. When ``alpha=1``, this becomes a
        `HanningWindow`.

    Notes
    -----
    Equivalent to ``SplitCosineBellWindow(alpha=alpha, beta=0.0)``.

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils.psf_matching import CosineBellWindow

        taper = CosineBellWindow(alpha=0.3)
        data = taper((101, 101))
        plt.imshow(data, origin='lower')
        plt.colorbar()

    A 1D cut across the image center:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils.psf_matching import CosineBellWindow

        taper = CosineBellWindow(alpha=0.3)
        data = taper((101, 101))
        plt.plot(data[50, :])
    """

    def __init__(self, alpha):
        super().__init__(alpha=alpha, beta=0.0)

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(alpha={self.alpha!r})')

    def __str__(self):
        return self.__repr__()


class TopHatWindow(SplitCosineBellWindow):
    """
    Class to define a 2D top hat window function.

    This window equals 1.0 inside a circular region defined by ``beta``
    and drops sharply to 0.0 outside, with no smooth transition. It is
    also known as a rectangular or boxcar window.

    This window preserves the most data (everything inside the cutoff
    radius is untouched), but the sharp edge creates strong ringing
    artifacts in Fourier space. Use this only when you need to strictly
    preserve data within a specific region and can tolerate significant
    artifacts, or when the sharp cutoff is explicitly desired.

    For most PSF matching applications, `TukeyWindow` is preferred as
    it provides much better artifact suppression while still preserving
    a large central region. Use `TopHatWindow` primarily for masking or
    when studying the effects of abrupt truncation.

    Parameters
    ----------
    beta : float, optional
        The inner diameter as a fraction of the array size beyond
        which the window drops to zero. Must be between 0.0 and 1.0,
        inclusive.

    Notes
    -----
    Equivalent to ``SplitCosineBellWindow(alpha=0.0, beta=beta)``.

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils.psf_matching import TopHatWindow

        taper = TopHatWindow(beta=0.4)
        data = taper((101, 101))
        plt.imshow(data, origin='lower', interpolation='nearest')
        plt.colorbar()

    A 1D cut across the image center:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils.psf_matching import TopHatWindow

        taper = TopHatWindow(beta=0.4)
        data = taper((101, 101))
        plt.plot(data[50, :])
    """

    def __init__(self, beta):
        super().__init__(alpha=0.0, beta=beta)

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(beta={self.beta!r})')

    def __str__(self):
        return self.__repr__()
