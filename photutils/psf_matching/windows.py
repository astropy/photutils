# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Window (tapering) functions for matching PSFs using Fourier methods.
"""

import numpy as np

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
                   'Got: {alpha}')
            raise ValueError(msg)
        if not (0.0 <= beta <= 1.0):
            msg = ('beta must be between 0.0 and 1.0, inclusive. '
                   'Got: {beta}')
            raise ValueError(msg)

        self.alpha = float(alpha)
        self.beta = float(beta)

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
    that touch zero.

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


class TukeyWindow(SplitCosineBellWindow):
    """
    Class to define a 2D `Tukey window
    <https://en.wikipedia.org/wiki/Window_function#Tukey_window>`_
    function.

    The Tukey window is a taper formed by using a split cosine bell
    function with ends that touch zero.

    Parameters
    ----------
    alpha : float, optional
        The percentage of array values that are tapered.

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


class CosineBellWindow(SplitCosineBellWindow):
    """
    Class to define a 2D cosine bell window function.

    Parameters
    ----------
    alpha : float, optional
        The percentage of array values that are tapered.

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


class TopHatWindow(SplitCosineBellWindow):
    """
    Class to define a 2D top hat window function.

    Parameters
    ----------
    beta : float, optional
        The inner diameter as a fraction of the array size beyond which
        the taper begins. ``beta`` must be less or equal to 1.0.

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
