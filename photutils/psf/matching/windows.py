# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Window (or tapering) functions for matching PSFs using Fourier methods.
"""

import numpy as np


__all__ = ['SplitCosineBellWindow', 'HanningWindow', 'TukeyWindow',
           'CosineBellWindow', 'TopHatWindow']


def _radial_distance(shape):
    """
    Return an array where each value is the Euclidean distance from the
    array center.

    Parameters
    ----------
    shape : tuple of int
        The size of the output array along each axis.

    Returns
    -------
    result : `~numpy.ndarray`
        An array containing the Euclidian radial distances from the
        array center.
    """

    if len(shape) != 2:
        raise ValueError('shape must have only 2 elements')
    position = (np.asarray(shape) - 1) / 2.
    x = np.arange(shape[1]) - position[1]
    y = np.arange(shape[0]) - position[0]
    xx, yy = np.meshgrid(x, y)
    return np.sqrt(xx**2 + yy**2)


class SplitCosineBellWindow:
    """
    Class to define a 2D split cosine bell taper function.

    Parameters
    ----------
    alpha : float, optional
        The percentage of array values that are tapered.

    beta : float, optional
        The inner diameter as a fraction of the array size beyond which
        the taper begins. ``beta`` must be less or equal to 1.0.

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils import SplitCosineBellWindow
        taper = SplitCosineBellWindow(alpha=0.4, beta=0.3)
        data = taper((101, 101))
        plt.imshow(data, cmap='viridis', origin='lower')
        plt.colorbar()

    A 1D cut across the image center:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils import SplitCosineBellWindow
        taper = SplitCosineBellWindow(alpha=0.4, beta=0.3)
        data = taper((101, 101))
        plt.plot(data[50, :])
    """

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, shape):
        """
        Return a 2D split cosine bell.

        Parameters
        ----------
        shape : tuple of int
            The size of the output array along each axis.

        Returns
        -------
        result : `~numpy.ndarray`
            A 2D array containing the cosine bell values.
        """

        radial_dist = _radial_distance(shape)
        npts = (np.array(shape).min() - 1.) / 2.
        r_inner = self.beta * npts
        r = radial_dist - r_inner
        r_taper = int(np.floor(self.alpha * npts))

        if r_taper != 0:
            f = 0.5 * (1.0 + np.cos(np.pi * r / r_taper))
        else:
            f = np.ones(shape)

        f[radial_dist < r_inner] = 1.
        r_cut = r_inner + r_taper
        f[radial_dist > r_cut] = 0.

        return f


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
        from photutils import HanningWindow
        taper = HanningWindow()
        data = taper((101, 101))
        plt.imshow(data, cmap='viridis', origin='lower')
        plt.colorbar()

    A 1D cut across the image center:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils import HanningWindow
        taper = HanningWindow()
        data = taper((101, 101))
        plt.plot(data[50, :])
    """

    def __init__(self):
        self.alpha = 1.0
        self.beta = 0.0


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
        from photutils import TukeyWindow
        taper = TukeyWindow(alpha=0.4)
        data = taper((101, 101))
        plt.imshow(data, cmap='viridis', origin='lower')
        plt.colorbar()

    A 1D cut across the image center:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils import TukeyWindow
        taper = TukeyWindow(alpha=0.4)
        data = taper((101, 101))
        plt.plot(data[50, :])

    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.beta = 1. - self.alpha


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
        from photutils import CosineBellWindow
        taper = CosineBellWindow(alpha=0.3)
        data = taper((101, 101))
        plt.imshow(data, cmap='viridis', origin='lower')
        plt.colorbar()

    A 1D cut across the image center:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils import CosineBellWindow
        taper = CosineBellWindow(alpha=0.3)
        data = taper((101, 101))
        plt.plot(data[50, :])
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.beta = 0.0


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
        from photutils import TopHatWindow
        taper = TopHatWindow(beta=0.4)
        data = taper((101, 101))
        plt.imshow(data, cmap='viridis', origin='lower',
                   interpolation='nearest')
        plt.colorbar()

    A 1D cut across the image center:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils import TopHatWindow
        taper = TopHatWindow(beta=0.4)
        data = taper((101, 101))
        plt.plot(data[50, :])
    """

    def __init__(self, beta):
        self.alpha = 0.0
        self.beta = beta
