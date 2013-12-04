
# Credits to scikit-image (scikit-image.measure_moments.pyx)
import numpy as np


def moments(image, order):
    """
    Calculate all raw image moments up to a certain order.

    The following properties can be calculated from raw image moments:
     * Area as ``m[0, 0]``.
     * Centroid as {``m[0, 1] / m[0, 0]``, ``m[1, 0] / m[0, 0]``}.

    Parameters
    ----------
    image : 2D double array
        Rasterized shape as image.
    order : int
        Maximum order of moments.

    Returns
    -------
    m : (``order + 1``, ``order + 1``) array
        Raw image moments.

    References
    ----------
    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [2] B. Jahne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [4] http://en.wikipedia.org/wiki/Image_moment

    """
    return moments_central(image, 0, 0, order)


def moments_central(image, xc, yc, order):
    """
    Calculate all central image moments up to a certain order.

    Parameters
    ----------
    image : 2D double array
        Rasterized shape as image.
    xc : double
        Center x (column) coordinate.
    yc : double
        Center y (row) coordinate.
    order : int
        Maximum order of moments.

    Returns
    -------
    mu : (``order + 1``, ``order + 1``) array
        Central image moments.

    References
    ----------
    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [2] B. Jahne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [4] http://en.wikipedia.org/wiki/Image_moment

    """
    mu = np.zeros((order + 1, order + 1), dtype=np.double)
    for p in range(order + 1):
        for q in range(order + 1):
            for j in range(image.shape[0]):
                for i in range(image.shape[1]):
                    mu[p, q] += image[j, i] * (j - yc) ** q * (i - xc) ** p
    return np.asarray(mu)

