#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np


def moments(double[:, :] image, Py_ssize_t order=3):
    """Calculate all raw image moments up to a certain order.

    The following properties can be calculated from raw image moments:
     * Area as ``m[0, 0]``.
     * Centroid as {``m[0, 1] / m[0, 0]``, ``m[1, 0] / m[0, 0]``}.

    Note that raw moments are neither translation, scale nor rotation
    invariant.

    Parameters
    ----------
    image : 2D double array
        Rasterized shape as image.
    order : int, optional
        Maximum order of moments. Default is 3.

    Returns
    -------
    m : (``order + 1``, ``order + 1``) array
        Raw image moments.

    References
    ----------
    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [2] B. Jähne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [4] http://en.wikipedia.org/wiki/Image_moment

    """
    return moments_central(image, 0, 0, order)


def moments_central(double[:, :] image, double cr, double cc,
                    Py_ssize_t order=3):
    """Calculate all central image moments up to a certain order.

    Note that central moments are translation invariant but not scale and
    rotation invariant.

    Parameters
    ----------
    image : 2D double array
        Rasterized shape as image.
    cr : double
        Center row coordinate.
    cc : double
        Center column coordinate.
    order : int, optional
        Maximum order of moments. Default is 3.

    Returns
    -------
    mu : (``order + 1``, ``order + 1``) array
        Central image moments.

    References
    ----------
    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [2] B. Jähne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [4] http://en.wikipedia.org/wiki/Image_moment

    """
    cdef Py_ssize_t p, q, r, c
    cdef double[:, ::1] mu = np.zeros((order + 1, order + 1), dtype=np.double)
    for p in range(order + 1):
        for q in range(order + 1):
            for r in range(image.shape[0]):
                for c in range(image.shape[1]):
                    mu[p, q] += image[r, c] * (r - cr) ** q * (c - cc) ** p
    return np.asarray(mu)

