.. _performance-tips:

****************
Performance Tips
****************

.. _bottleneck-performance:

Bottleneck
==========

The optional `Bottleneck <https://github.com/pydata/bottleneck>`_
package provides fast, NaN-aware replacements for NumPy's ``nansum``,
``nanmin``, ``nanmax``, ``nanmean``, ``nanmedian``, ``nanstd``, and
``nanvar`` functions. When Bottleneck is installed, Photutils will
automatically use it for these operations, improving performance for any
workflow that computes statistics on arrays containing NaN values (e.g.,
masked pixels).

Bottleneck acceleration is used internally by the following Photutils
packages:

* `~photutils.background` — background and background RMS estimation
  (e.g., `~photutils.background.Background2D`)
* `~photutils.detection` — source detection peak finding
* `~photutils.profiles` — radial-profile and curve-of-growth
  calculations
* `~photutils.psf` — ePSF building
  (e.g., `~photutils.psf.EPSFBuilder`)
* `~photutils.segmentation` — source detection and deblending

.. note::

    Due to known accuracy issues in Bottleneck with ``float32``
    arrays (see `bottleneck #379
    <https://github.com/pydata/bottleneck/issues/379>`_ and
    `bottleneck #462
    <https://github.com/pydata/bottleneck/issues/462>`_),
    Photutils uses Bottleneck only for ``float64`` arrays and falls back
    to NumPy for other dtypes.

To install Bottleneck::

    python -m pip install bottleneck


.. _byteorder-performance:

Array Byte Order (Endianness)
=============================

Bottleneck requires that the byte order of the input data array matches
the native byte order of the operating system (typically little-endian
on modern processors). Arrays loaded by `astropy.io.fits` are stored as
big-endian. If the byte order does not match, Bottleneck will not be
used and the code will fall back to NumPy.

You can convert a big-endian FITS array to native byte order *in place*,
without allocating additional memory, using::

    >>> data.byteswap(inplace=True)  # doctest: +SKIP
    >>> data.dtype = data.dtype.newbyteorder('=')  # doctest: +SKIP

Alternatively, you can create a native-endian copy with::

    >>> data = data.astype(float)  # doctest: +SKIP

The first approach is preferred for large arrays because it avoids
allocating a temporary copy of the entire array.
