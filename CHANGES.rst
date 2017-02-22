0.4 (unreleased)
----------------

General
^^^^^^^

New Features
^^^^^^^^^^^^

API changes
^^^^^^^^^^^

- ``photutils.aperture``

  - The ``ApertureMask`` ``apply()`` method has been renamed to
    ``multiply()``. [#481].

Bug Fixes
^^^^^^^^^


0.3.1 (unreleased)
------------------

General
^^^^^^^

- Dropped numpy 1.7 support. Minimal required version is now numpy
  1.8. [#327]

- ``photutils.datasets``

  - The ``load_*`` functions that use remote data now retrieve the
    data from ``data.astropy.org`` (the astropy data repository).
    [#472]

Bug Fixes
^^^^^^^^^

- ``photutils.background``

  - Fixed issue with ``Background2D`` with ``edge_method='pad'`` that
    occurred when unequal padding needed to be applied to each axis.
    [#498]

  - Fixed issue with ``Background2D`` that occurred when zero padding
    needed to apply along only one axis. [#500]

- ``photutils.geometry``

  - Fixed a bug in ``circular_overlap_grid`` affecting 32-bit machines
    that could cause errors circular aperture photometry. [#475]

- ``photutils.psf``

  - Fixed a bug in how ``FittableImageModel`` represents its center.
    [#460]

  -  Fix bug which modified user's input table when doing forced
     photometry. [#485]


0.3 (2016-11-06)
----------------

General
^^^^^^^

New Features
^^^^^^^^^^^^

- ``photutils.aperture``

  - Added new ``origin`` keyword to aperture ``plot`` methods. [#395]

  - Added new ``id`` column to ``aperture_photometry`` output table. [#446]

  - Added ``__len__`` method for aperture classes. [#446]

  - Added new ``to_mask`` method to ``PixelAperture`` classes. [#453]

  - Added new ``ApertureMask`` class to generate masks from apertures.
    [#453]

  - Added new ``mask_area()`` method to ``PixelAperture`` classes.
    [#453]

  - The ``aperture_photometry()`` function now accepts a list of
    aperture objects. [#454]

- ``photutils.background``

  - Added new ``MeanBackground``, ``MedianBackground``,
    ``MMMBackground``, ``SExtractorBackground``,
    ``BiweightLocationBackground``, ``StdBackgroundRMS``,
    ``MADStdBackgroundRMS``, and ``BiweightMidvarianceBackgroundRMS``
    classes. [#370]

  - Added ``axis`` keyword to new background classes. [#392]

  - Added new ``removed_masked``, ``meshpix_threshold``, and
    ``edge_method`` keywords for the 2D background classes. [#355]

  - Added new ``std_blocksum`` function. [#355]

  - Added new ``SigmaClip`` class. [#423]

  - Added new ``BkgZoomInterpolator`` and ``BkgIDWInterpolator``
    classes. [#437]

- ``photutils.datasets``

  - Added ``load_irac_psf`` function. [#403]

- ``photutils.detection``

  - Added new ``make_source_mask`` convenience function. [#355]

  - Added ``filter_data`` function. [#398]

  - Added ``DAOStarFinder`` and ``IRAFStarFinder`` as oop interfaces for
    ``daofind`` and ``irafstarfinder``, respectively, which are now
    deprecated. [#379]

- ``photutils.psf``

  - Added ``BasicPSFPhotometry``, ``IterativelySubtractedPSFPhotometry``, and
    ``DAOPhotPSFPhotometry`` classes to perform PSF photometry in
    crowded fields. [#427]

  - Added ``DAOGroup`` and ``DBSCANGroup`` classes for grouping overlapping
    sources. [#369]

- ``photutils.psf_match``

  - Added ``create_matching_kernel`` and ``resize_psf`` functions.  Also
    added ``CosineBellWindow``, ``HanningWindow``,
    ``SplitCosineBellWindow``, ``TopHatWindow``, and ``TukeyWindow``
    classes. [#403]

- ``photutils.segmentation``

  - Created new ``photutils.segmentation`` subpackage. [#442]

  - Added ``copy`` and ``area`` methods and an ``areas`` property to
    ``SegmentationImage``. [#331]

API changes
^^^^^^^^^^^

- ``photutils.aperture``

  - Removed the ``effective_gain`` keyword from
    ``aperture_photometry``.  Users must now input the total error,
    which can be calculated using the ``calc_total_error`` function.
    [#368]

  - ``aperture_photometry`` now outputs a ``QTable``. [#446]

  - Renamed ``source_id`` keyword to ``indices`` in the aperture
    ``plot()`` method. [#453]

  - Added ``mask`` and ``unit`` keywords to aperture
    ``do_photometry()`` methods.  [#453]

- ``photutils.background``

  - For the background classes, the ``filter_shape`` keyword was
    renamed to ``filter_size``.  The ``background_low_res`` and
    ``background_rms_low_res`` class attributes were renamed to
    ``background_mesh`` and ``background_rms_mesh``, respectively.
    [#355, #437]

  - The ``Background2D`` ``method`` and ``backfunc`` keywords have
    been removed.  In its place one can input callable objects via the
    ``sigma_clip``, ``bkg_estimator``, and ``bkgrms_estimator``
    keywords. [#437]

  - The interpolator to be used by the ``Background2D`` class can be
    input as a callable object via the new ``interpolator`` keyword.
    [#437]

- ``photutils.centroids``

  - Created ``photutils.centroids`` subpackage, which contains the
    ``centroid_com``, ``centroid_1dg``, and ``centroid_2dg``
    functions.  These functions now return a two-element numpy
    ndarray.  [#428]

- ``photutils.detection``

  - Changed finding algorithm implementations (``daofind`` and
    ``starfind``) from functional to object-oriented style. Deprecated
    old style. [#379]

- ``photutils.morphology``

  - Created ``photutils.morphology`` subpackage. [#428]

  - Removed ``marginalize_data2d`` function. [#428]

  - Moved ``cutout_footprint`` from ``photutils.morphology`` to
    ``photutils.utils``. [#428]

  - Added a function to calculate the Gini coefficient (``gini``).
    [#343]

- ``photutils.psf``

  - Removed the ``effective_gain`` keyword from ``psf_photometry``.
    Users must now input the total error, which can be calculated
    using the ``calc_total_error`` function. [#368]

- ``photutils.segmentation``

  - Removed the ``effective_gain`` keyword from ``SourceProperties``
    and ``source_properties``.  Users must now input the total error,
    which can be calculated using the ``calc_total_error`` function.
    [#368]

- ``photutils.utils``

  - Renamed ``calculate_total_error`` to ``calc_total_error``. [#368]

Bug Fixes
^^^^^^^^^

- ``photutils.aperture``

  - Fixed a bug in ``aperture_photometry`` so that single-row output
    tables do not return a multidimensional column. [#446]

- ``photutils.centroids``

  - Fixed a bug in ``centroid_1dg`` and ``centroid_2dg`` that occured
    when the input data contained invalid (NaN or inf) values.  [#428]

- ``photutils.segmentation``

  - Fixed a bug in ``SourceProperties`` where ``error`` and
    ``background`` units were sometimes dropped. [#441]


0.2.2 (2016-07-06)
------------------

General
^^^^^^^

- Dropped numpy 1.6 support. Minimal required version is now numpy
  1.7. [#327]

- Fixed configparser for Python 3.5. [#366, #384]

Bug Fixes
^^^^^^^^^

- ``photutils.detection``

  - Fixed an issue to update segmentation image slices after
    deblending. [#340]

  - Fixed source deblending to pass the pixel connectivity to the
    watershed algorithm. [#347]

  - SegmentationImage properties are now cached instead of recalculated,
    which significantly improves performance. [#361]

- ``photutils.utils``

  - Fixed a bug in ``pixel_to_icrs_coords`` where the incorrect pixel
    origin was being passed. [#348]


0.2.1 (2016-01-15)
------------------

Bug Fixes
^^^^^^^^^

- ``photutils.background``

  - Added more robust version checking of Astropy. [#318]

- ``photutils.detection``

  - Added more robust version checking of Astropy. [#318]

- ``photutils.segmentation``

  - Fixed issue where ``SegmentationImage`` slices were not being updated.
    [#317]

  - Added more robust version checking of scikit-image. [#318]


0.2 (2015-12-31)
----------------

General
^^^^^^^

- Photutils has the following requirements:

  - Python 2.7 or 3.3 or later

  - Numpy 1.6 or later

  - Astropy v1.0 or later

New Features
^^^^^^^^^^^^

- ``photutils.detection``

  - ``find_peaks`` now returns an Astropy Table containing the (x, y)
    positions and peak values. [#240]

  - ``find_peaks`` has new ``mask``, ``error``, ``wcs`` and ``subpixel``
    precision options. [#244]

  - ``detect_sources`` will now issue a warning if the filter kernel
    is not normalized to 1. [#298]

  - Added new ``deblend_sources`` function, an experimental source
    deblender. [#314]

- ``photutils.morphology``

  - Added new ``GaussianConst2D`` (2D Gaussian plus a constant) model.
    [#244]

  - Added new ``marginalize_data2d`` function. [#244]

  - Added new ``cutout_footprint`` function. [#244]

- ``photutils.segmentation``

  - Added new ``SegmentationImage`` class. [#306]

  - Added new ``check_label``, ``keep_labels``, and ``outline_segments``
    methods for modifying ``SegmentationImage``. [#306]

- ``photutils.utils``

  - Added new ``random_cmap`` function to generate a colormap comprised
    of random colors. [#299]

  - Added new ``ShepardIDWInterpolator`` class to perform Inverse
    Distance Weighted (IDW) interpolation. [#307]

  - The ``interpolate_masked_data`` function can now interpolate
    higher-dimensional data. [#310]

API changes
^^^^^^^^^^^

- ``photutils.segmentation``

  - The ``relabel_sequential``, ``relabel_segments``,
    ``remove_segments``, ``remove_border_segments``, and
    ``remove_masked_segments`` functions are now ``SegmentationImage``
    methods (with slightly different names). [#306]

  - The ``SegmentProperties`` class has been renamed to
    ``SourceProperties``.  Likewise the ``segment_properties`` function
    has been renamed to ``source_properties``. [#306]

  - The ``segment_sum`` and ``segment_sum_err`` attributes have been
    renamed to ``source_sum`` and ``source_sum_err``, respectively. [#306]

  - The ``background_atcentroid`` attribute has been renamed to
    ``background_at_centroid``. [#306]

Bug Fixes
^^^^^^^^^

- ``photutils.aperture_photometry``

  - Fixed an issue where ``np.nan`` or ``np.inf`` were not properly
    masked. [#267]

- ``photutils.geometry``

  - ``overlap_area_triangle_unit_circle`` handles correctly a corner case
    in some i386 systems where the area of the aperture was not computed
    correctly. [#242]

  - ``rectangular_overlap_grid`` and ``elliptical_overlap_grid`` fixes to
    normalization of subsampled pixels. [#265]

  - ``overlap_area_triangle_unit_circle`` handles correctly the case where
    a line segment intersects at a triangle vertex. [#277]

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Updated astropy-helpers to v1.1. [#302]


0.1 (2014-12-22)
----------------

Photutils 0.1 was released on December 22, 2014.  It requires Astropy
version 0.4 or later.
