
0.3 (unreleased)
----------------

General
^^^^^^^

- Drop numpy 1.6 support, minimal required version is now numpy 1.7.
  [#327]

New Features
^^^^^^^^^^^^

- ``photutils.background``

  - Added new ``BackgroundBase`` and ``BackgroundIDW`` classes and
    ``std_blocksum`` function. [#355]

  - Added new ``removed_masked``, ``meshpix_threshold``, and
    ``edge_method`` keywords for the background classes. [#355]

  - Added new ``std_blocksum`` function. [#355]

- ``photutils.detection``

  - Added new ``make_source_mask`` convenience function. [#355]

<<<<<<< HEAD
- ``photutils.segmentation``

  - Added ``copy`` and ``area`` methods and a ``areas`` property to
    ``SegmentationImage``. [#331]

  - SegmentationImage properties are now cached to significantly
    improve performance. [#361]

=======
- ``photutils.psf_photometry``

  - Added the option ``param_uncert`` so that users can easily get
    uncertainties on fitted parameters. [#358]
>>>>>>> add changelog and proper warning in case fitter does not contain 'param_cov' key

API changes
^^^^^^^^^^^

- ``photutils.aperture``

  - Removed the ``effective_gain`` keyword from
    ``aperture_photometry``.  Users must now input the total error,
    which can be calculating using the ``calc_total_error`` function.
    [#368]

- ``photutils.background``

  - For the background classes, the ``filter_shape`` keyword was
    renamed to ``filter_size``.  The ``mode_estimate`` keyword was
    renamed to ``mode_estimator``.  The ``background_low_res`` and
    ``background_rms_low_res`` class attributes were renamed to
    ``background_mesh2d`` and ``background_rms_mesh2d``, respectively.
    [#355]

- ``photutils.psf``

  - Removed the ``effective_gain`` keyword from ``psf_photometry``.
    Users must now input the total error, which can be calculating using
    the ``calc_total_error`` function. [#368]

- ``photutils.segmentation``

  - Removed the ``effective_gain`` keyword from ``SourceProperties``
    and ``source_properties``.  Users must now input the total error,
    which can be calculating using the ``calc_total_error`` function.
    [#368]

- ``photutils.utils``

  - Renamed ``calculate_total_error`` to ``calc_total_error``. [#368]


0.2.2 (unreleased)
------------------

Bug Fixes
^^^^^^^^^

- ``photutils.detection``

  - Fixed an issue to update segmentation image slices after
    deblending. [#340]

  - Fixed source deblending to pass the pixel connectivity to the
    watershed algorithm. [#347]

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
