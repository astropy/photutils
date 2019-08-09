0.7 (unreleased)
----------------

General
^^^^^^^

 - Any WCS object that supports the `astropy shared interface for WCS
   <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ is now
   supported. [#899]

New Features
^^^^^^^^^^^^

- ``photutils.aperture``

  - Added parameter validation for all aperture classes. [#846]

  - Added ``from_float``, ``as_artist``, ``union`` and
    ``intersection`` methods to ``BoundingBox`` class. [#851]

  - Added ``shape`` and ``isscalar`` properties to Aperture objects.
    [#852]

  - Significantly improved the performance (~10-20 times faster) of
    aperture photometry, especially when using ``errors`` and
    ``Quantity`` inputs with many aperture positions. [#861]

  - ``aperture_photometry`` now supports ``NDData`` with
    ``StdDevUncertainty`` to input errors. [#866]

  - The ``mode`` keyword in the ``to_sky`` and ``to_pixel`` aperture
    methods was removed to implement the shared WCS interface.  All
    WCS transforms now include distortions (if present). [#899]

- ``photutils.datasets``

  - Added ``make_gwcs`` function to create an example ``gwcs.wcs.WCS``
    object. [#871]

- ``photutils.isophote``

  - Significantly improved the performance (~5 times faster) of
    ellipse fitting. [#826]

  - Added the ability to individually fix the ellipse-fitting
    parameters. [#922]

- ``photutils.psf``

  - Added new centroiding function ``centroid_epsf``. [#816]

- ``photutils.segmentation``

  - Significantly improved the performance of relabeling in
    segmentation images (e.g. ``remove_labels``, ``keep_labels``).
    [#810]

  - Added new ``background_area`` attribute to ``SegmentationImage``.
    [#825]

  - Added new ``data_ma`` attribute to ``Segment``. [#825]

  - Added new ``SegmentationImage`` methods:  ``find_index``,
    ``find_indices``, ``find_areas``, ``check_label``, ``keep_label``,
    ``remove_label``, and ``reassign_labels``. [#825]

  - Added ``__repr__`` and ``__str__`` methods to
    ``SegmentationImage``. [#825]

  - Added ``slices``, ``indices``, and ``filtered_data_cutout_ma``
    attributes to ``SourceProperties``. [#858]

  - Added ``__repr__`` and ``__str__`` methods to ``SourceProperties``
    and ``SourceCatalog``. [#858]

  - Significantly improved the performance of calculating the
    ``background_at_centroid`` property in ``SourceCatalog``. [#863]

  - The default output table columns (source properties) are defined
    in a publicly-accessible variable called
    ``photutils.segmentation.properties.DEFAULT_COLUMNS``. [#863]

  - Added the ``gini`` source property representing the Gini
    coefficient. [#864]

  - Cached (lazy) properties can now be reset in ``SegmentationImage``
    subclasses. [#916]

- ``photutils.utils``

  - Added ``NoDetectionsWarning`` class. [#836]

Bug Fixes
^^^^^^^^^

- ``photutils.aperture``

  - Fixed an issue where the ``ApertureMask.cutout`` method would drop
    the data units when ``copy=True``. [#842]

  - Fixed a corner-case issue where aperture photometry would return
    NaN for non-finite data values outside the aperture but within the
    aperture bounding box. [#843]

  - Fixed an issue where the ``celestial_center`` column (for sky
    apertures) would be a length-1 array containing a ``SkyCoord``
    object instead of a length-1 ``SkyCoord`` object. [#844]

- ``photutils.isophote``

  - Fixed an issue where the linear fitting mode was not working.
    [#912]

- ``photutils.psf``

  - Fixed a bug in the ``EPSFStar`` ``register_epsf`` and
    ``compute_residual_image`` computations. [#885]

  - A ValueError is raised if ``aperture_radius`` is not input and
    cannot be determined from the input ``psf_model``. [#903]

  - Fixed normalization of ePSF model, now correctly normalizing on
    undersampled pixel grid. [#817]

- ``photutils.segmentation``

  - Fixed an issue where ``deblend_sources`` could fail for sources
    with labels that are a power of 2 and greater than 255. [#806]

  - ``SourceProperties`` and ``source_properties`` will no longer
    raise an exception if a source is completely masked. [#822]

  - Fixed an issue in ``SourceProperties`` and ``source_properties``
    where inf values in the data array were not automatically masked.
    [#822]

  - ``error`` and ``background`` arrays are now always masked
    identically to the input ``data``. [#822]

  - Fixed the ``perimeter`` property to take into account the source
    mask. [#822]

  - Fixed the ``background_at_centroid`` source property to use
    bilinear interpolation. [#822]

  - Fixed ``SegmentationImage`` ``outline_segments`` to include
    outlines along the image boundaries. [#825]

  - Fixed ``SegmentationImage.is_consecutive`` to return ``True`` only
    if the labels are consecutive and start with label=1. [#886]

  - Fixed a bug in ``deblend_sources`` where sources could be
    deblended too much when ``connectivity=8``. [#890]

  - Fixed a bug in ``deblend_sources`` where the ``contrast``
    parameter had little effect if the original segment contained
    three or more sources. [#890]

- ``photutils.utils``

  - Fixed a bug in ``filter_data`` where units were dropped for data
    ``Quantity`` objects. [#872]

API changes
^^^^^^^^^^^

- ``photutils.aperture``

  - Deprecated inputting aperture pixel positions shaped as 2xN.
    [#847]

  - Renamed the ``celestial_center`` column to ``sky_center`` in the
    ``aperture_photometry`` output table. [#848]

  - Aperture objects defined with a single (x, y) position (input as
    1D) are now considered scalar objects, which can be checked with
    the new ``isscalar`` Aperture property. [#852]

  - Non-scalar Aperture objects can now be indexed, sliced, and
    iterated. [#852]

  - Scalar Aperture objects now return scalar ``positions`` and
    ``bounding_boxes`` properties and its ``to_mask`` method returns
    an ``ApertureMask`` object instead of a length-1 list containing
    an ``ApertureMask``. [#852]

  - Deprecated the Aperture ``mask_area`` method. [#853]

  - Aperture ``area`` is now an attribute instead of a method. [#854]

  - The Aperture plot keyword ``ax`` was deprecated and renamed to
    ``axes``. [#854]

  - Deprecated the ``units`` keyword in ``aperture_photometry``
    and the ``PixelAperture.do_photometry`` method. [#866, #861]

  - Deprecated ``PrimaryHDU``, ``ImageHDU``, and ``HDUList`` inputs
    to ``aperture_photometry``. [#867]

  - The ``aperture_photometry`` function moved to a new
    ``photutils.aperture.photometry`` module. [#876]

  - Renamed the ``bounding_boxes`` attribute for pixel-based apertures
    to ``bbox`` for consistency. [#896]

- ``photutils.background``

  - The ``Background2D`` ``plot_meshes`` keyword ``ax`` was deprecated
    and renamed to ``axes``. [#854]

- ``photutils.datasets``

  - The ``make_noise_image`` ``type`` keyword was deprecated and
    renamed to ``distribution``. [#877]

- ``photutils.detection``

  - Removed deprecated ``subpixel`` keyword for ``find_peaks``. [#835]

  - ``DAOStarFinder``, ``IRAFStarFinder``, and ``find_peaks`` now return
    ``None`` if no source/peaks are found.  Also, for this case a
    ``NoDetectionsWarning`` is issued. [#836]

  - Deprecated the ``BoundingBox`` ``as_patch`` method (instead use
    ``as_artist``). [#851]

  - Renamed the ``snr`` (deprecated) keyword to ``nsigma`` in
    ``detect_threshold``. [#917]

- ``photutils.isophote``

  - Isophote central values and intensity gradients are now returned
    in the output table. [#892]

- ``photutils.psf``

  - ``FittableImageModel`` and subclasses now allow for different
    ``oversampling`` factors to be specified in the x and y
    directions. [#834]

  - Removed ``pixel_scale`` keyword from ``EPSFStar``, ``EPSFBuilder``,
    and ``EPSFModel``. [#815]

  - Added ``oversampling`` keyword to ``centroid_com``. [#816]

  - Removed deprecated ``Star``, ``Stars``, and ``LinkedStar``
    classes. [#894]

- ``photutils.segmentation``

  - Removed deprecated ``SegmentationImage`` attributes
    ``data_masked``, ``max``, and ``is_sequential``  and methods
    ``area`` and ``relabel_sequential``. [#825]

  - Renamed ``SegmentationImage`` methods ``cmap`` (deprecated) to
    ``make_cmap`` and ``relabel`` (deprecated) to ``reassign_label``.
    The new ``reassign_label`` method gains a ``relabel`` keyword.
    [#825]

  - The ``SegmentationImage`` ``segments`` and ``slices`` attributes
    now have the same length as ``labels`` (no ``None`` placeholders).
    [#825]

  - ``detect_sources`` now returns ``None`` if no sources are found.
    Also, for this case a ``NoDetectionsWarning`` is issued. [#836]

  - The ``SegmentationImage`` input ``data`` array must contain at
    least one non-zero pixel and must not contain any non-finite values.
    [#836]

  - A ``ValueError`` is raised if an empty list is input into
    ``SourceCatalog`` or no valid sources are defined in
    ``source_properties``. [#836]

  - Deprecated the ``values`` and ``coords`` attributes in
    ``SourceProperties``. [#858]

  - Deprecated the unused ``mask_value`` keyword in
    ``make_source_mask``. [#858]

  - The ``bbox`` property now returns a ``BoundingBox`` instance.
    [#863]

  - The ``xmin/ymin`` and ``xmax/ymax`` properties have been
    deprecated with the replacements having a ``bbox_`` prefix (e.g.
    ``bbox_xmin``). [#863]

  - The ``orientation`` property is now returned as a ``Quantity``
    instance in units of degrees. [#863]

  - Renamed the ``snr`` (deprecated) keyword to ``nsigma`` in
    ``make_source_mask``. [#917]

- ``photutils.utils``

  - Renamed ``random_cmap`` to ``make_random_cmap``. [#825]

  - Removed deprecated ``cutout_footprint`` function. [#835]

  - Deprecated the ``wcs_helpers`` functions
    ``pixel_scale_angle_at_skycoord``, ``assert_angle_or_pixel``,
    ``assert_angle``, and ``pixel_to_icrs_coords``. [#846]

  - Removed deprecated ``interpolate_masked_data`` function. [#895]

  - Deprecated the ``mask_to_mirrored_num`` function. [#895]

  - Deprecated the ``get_version_info``, ``filter_data``, and
    ``std_blocksum`` functions. [#918]

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

-


0.6 (2018-12-11)
----------------

General
^^^^^^^

 - Versions of Numpy <1.11 are no longer supported. [#783]

New Features
^^^^^^^^^^^^

- ``photutils.detection``

  - ``DAOStarFinder`` and ``IRAFStarFinder`` gain two new parameters:
    ``brightest`` to keep the top ``brightest`` (based on the flux)
    objects in the returned catalog (after all other filtering has
    been applied) and ``peakmax`` to exclude sources with peak pixel
    values larger or equal to ``peakmax``. [#750]

  - Added a ``mask`` keyword to ``DAOStarFinder`` and
    ``IRAFStarFinder`` that can be used to mask regions of the input
    image.  [#759]

- ``photutils.psf``

  - The ``Star``, ``Stars``, and ``LinkedStars`` classes are now
    deprecated and have been renamed ``EPSFStar``, ``EPSFStars``, and
    ``LinkedEPSFStars``, respectively. [#727]

  - Added a ``GriddedPSFModel`` class for spatially-dependent PSFs.
    [#772]

  - The ``pixel_scale`` keyword in ``EPSFStar``, ``EPSFBuilder`` and
    ``EPSFModel`` is now deprecated.  Use the ``oversampling`` keyword
    instead. [#780]


API changes
^^^^^^^^^^^

- ``photutils.detection``

  - The ``find_peaks`` function now returns an empty
    ``astropy.table.Table`` instead of an empty list if the input data
    is an array of constant values. [#709]

  - The ``find_peaks`` function will no longer issue a RuntimeWarning
    if the input data contains NaNs. [#712]

  - If no sources/peaks are found, ``DAOStarFinder``,
    ``IRAFStarFinder``, and ``find_peaks`` now will return an empty
    table with column names and types. [#758, #762]

- ``photutils.psf``

  - The ``photutils.psf.funcs.py`` module was renamed
    ``photutils.psf.utils.py``. The ``prepare_psf_model`` and
    ``get_grouped_psf_model`` functions were also moved to this new
    ``utils.py`` module.  [#777]

Bug Fixes
^^^^^^^^^

- ``photutils.aperture``

  - If a single aperture is input as a list into the
    ``aperture_photometry`` function, then the output columns will be
    called ``aperture_sum_0`` and ``aperture_sum_err_0`` (if errors
    are used).  Previously these column names did not have the
    trailing "_0". [#779]

- ``photutils.segmentation``

  - Fixed a bug in the computation of ``sky_bbox_ul``,
    ``sky_bbox_lr``, ``sky_bbox_ur`` in the ``SourceCatalog``. [#716]

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Updated background and detection functions that call
  ``astropy.stats.SigmaClip`` or ``astropy.stats.sigma_clipped_stats``
  to support both their ``iters`` (for astropy < 3.1) and ``maxiters``
  keywords. [#726]


0.5 (2018-08-06)
----------------

General
^^^^^^^

- Versions of Python <3.5 are no longer supported. [#702, #703]

- Versions of Numpy <1.10 are no longer supported. [#697, #703]

- Versions of Pytest <3.1 are no longer supported. [#702]

- ``pytest-astropy`` is now required to run the test suite. [#702, #703]

- The documentation build now uses the Sphinx configuration from
  ``sphinx-astropy`` rather than from ``astropy-helpers``. [#702]


New Features
^^^^^^^^^^^^

- ``photutils.aperture``

  - Added ``plot`` and ``to_aperture`` methods to ``BoundingBox``. [#662]

  - Added default theta value for elliptical and rectangular
    apertures. [#674]

- ``photutils.centroid``

  - Added a ``centroid_sources`` function to calculate centroid of
    many sources in a single image. [#656]

  - An n-dimensional array can now be input into the ``centroid_com``
    function. [#685]

- ``photutils.datasets``

  - Added a ``load_simulated_hst_star_image`` function to load a
    simulated HST WFC3/IR F160W image of stars. [#695]

- ``photutils.detection``

  - Added a ``centroid_func`` keyword to ``find_peaks``.  The
    ``subpixels`` keyword is now deprecated. [#656]

  - The ``find_peaks`` function now returns ``SkyCoord`` objects in
    the table instead of separate RA and Dec. columns. [#656]

  - The ``find_peaks`` function now returns an empty Table and issues
    a warning when no peaks are found. [#668]

- ``photutils.psf``

  - Added tools to build and fit an effective PSF (``EPSFBuilder`` and
    ``EPSFFitter``). [#695]

  - Added ``extract_stars`` function to extract cutouts of stars used
    to build an ePSF. [#695]

  - Added ``EPSFModel`` class to hold a fittable ePSF model. [#695]

- ``photutils.segmentation``

  - Added a ``mask`` keyword to the ``detect_sources`` function. [#621]

  - Renamed ``SegmentationImage`` ``max`` attribute to ``max_label``.
    ``max`` is deprecated. [#662]

  - Added a ``Segment`` class to hold the cutout image and properties
    of single labeled region (source segment). [#662]

  - Deprecated the ``SegmentationImage`` ``area`` method.  Instead use
    the ``areas`` attribute. [#662]

  - Renamed ``SegmentationImage`` ``data_masked`` attribute to
    ``data_ma``.  ``data_masked`` is deprecated. [#662]

  - Renamed ``SegmentationImage`` ``is_sequential`` attribute to
    ``is_consecutive``.  ``is_sequential`` is deprecated. [#662]

  - Renamed ``SegmentationImage`` ``relabel_sequential`` attribute to
    ``relabel_consecutive``.  ``relabel_sequential`` is deprecated.
    [#662]

  - Added a ``missing_labels`` property to ``SegmentationImage``.
    [#662]

  - Added a ``check_labels`` method to ``SegmentationImage``.  The
    ``check_label`` method is deprecated. [#662]

- ``photutils.utils``

  - Deprecated the ``cutout_footprint`` function. [#656]

Bug Fixes
^^^^^^^^^

- ``photutils.aperture``

  - Fixed a bug where quantity inputs to the aperture classes would
    sometimes fail. [#693]

- ``photutils.detection``

  - Fixed an issue in ``detect_sources`` where in some cases sources
    with a size less than ``npixels`` could be returned. [#663]

  - Fixed an issue in ``DAOStarFinder`` where in some cases a few too
    many sources could be returned. [#671]

- ``photutils.isophote``

  - Fixed a bug where isophote fitting would fail when the initial
    center was not specified for an image with an elongated aspect
    ratio. [#673]

- ``photutils.segmentation``

  - Fixed ``deblend_sources`` when other sources are in the
    neighborhood. [#617]

  - Fixed ``source_properties`` to handle the case where the data
    contain one or more NaNs. [#658]

  - Fixed an issue with ``deblend_sources`` where sources were not
    deblended where the data contain one or more NaNs. [#658]

  - Fixed the ``SegmentationImage`` ``areas`` attribute to not include
    the zero (background) label. [#662]

Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``photutils.isophote``

  - Corrected the units for isophote ``sarea`` in the documentation. [#657]


0.4 (2017-10-30)
----------------

General
^^^^^^^

- Dropped python 3.3 support. [#542]

- Dropped numpy 1.8 support. Minimal required version is now numpy
  1.9. [#542]

- Dropped support for astropy 1.x versions.  Minimal required version
  is now astropy 2.0. [#575]

- Dropped scipy 0.15 support.  Minimal required version is now scipy
  0.16. [#576]

- Explicitly require six as dependency. [#601]

New Features
^^^^^^^^^^^^

- ``photutils.aperture``

  - Added ``BoundingBox`` class, used when defining apertures. [#481]

  - Apertures now have ``__repr__`` and ``__str__`` defined. [#493]

  - Improved plotting of annulus apertures using Bezier curves. [#494]

  - Rectangular apertures now use the true minimal bounding box. [#507]

  - Elliptical apertures now use the true minimal bounding box. [#508]

  - Added a ``to_sky`` method for pixel apertures. [#512]

- ``photutils.background``

  - Mesh rejection now also applies to pixels that are masked during
    sigma clipping. [#544]

- ``photutils.datasets``

  - Added new ``make_wcs`` and ``make_imagehdu`` functions. [#527]

  - Added new ``show_progress`` keyword to the ``load_*`` functions.
    [#590]

- ``photutils.isophote``

  - Added a new ``photutils.isophote`` subpackage to provide tools to
    fit elliptical isophotes to a galaxy image. [#532, #603]

- ``photutils.segmentation``

  - Added a ``cmap`` method to ``SegmentationImage`` to generate a
    random matplotlib colormap. [#513]

  - Added ``sky_centroid`` and ``sky_centroid_icrs`` source
    properties. [#592]

  - Added new source properties representing the sky coordinates of
    the bounding box corner vertices (``sky_bbox_ll``, ``sky_bbox_ul``
    ``sky_bbox_lr``, and ``sky_bbox_ur``). [#592]

  - Added new ``SourceCatalog`` class to hold the list of
    ``SourceProperties``. [#608]

  - The ``properties_table`` function is now deprecated.  Use the
    ``SourceCatalog.to_table()`` method instead. [#608]

- ``photutils.psf``

  - Uncertainties on fitted parameters are added to the final table. [#516]

  - Fitted results of any free parameter are added to the final table. [#471]


API changes
^^^^^^^^^^^

- ``photutils.aperture``

  - The ``ApertureMask`` ``apply()`` method has been renamed to
    ``multiply()``. [#481].

  - The ``ApertureMask`` input parameter was renamed from ``mask`` to
    ``data``. [#548]

  - Removed the ``pixelwise_errors`` keyword from
    ``aperture_photometry``. [#489]

- ``photutils.background``

  - The ``Background2D`` keywords ``exclude_mesh_method`` and
    ``exclude_mesh_percentile`` were removed in favor of a single
    keyword called ``exclude_percentile``. [#544]

  - Renamed ``BiweightMidvarianceBackgroundRMS`` to
    ``BiweightScaleBackgroundRMS``. [#547]

  - Removed the ``SigmaClip`` class.  ``astropy.stats.SigmaClip`` is
    a direct replacement. [#569]

- ``photutils.datasets``

  - The ``make_poission_noise`` function was renamed to
    ``apply_poisson_noise``.  [#527]

  - The ``make_random_gaussians`` function was renamed to
    ``make_random_gaussians_table``.  The parameter ranges
    must now be input as a dictionary.  [#527]

  - The ``make_gaussian_sources`` function was renamed to
    ``make_gaussian_sources_image``. [#527]

  - The ``make_random_models`` function was renamed to
    ``make_random_models_table``. [#527]

  - The ``make_model_sources`` function was renamed to
    ``make_model_sources_image``. [#527]

  - The ``unit``, ``hdu``, ``wcs``, and ``wcsheader`` keywords in
    ``photutils.datasets`` functions were removed. [#527]

  - ``'photutils-datasets'`` was added as an optional ``location`` in
    the ``get_path`` function. This option is used as a fallback in
    case the ``'remote'`` location (astropy data server) fails.
    [#589]

- ``photutils.detection``

  - The ``daofind`` and ``irafstarfinder`` functions were removed.
    [#588]

- ``photutils.psf``

  - ``IterativelySubtractedPSFPhotometry`` issues a "no sources
    detected" warning only on the first iteration, if applicable.
    [#566]

- ``photutils.segmentation``

  - The ``'icrs_centroid'``, ``'ra_icrs_centroid'``, and
    ``'dec_icrs_centroid'`` source properties are deprecated and are no
    longer default columns returned by ``properties_table``. [#592]

  - The ``properties_table`` function now returns a ``QTable``. [#592]

- ``photutils.utils``

  - The ``background_color`` keyword was removed from the
    ``random_cmap`` function. [#528]

  - Deprecated unused ``interpolate_masked_data()``. [#526, #611]

Bug Fixes
^^^^^^^^^

- ``photutils.segmentation``

  - Fixed ``deblend_sources`` so that it correctly deblends multiple
    sources. [#572]

  - Fixed a bug in calculation of the ``sky_centroid_icrs`` (and
    deprecated ``icrs_centroid``) property where the incorrect pixel
    origin was being passed. [#592]

- ``photutils.utils``

  - Added a check that ``data`` and ``bkg_error`` have the same units
    in ``calc_total_error``. [#537]


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
