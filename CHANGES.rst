0.2 (unreleased)
----------------

General
^^^^^^^

- photutils now requires AstroPy v1.0 or later.

New Features
^^^^^^^^^^^^

- ``photutils.detection``

  - ``find_peaks`` now returns an Astropy Table containing the (x, y)
    positions and peak values. [#240]

  - ``find_peaks`` has new ``mask``, ``error``, ``wcs`` and ``subpixel``
    precision options. [#244]

- ``photutils.morphology``

  - Added new ``GaussianConst2D`` (2D Gaussian plus a constant) model
    [#244].

  - Added new ``marginalize_data2d`` function [#244].

  - Added new ``cutout_footprint`` function [#244].


Other Changes and Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Bundled copy of astropy-helpers upgraded to v1.0. [#251]

Bug Fixes
^^^^^^^^^

- ``photutils.geometry``

  - ``overlap_area_triangle_unit_circle`` handles correctly a corner case
    in some i386 systems where the area of the aperture was not computed
    correctly. [#242]
