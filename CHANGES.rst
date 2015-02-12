0.2 (unreleased)
----------------

General
^^^^^^^


New Features
^^^^^^^^^^^^

- ``photutils.detection``

  - ``find_peaks`` now returns an Astropy Table containing the (x, y)
    positions and peak values. [#240]

  - ``find_peaks`` has new ``mask``, ``error``, ``wcs`` and ``subpixel``
    precision options. [#244]

- ``photutils.morphology``

  - Added new ``GaussianConst1D`` and ``GaussianConst2D`` models [#244].

  - Added new ``marginalize_data2d`` function [#244].

  - Added new ``cutout_footprint`` function [#244].
