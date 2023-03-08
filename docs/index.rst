
.. the "raw" directive below is used to hide the title in favor of
   just the logo being visible
.. raw:: html

    <style media="screen" type="text/css">
      h1 {display:none;}
    </style>

.. |br| raw:: html

    <div style="min-height:0.1em;"></div>

*********
Photutils
*********

.. raw:: html

   <img src="_static/photutils_banner.svg" onerror="this.src='_static/photutils_banner-475x120.png'; this.onerror=null;" width="495"/>

.. only:: latex

    .. image:: _static/photutils_banner.pdf


**Photutils** is an  `affiliated package
<https://www.astropy.org/affiliated/index.html>`_ of `Astropy`_ that
primarily provides tools for detecting and performing photometry of
astronomical sources.  It is an open source Python package and is
licensed under a :ref:`3-clause BSD license <photutils_license>`.

|br|

.. Important::
    If you use Photutils for a project that leads to a publication,
    whether directly or as a dependency of another package, please
    include an :doc:`acknowledgment and/or citation <citation>`.

|br|

Getting Started
===============

.. toctree::
    :maxdepth: 1

    install.rst
    whats_new/index.rst
    overview.rst
    pixel_conventions.rst
    getting_started.rst
    contributing.rst
    citation.rst
    license.rst
    changelog


User Documentation
==================

.. toctree::
    :maxdepth: 1

    background.rst
    detection.rst
    grouping.rst
    aperture.rst
    psf.rst
    epsf.rst
    psf_matching.rst
    segmentation.rst
    centroids.rst
    profiles.rst
    morphology.rst
    isophote.rst
    geometry.rst
    datasets.rst
    utils.rst


Developer Documentation
=======================

.. toctree::
    :maxdepth: 1

    dev/releasing.rst


.. toctree::
    :hidden:

    test_function.rst
