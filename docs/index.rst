
.. the "raw" directive below is used to hide the title in favor of
   just the logo being visible
.. raw:: html

    <style media="screen" type="text/css">
      h1 {display:none;}
    </style>

*********
Photutils
*********

.. raw:: html

   <img src="_static/photutils_banner.svg" onerror="this.src='_static/photutils_banner-475x120.png'; this.onerror=null;" width="495"/>

.. only:: latex

    .. image:: _static/photutils_banner.pdf


**Photutils** is an  `affiliated package
<http://www.astropy.org/affiliated/index.html>`_ of `Astropy`_ to
provide tools for detecting and performing photometry of astronomical
sources.  It is an open source (BSD licensed) Python package.  Bug
reports, comments, and help with development are very welcome.


Photutils at a glance
=====================

.. toctree::
    :maxdepth: 1

    photutils/install.rst
    photutils/overview.rst
    photutils/getting_started.rst
    changelog


User Documentation
==================

.. toctree::
    :maxdepth: 1

    photutils/background.rst
    photutils/detection.rst
    photutils/grouping.rst
    photutils/aperture.rst
    photutils/psf.rst
    photutils/psf_matching.rst
    photutils/segmentation.rst
    photutils/centroids.rst
    photutils/morphology.rst
    photutils/geometry.rst
    photutils/isophote.rst
    photutils/datasets.rst
    photutils/utils.rst

.. toctree::
    :maxdepth: 1

    photutils/high-level_API.rst

.. note::

    Like much astronomy software, Photutils is an evolving package.
    The developers make an effort to maintain backwards compatibility,
    but at times the API may change if there is a benefit to doing so.
    If there are specific areas you think API stability is important,
    please let us know as part of the development process!


Reporting Issues
================

If you have found a bug in Photutils please report it by creating a
new issue on the `Photutils GitHub issue tracker
<https://github.com/astropy/photutils/issues>`_.

Please include an example that demonstrates the issue that will allow
the developers to reproduce and fix the problem. You may be asked to
also provide information about your operating system and a full Python
stack trace.  The developers will walk you through obtaining a stack
trace if it is necessary.

Photutils uses a package of utilities called `astropy-helpers
<https://github.com/astropy/astropy-helpers>`_ during building and
installation.  If you have any build or installation issue mentioning
the ``astropy_helpers`` or ``ah_bootstrap`` modules please send a
report to the `astropy-helpers issue tracker
<https://github.com/astropy/astropy-helpers/issues>`_.  If you are
unsure, then it's fine to report to the main Photutils issue tracker.


Contributing
============

Like the `Astropy`_ project, Photutils is made both by and for its
users.  We accept contributions at all levels, spanning the gamut from
fixing a typo in the documentation to developing a major new feature.
We welcome contributors who will abide by the `Python Software
Foundation Code of Conduct
<https://www.python.org/psf/codeofconduct/>`_.

Photutils follows the same workflow and coding guidelines as
`Astropy`_.  The following pages will help you get started with
contributing fixes, code, or documentation (no git or GitHub
experience necessary):

* `How to make a code contribution <http://astropy.readthedocs.io/en/stable/development/workflow/development_workflow.html>`_

* `Coding Guidelines <http://docs.astropy.org/en/latest/development/codeguide.html>`_

* `Try the development version <http://astropy.readthedocs.io/en/stable/development/workflow/get_devel_version.html>`_

* `Developer Documentation <http://docs.astropy.org/en/latest/#developer-documentation>`_

Citing Photutils
================

If you use Photutils, please consider citing the package via its Zenodo record.
If you just want the latest release, cite this (follow the link on the badge
and then use one of the citation methods on the right):

.. image:: https://zenodo.org/badge/2640766.svg
    :target: https://zenodo.org/badge/latestdoi/2640766

If you want to cite an earlier version, you can
`search for photutils on Zenodo <https://zenodo.org/search?q=photutils>`_.  Then
cite the Zenodo DOI for whatever version(s) of Photutils you are using.
