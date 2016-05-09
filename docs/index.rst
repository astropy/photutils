
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


Photutils at a Glance
=====================

**Photutils** is an in-development `affiliated package
<http://www.astropy.org/affiliated/index.html>`_ of `Astropy`_ to
provide tools for detecting and performing photometry of astronomical
sources.  It is an open source (BSD licensed) Python package.  Bug
reports, comments, and help with development are very welcome.

.. toctree::
    :maxdepth: 2

    photutils/install.rst
    photutils/overview.rst
    photutils/getting_started.rst
    changelog

.. note::

    Photutils is still under development and has not seen widespread
    use yet.  We will change its API if we find that something can be
    improved.


User Documentation
==================

.. toctree::
    :maxdepth: 1

    photutils/background.rst
    photutils/detection.rst
    photutils/aperture.rst
    photutils/psf.rst
    photutils/segmentation.rst
    photutils/morphology.rst
    photutils/geometry.rst
    photutils/datasets.rst
    photutils/utils.rst
    photutils/high-level_API.rst


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
