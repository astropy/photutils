************
Installation
************

Requirements
============

Photutils has the following strict requirements:

* `Python <https://www.python.org/>`_ 3.7 or later

* `Numpy <https://numpy.org/>`_ 1.17 or later

* `Astropy`_ 4.0 or later

Photutils also optionally depends on other packages for some features:

* `Scipy <https://www.scipy.org/>`_ 1.6.0 or later:  To power a variety of
  features in several modules (strongly recommended).

* `matplotlib <https://matplotlib.org/>`_ 2.2 or later:  To power a
  variety of plotting features (e.g., plotting apertures).

* `scikit-image <https://scikit-image.org/>`_ 0.14.2 or later:  Used in
  `~photutils.segmentation.deblend_sources` for deblending segmented
  sources.

* `scikit-learn <https://scikit-learn.org/>`_ 0.19 or later:  Used in
  `~photutils.psf.DBSCANGroup` to create star groups.

* `gwcs <https://github.com/spacetelescope/gwcs>`_ 0.12 or later:
  Used in `~photutils.datasets.make_gwcs` to create a simple celestial
  gwcs object.

* `bottleneck <https://github.com/pydata/bottleneck>`_: Improves the
  performance of sigma clipping and other functionality that may require
  computing statistics on arrays with NaN values.

Photutils depends on `pytest-astropy
<https://github.com/astropy/pytest-astropy>`_ (0.4 or later) to run
the test suite.


Installing the latest released version
======================================

The latest released (stable) version of Photutils can be installed
either with `pip`_ or `conda`_.

Using pip
---------

To install Photutils with `pip`_, run::

    pip install photutils

If you want to make sure that none of your existing dependencies get
upgraded, instead you can do::

    pip install photutils --no-deps

Note that you may need a C compiler (e.g., ``gcc`` or ``clang``) to be
installed for the installation to succeed.

If you get a ``PermissionError``, this means that you do not have the
required administrative access to install new packages to your Python
installation.  In this case you may consider using the ``--user``
option to install the package into your home directory.  You can read
more about how to do this in the `pip documentation
<https://pip.pypa.io/en/stable/user_guide/#user-installs>`_.

Do **not** install Photutils or other third-party packages using
``sudo`` unless you are fully aware of the risks.

Using conda
-----------

Photutils can be installed with `conda`_ if you have installed
`Anaconda <https://www.anaconda.com/products/individual>`_ or
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.  To
install Photutils using the `conda-forge Anaconda channel
<https://anaconda.org/conda-forge/photutils>`_, run::

    conda install -c conda-forge photutils


Installing the latest development version from Source
=====================================================

Prerequisites
-------------

You will need `Cython <https://cython.org/>`_ (0.28 or later), a
compiler suite, and the development headers for Python and Numpy in
order to build Photutils from the source distribution.  On Linux,
using the package manager for your distribution will usually be the
easiest route.

On MacOS X you will need the `XCode`_ command-line tools, which can be
installed using::

    xcode-select --install

Follow the onscreen instructions to install the command-line tools
required.  Note that you do not need to install the full `XCode`_
distribution (assuming you are using MacOS X 10.9 or later).


Building and installing manually
--------------------------------

Photutils is being developed on `GitHub`_.  The latest development
version of the Photutils source code can be retrieved using git::

    git clone https://github.com/astropy/photutils.git

Then to build and install Photutils, run::

    cd photutils
    pip install .[all]

If you wish to install the package in "editable" mode, instead include
the "-e" option::

    pip install -e .[all]


Building and installing using pip
---------------------------------

Alternatively, `pip`_ can be used to retrieve, build, and install the
latest development version from `GitHub`_::

    pip install git+https://github.com/astropy/photutils.git

Again, if you want to make sure that none of your existing
dependencies get upgraded, instead you can do::

    pip install git+https://github.com/astropy/photutils.git --no-deps


Testing an installed Photutils
==============================

The easiest way to test your installed version of Photutils is running
correctly is to use the :func:`photutils.test` function:

.. doctest-skip::

    >>> import photutils
    >>> photutils.test()

Note that this may not work if you start Python from within the
Photutils source distribution directory.

The tests should run and report any failures, which you can report to
the `Photutils issue tracker
<https://github.com/astropy/photutils/issues>`_.


.. _pip: https://pip.pypa.io/en/latest/
.. _conda: https://docs.conda.io/en/latest/
.. _GitHub: https://github.com/astropy/photutils
.. _Xcode: https://developer.apple.com/xcode/
