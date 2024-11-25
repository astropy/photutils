************
Installation
************

Requirements
============

Photutils has the following strict requirements:

* `Python <https://www.python.org/>`_ 3.11 or later

* `NumPy <https://numpy.org/>`_ 1.24 or later

* `Astropy`_ 5.3 or later

* `SciPy <https://scipy.org/>`_ 1.10 or later

Photutils also optionally depends on other packages for some features:

* `Matplotlib <https://matplotlib.org/>`_ 3.7 or later: Used to power a
  variety of plotting features (e.g., plotting apertures).

* `Regions <https://astropy-regions.readthedocs.io/>`_ 0.9 or
  later: Required to perform aperture photometry using region objects.

* `scikit-image <https://scikit-image.org/>`_ 0.20 or later: Required
  to deblend segmented sources.

* `GWCS <https://gwcs.readthedocs.io/en/stable/>`_ 0.20 or later:
  Required in `~photutils.datasets.make_gwcs` to create a simple celestial
  gwcs object.

* `Bottleneck <https://github.com/pydata/bottleneck>`_: Improves the
  performance of sigma clipping and other functionality that may require
  computing statistics on arrays with NaN values.

* `tqdm <https://tqdm.github.io/>`_: Required to display optional
  progress bars.

* `Rasterio <https://rasterio.readthedocs.io/en/stable/>`_: Required to
  convert source segments into polygon objects.

* `Shapely <https://shapely.readthedocs.io/en/stable/>`_: Required to
  convert source segments into polygon objects.


Installing the latest released version
======================================

Using pip
---------

To install Photutils with `pip`_, run::

    pip install photutils

If you want to install Photutils along with all of its optional
dependencies, you can instead run::

    pip install "photutils[all]"

In most cases, this will install a pre-compiled version (called a wheel)
of Photutils, but if you are using a very recent version of Python
or if you are installing Photutils on a platform that is not common,
Photutils will be installed from a source file. In this case you will
need a C compiler (e.g., ``gcc`` or ``clang``) to be installed for the
installation to succeed (see :ref:`building_source` prerequisites).

If you get a ``PermissionError``, this means that you do not have the
required administrative access to install new packages to your Python
installation.  In this case you may consider using the ``--user``
option to install the package into your home directory.  You can read
more about how to do this in the `pip documentation
<https://pip.pypa.io/en/stable/user_guide/#user-installs>`_.


Using conda
-----------

Photutils can also be installed using the ``conda`` package manager.
There are several methods for installing ``conda`` and many different
ways to set up your Python environment, but that is beyond the
scope of this documentation. We recommend installing `miniforge
<https://github.com/conda-forge/miniforge>`__.

Once you have installed ``conda``, you can install Photutils using the
``conda-forge`` channel::

    conda install -c conda-forge photutils


.. _building_source:

Building from Source
====================

Prerequisites
-------------

You will need a compiler suite and the development headers for Python
and Numpy in order to build Photutils from the source distribution. You
do not need to install any other specific build dependencies (such as
Cython) since these will be automatically installed into a temporary
build environment by `pip`_.

On Linux, using the package manager for your distribution will usually be
the easiest route.

On macOS you will need the `XCode`_ command-line tools, which can be
installed using::

    xcode-select --install

Follow the onscreen instructions to install the command-line tools
required.  Note that you do not need to install the full `XCode`_
distribution (assuming you are using MacOS X 10.9 or later).


Installing the development version
----------------------------------

Photutils is being developed on `GitHub`_.  The latest development
version of the Photutils source code can be retrieved using git::

    git clone https://github.com/astropy/photutils.git

Then to build and install Photutils (with all of its optional
dependencies), run::

    cd photutils
    pip install ".[all]"

If you wish to install the package in "editable" mode, instead include
the "-e" option::

    pip install -e ".[all]"

Alternatively, `pip`_ can be used to retrieve and install the
latest development pre-built wheel::

    pip install --upgrade --extra-index-url https://pypi.anaconda.org/astropy/simple "photutils[all]" --pre

Testing an installed Photutils
==============================

To test your installed version of Photutils, you can run the test suite
using the `pytest`_ command. Running the test suite requires installing
the `pytest-astropy <https://github.com/astropy/pytest-astropy>`_ (0.11
or later) package.

To run the test suite, use the following command::

    pytest --pyargs photutils

Any test failures can be reported to the `Photutils issue tracker
<https://github.com/astropy/photutils/issues>`_.


.. _pip: https://pip.pypa.io/en/latest/
.. _GitHub: https://github.com/astropy/photutils
.. _Xcode: https://developer.apple.com/xcode/
.. _pytest: https://docs.pytest.org/en/latest/
