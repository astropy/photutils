************
Installation
************

Requirements
============

Photutils has the following strict requirements:

* `Python <http://www.python.org/>`_ 2.7, 3.3, 3.4 or 3.5

* `Numpy <http://www.numpy.org/>`_ 1.6 or later

* `Astropy`_ 1.0 or later

You will also need `Cython`_ (0.15 or later) to build Photutils from
source, unless you are installing a numbered release (see
:ref:`sourceinstall` below).

Some functionality is available only if the following optional
dependencies are installed:

* `Scipy`_: To power a variety of features including background
  estimation, source detection, source segmentation properties, and
  fitting.

* `scikit-image`_:  To power a variety of features including source
  detection and source morphological properties.

* `matplotlib <http://matplotlib.org/>`_:  For plotting functions.

.. warning::

    While Photutils will import even if these dependencies are not
    installed, the functionality will be severely limited.  It is very
    strongly recommended that you install `Scipy`_ and `scikit-image`_
    to use Photutils.  Both are easily installed via `pip`_ or
    `conda`_.

.. _Scipy: http://www.scipy.org/
.. _scikit-image: http://scikit-image.org/
.. _pip: https://pip.pypa.io/en/latest/
.. _conda: http://conda.pydata.org/docs/
.. _Cython: http://cython.org


Installing Photutils Using pip
==============================

To install the latest Photutils **stable** version with `pip`_, simply
run::

    pip install --no-deps photutils

To install the current Photutils **development** version using
`pip`_::

    pip install --no-deps git+https://github.com/astropy/photutils.git

.. note::

    You will need a C compiler (e.g. ``gcc`` or ``clang``) to be
    installed (see :ref:`sourceinstall` below) for the installation to
    succeed.

.. note::

    The ``--no-deps`` flag is optional, but highly recommended if you
    already have Numpy and Astropy installed, since otherwise pip will
    sometimes try to "help" you by upgrading your Numpy and Astropy
    installations, which may not always be desired.

.. note::

    If you get a ``PermissionError`` this means that you do not have
    the required administrative access to install new packages to your
    Python installation.  In this case you may consider using the
    ``--user`` option to install the package into your home directory.
    You can read more about how to do this in the `pip documentation
    <http://www.pip-installer.org/en/1.2.1/other-tools.html#using-pip-with-the-user-scheme>`_.

    Do **not** install Photutils or other third-party packages using
    ``sudo`` unless you are fully aware of the risks.


.. _sourceinstall:

Installing Photutils from Source
================================

Prerequisites
-------------

You will need a compiler suite and the development headers for Python
and Numpy in order to build Photutils.  On Linux, using the package
manager for your distribution will usually be the easiest route, while
on MacOS X you will need the XCode command line tools.

The `instructions for building Numpy from source
<http://docs.scipy.org/doc/numpy/user/install.html>`_ are also a good
resource for setting up your environment to build Python packages.

You will also need `Cython`_ (0.15 or later) to build from source,
unless you are installing a numbered release. (The released packages
have the necessary C files packaged with them, and hence do not
require Cython.)

.. note::

    If you are using MacOS X, you will need to the XCode command line
    tools.  One way to get them is to install `XCode
    <https://developer.apple.com/xcode/>`_. If you are using OS X 10.7
    (Lion) or later, you must also explicitly install the command line
    tools. You can do this by opening the XCode application, going to
    **Preferences**, then **Downloads**, and then under
    **Components**, click on the Install button to the right of
    **Command Line Tools**.  Alternatively, on 10.7 (Lion) or later,
    you do not need to install XCode, you can download just the
    command line tools from
    https://developer.apple.com/downloads/index.action (requires an
    Apple developer account).


Obtaining the Source Package
----------------------------

Stable Version
^^^^^^^^^^^^^^

The latest stable source package for Photutils can be `downloaded here
<https://pypi.python.org/pypi/photutils>`_.

Development Version
^^^^^^^^^^^^^^^^^^^

The latest development version of Photutils can be cloned from github
using this command::

   git clone git://github.com/astropy/photutils.git


Building and Installing
-----------------------

Photutils uses the Python `distutils framework
<http://docs.python.org/install/index.html>`_ for building and
installing and requires the `distribute
<http://pypi.python.org/pypi/distribute>`_ extension--the later is
automatically downloaded when running ``python setup.py`` if it is not
already provided by your system.

Numpy and Astropy must already installed in your Python environment.

To build Photutils (from the root of the source tree)::

    python setup.py build

To install Photutils (from the root of the source tree)::

    python setup.py install


.. _sourcebuildtest:

Testing a Source-Code Build of Photutils
----------------------------------------

The easiest way to test that your Photutils built correctly (without
installing Photutils) is to run this from the root of the source
tree::

    python setup.py test

See the Astropy documentation for alternative methods of
:ref:`running-tests`.


Testing an Installed Photutils
==============================

The easiest way to test your installed version of Photutils is running
correctly is to use the :func:`photutils.test()` function:

.. doctest-skip::

    >>> import photutils
    >>> photutils.test()

The tests should run and print out any failures, which you can report
to the `Photutils issue tracker
<http://github.com/astropy/photutils/issues>`_.

.. note::

    This way of running the tests may not work if you do it in the
    Photutils source distribution directory.  See
    :ref:`sourcebuildtest` for how to run the tests from the source
    code directory.
