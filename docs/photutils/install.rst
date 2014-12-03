************
Installation
************

Requirements
============

Photutils has the following strict requirements:

* `Python <http://www.python.org/>`_ 2.6 (>=2.6.5), 2.7, 3.3, or 3.4

* `Numpy <http://www.numpy.org/>`_ 1.6 or later

* `Astropy`_ 0.4 or later

You will also need `Cython`_ (0.15 or later) installed to build
photutils from source, unless you are installing a numbered release
(see :ref:`sourceinstall` below).

Some functionality is available only if the following optional
dependencies are installed:

* `Scipy`_: To power a variety of features including background
  estimation, source detection, source segmentation properties, and
  fitting.

* `scikit-image`_:  To power a variety of features including source
  detection and source morphological properties.

* `matplotlib <http://matplotlib.org/>`_:  For plotting functions.

.. warning::

    While photutils will import even if these dependencies are not
    installed, the functionality will be severely limited.  It is very
    strongly recommended that you install `Scipy`_ and `scikit-image`_
    to use photutils.  Both are easy installed via `pip`_ or `conda`_.

.. _Scipy: http://www.scipy.org/
.. _scikit-image: http://scikit-image.org/
.. _pip: https://pip.pypa.io/en/latest/
.. _conda: http://conda.pydata.org/docs/


Installing Photutils using pip
==============================

To install the latest photutils **stable** version with `pip`_, simply
run::

    pip install --no-deps photutils

To install the current photutils **development** version using
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

    Do **not** install photutils or other third-party packages using
    ``sudo`` unless you are fully aware of the risks.


.. _sourceinstall:

Installing Photutils from source
================================

Prerequisites
-------------

You will need a compiler suite and the development headers for Python
and Numpy in order to build photutils.  On Linux, using the package
manager for your distribution will usually be the easiest route, while
on MacOS X you will need the XCode command line tools.

The `instructions for building Numpy from source
<http://docs.scipy.org/doc/numpy/user/install.html>`_ are also a good
resource for setting up your environment to build Python packages.

You will also need `Cython`_ (0.15 or later) installed to build from
source, unless you are installing a numbered release. (The releases
packages have the necessary C files packaged with them, and hence do
not require Cython.)

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


Obtaining the source package
----------------------------

Stable version
^^^^^^^^^^^^^^

The latest stable source package for photutils can be `downloaded here
<https://pypi.python.org/pypi/photutils>`_.

Development version
^^^^^^^^^^^^^^^^^^^

The latest development version of photutils can be cloned from github
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

To build photutils (from the root of the source tree)::

    python setup.py build

To install photutils (from the root of the source tree)::

    python setup.py install


.. _sourcebuildtest:

Testing a source code build of Photutils
----------------------------------------

The easiest way to test that your photutils built correctly (without
installing photutils) is to run this from the root of the source
tree::

    python setup.py test

See the Astropy documentation for alternative methods of
:ref:`running-tests`.


Testing an installed Photutils
==============================

The easiest way to test your installed version of photutils is running
correctly is to use the :func:`photutils.test()` function:

.. doctest-skip::

    >>> import photutils
    >>> photutils.test()

The tests should run and print out any failures, which you can report
at the `Photutils issue tracker
<http://github.com/astropy/photutils/issues>`_.

.. note::

    This way of running the tests may not work if you do it in the
    photutils source distribution.  See :ref:`sourcebuildtest` for how
    to run the tests from the source code directory.

.. note::

    Running the tests this way is currently disabled in the IPython
    REPL due to conflicts with some common display settings in
    IPython.  Please run the photutils tests under the standard Python
    command-line interpreter.


.. _Cython: http://cython.org
