************
Installation
************

Requirements
============

Photutils has the following strict requirements:

* `Python <http://www.python.org/>`_ 2.7, 3.3, 3.4 or 3.5

* `Numpy <http://www.numpy.org/>`_ 1.7 or later

* `Astropy`_ 1.0 or later

Additionally, some functionality is available only if the following
optional dependencies are installed:

* `Scipy`_ 0.15 or later

* `scikit-image`_ 0.11 or later

* `matplotlib <http://matplotlib.org/>`_ 1.3 or later

.. warning::

    While Photutils will import even if these dependencies are not
    installed, the functionality will be severely limited.  It is very
    strongly recommended that you install `Scipy`_ and `scikit-image`_
    to use Photutils.  Both are easily installed via `pip`_ or
    `conda`_.


Installing the latest released version
======================================

The latest released (stable) version of Photutils can be installed
either with `conda`_ or `pip`_.


Using conda
-----------

Photutils can be installed with `conda`_ using the `astropy Anaconda
channel <https://anaconda.org/astropy>`_::

    conda install -c astropy photutils


Using pip
---------

To install using `pip`_, simply run::

    pip install --no-deps photutils

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


Installing the latest development version
=========================================


Prerequisites
-------------

You will need `Cython`_ (0.15 or later), a compiler suite, and the
development headers for Python and Numpy in order to build Photutils
from the source distribution.  On Linux, using the package manager for
your distribution will usually be the easiest route, while on MacOS X
you will need the XCode command line tools.

The `instructions for building Numpy from source
<http://docs.scipy.org/doc/numpy/user/install.html>`_ are also a good
resource for setting up your environment to build Python packages.

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


Building and installing Manually
--------------------------------

Photutils is being developed on `github`_.  The latest development
version of the Photutils source code can be retrieved using git::

    git clone https://github.com/astropy/photutils.git

Then, to build and install Photutils (from the root of the source
tree)::

    cd photutils
    python setup.py install


Building and installing using pip
---------------------------------

Alternatively, `pip`_ can be used to retrieve, build, and install the
latest development version from `github`_::

    pip install --no-deps git+https://github.com/astropy/photutils.git

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


Testing an installed Photutils
==============================

The easiest way to test your installed version of Photutils is running
correctly is to use the :func:`photutils.test()` function:

.. doctest-skip::

    >>> import photutils
    >>> photutils.test()

The tests should run and report any failures, which you can report to
the `Photutils issue tracker
<http://github.com/astropy/photutils/issues>`_.

.. note::

    This way of running the tests may not work if you start Python
    from within the Photutils source distribution directory.


.. _Scipy: http://www.scipy.org/
.. _scikit-image: http://scikit-image.org/
.. _pip: https://pip.pypa.io/en/latest/
.. _conda: http://conda.pydata.org/docs/
.. _Cython: http://cython.org
.. _github: https://github.com/astropy/photutils
