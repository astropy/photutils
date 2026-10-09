# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This plugin provides customization of the header displayed by pytest for
reporting purposes.

PYTEST_DONT_REWRITE

"""
import datetime
import importlib
import locale
import os
import sys

TESTED_VERSIONS = {}
DEFAULT_PACKAGES = ("Numpy", "Scipy", "Matplotlib", "h5py", "Pandas")
PYTEST_HEADER_MODULES = {pkg: pkg.lower() for pkg in DEFAULT_PACKAGES}

# This import is needed so astropy can use pytest-astropy-header.
# See https://github.com/astropy/pytest-astropy-header/issues/41
try:
    import astropy  # noqa: F401
except ImportError:
    pass


def pytest_addoption(parser):

    group = parser.getgroup("astropy header options")
    group.addoption('--astropy-header', action='store_true',
                    help="Show the pytest-astropy header")
    group.addoption('--astropy-header-packages', default=None,
                    help="Comma-separated list of packages to include in the header")
    parser.addini('astropy_header', type="bool",
                  help="Show the pytest-astropy header")
    parser.addini('astropy_header_packages', type='linelist',
                  help="Comma-separated list of packages to include in the header")


def pytest_report_header(config):

    if not config.getoption("astropy_header") and not config.getini("astropy_header"):
        return

    astropy_header_packages_option = config.getoption("astropy_header_packages")
    astropy_header_packages_ini = config.getini("astropy_header_packages")

    if astropy_header_packages_option is not None:
        if isinstance(astropy_header_packages_option, str):
            astropy_header_packages_option = [x.strip() for x in astropy_header_packages_option.split(',')]
        packages_to_display = {x: x for x in astropy_header_packages_option}
    elif len(astropy_header_packages_ini) > 0:
        if len(astropy_header_packages_ini) == 1:
            astropy_header_packages_ini = [x.strip() for x in astropy_header_packages_ini[0].split(',')]
        packages_to_display = {x: x for x in astropy_header_packages_ini}
    else:
        packages_to_display = PYTEST_HEADER_MODULES

    try:
        stdoutencoding = sys.stdout.encoding or 'ascii'
    except AttributeError:
        stdoutencoding = 'ascii'

    args = config.args

    s = ''
    for pkg, version in TESTED_VERSIONS.items():
        s += f"\nRunning tests with {pkg} version {version}.\n"

    # Per https://github.com/astropy/astropy/pull/4204, strip the rootdir from
    # each directory argument
    if hasattr(config, 'rootdir'):
        rootdir = str(config.rootdir)
        if not rootdir.endswith(os.sep):
            rootdir += os.sep

        dirs = [arg[len(rootdir):] if arg.startswith(rootdir) else arg
                for arg in args]
    else:
        dirs = args

    s += "Running tests in {}.\n\n".format(" ".join(dirs))

    s += "Date: {}\n\n".format(datetime.datetime.now().isoformat()[:19])

    import warnings
    from platform import platform
    plat = platform()
    if isinstance(plat, bytes):
        plat = plat.decode(stdoutencoding, 'replace')
    s += f"Platform: {plat}\n\n"
    s += f"Executable: {sys.executable}\n\n"
    s += f"Full Python Version: \n{sys.version}\n\n"

    s += "encodings: sys: {}, locale: {}, filesystem: {}".format(
        sys.getdefaultencoding(),
        locale.getpreferredencoding(),
        sys.getfilesystemencoding())
    s += '\n'

    s += f"byteorder: {sys.byteorder}\n"
    s += "float info: dig: {0.dig}, mant_dig: {0.dig}\n\n".format(
        sys.float_info)

    s += "Package versions: \n"

    for module_display, module_name in packages_to_display.items():
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=DeprecationWarning)
                module = importlib.import_module(module_name)
        except ImportError:
            s += f"{module_display}: not available\n"
        else:
            try:
                version = module.__version__
            except AttributeError:
                version = 'unknown (no __version__ attribute)'
            s += f"{module_display}: {version}\n"

    s += "\n"

    opts = []
    for op in ("remote_data", "pep8", "run_slow", "run_hugemem"):
        op_value = getattr(config.option, op, None)
        if op_value:
            if isinstance(op_value, str):
                op = ': '.join((op, op_value))
            opts.append(op)
    if opts:
        s += "Using Astropy options: {}.\n".format(", ".join(opts))

    return s
