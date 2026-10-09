"""Wrapper, ERFA and SOFA version information."""

# NOTE: First try _dev.scm_version if it exists and setuptools_scm is installed
# This file is not included in pyerfa wheels/tarballs, so otherwise it will
# fall back on the generated _version module.
try:
    try:
        from ._dev.scm_version import get_version as _get_version
        _version = _get_version()
        del _get_version
    except ImportError:
        from ._version import version as _version
except Exception:
    import warnings
    warnings.warn(
        f'could not determine {__name__.split(".")[0]} package version; '
        f'this indicates a broken installation')
    del warnings

    _version = '0.0.0'

# Set the version numbers a bit indirectly, so that Sphinx can pick up
# up the docstrings and list the values.
try:
    from . import ufunc
except ImportError as exc:
    # If compiled to use a system liberfa, that library can be too old, and
    # miss functions that are available in newer liberfa. If so, we should
    # bail since nothing will work, but let's try to give a more informative
    # error message.
    try:
        from ctypes import CDLL, util, c_char_p
        liberfa = CDLL(util.find_library('erfa'))
        liberfa.eraVersion.restype = c_char_p
        erfa_version = liberfa.eraVersion().decode('ascii')
    except Exception:
        pass
    else:
        if erfa_version.split(".")[:2] < _version.split(".")[:2]:
            raise ImportError(
                f"liberfa {erfa_version} too old for PyERFA {_version}. "
                "This should only be possible if you are using a system liberfa; "
                "try installing using 'pip install pyerfa', with environment variable "
                "PYERFA_USE_SYSTEM_LIBERFA unset or 0.") from exc

    raise


version = _version
'''Version of the python wrappers.'''

erfa_version = ufunc.erfa_version
'''Version of the C ERFA library that is wrapped.'''

sofa_version = ufunc.sofa_version
'''Version of the SOFA library the ERFA library is based on.'''

del ufunc, _version
