# Copyright (c) 2016, Thomas P. Robitaille
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# This package was derived from pytest-mpl, which is released under a BSD
# license and can be found here:
#
#   https://github.com/astrofrog/pytest-mpl

import os
import abc
import shutil
import tempfile
import warnings
from urllib.request import urlopen

import pytest
import numpy as np


abstractstaticmethod = abc.abstractstaticmethod
abstractclassmethod = abc.abstractclassmethod


class BaseDiff(metaclass=abc.ABCMeta):

    @abstractstaticmethod
    def read(filename):
        """
        Given a filename, return a data object.
        """
        raise NotImplementedError()

    @abstractstaticmethod
    def write(filename, data, **kwargs):
        """
        Given a filename and a data object (and optional keyword arguments),
        write the data to a file.
        """
        raise NotImplementedError()

    @abstractclassmethod
    def compare(self, reference_file, test_file, atol=None, rtol=None):
        """
        Given a reference and test filename, compare the data to the specified
        absolute (``atol``) and relative (``rtol``) tolerances.

        Should return two arguments: a boolean indicating whether the data are
        identical, and a string giving the full error message if not.
        """
        raise NotImplementedError()


class SimpleArrayDiff(BaseDiff):

    @classmethod
    def compare(cls, reference_file, test_file, atol=None, rtol=None):

        array_ref = cls.read(reference_file)
        array_new = cls.read(test_file)

        try:
            np.testing.assert_allclose(array_ref, array_new, atol=atol, rtol=rtol)
        except AssertionError as exc:
            message = f"\n\na: {test_file}" + '\n'
            message += f"b: {reference_file}" + '\n'
            message += exc.args[0]
            return False, message
        else:
            return True, ""


class FITSDiff(BaseDiff):

    extension = 'fits'

    @staticmethod
    def read(filename):
        from astropy.io import fits
        return fits.getdata(filename)

    @staticmethod
    def write(filename, data, **kwargs):
        from astropy.io import fits
        if isinstance(data, np.ndarray):
            data = fits.PrimaryHDU(data)
        return data.writeto(filename, **kwargs)

    @classmethod
    def compare(cls, reference_file, test_file, atol=None, rtol=None):
        import astropy
        from astropy.io.fits.diff import FITSDiff
        from astropy.utils.introspection import minversion
        if minversion(astropy, '2.0'):
            diff = FITSDiff(reference_file, test_file, rtol=rtol, atol=atol)
        else:
            # `atol` is not supported prior to Astropy 2.0
            diff = FITSDiff(reference_file, test_file, tolerance=rtol)
        return diff.identical, diff.report()


class TextDiff(SimpleArrayDiff):

    extension = 'txt'

    @staticmethod
    def read(filename):
        return np.loadtxt(filename)

    @staticmethod
    def write(filename, data, **kwargs):
        fmt = kwargs.get('fmt', '%g')
        kwargs['fmt'] = fmt
        return np.savetxt(filename, data, **kwargs)


class PDHDFDiff(BaseDiff):

    extension = 'h5'

    @staticmethod
    def read(filename):
        import pandas as pd
        return pd.read_hdf(filename)

    @staticmethod
    def write(filename, data, **kwargs):
        import pandas as pd  # noqa: F401
        key = os.path.basename(filename).replace('.h5', '')
        return data.to_hdf(filename, key=key, **kwargs)

    @classmethod
    def compare(cls, reference_file, test_file, atol=None, rtol=None):
        import pandas.testing as pdt
        import pandas as pd

        ref_data = pd.read_hdf(reference_file)
        test_data = pd.read_hdf(test_file)
        try:
            pdt.assert_frame_equal(ref_data, test_data)
        except AssertionError as exc:
            message = f"\n\na: {test_file}" + '\n'
            message += f"b: {reference_file}" + '\n'
            message += exc.args[0]
            return False, message
        else:
            return True, ""


FORMATS = {}
FORMATS['fits'] = FITSDiff
FORMATS['text'] = TextDiff
FORMATS['pd_hdf'] = PDHDFDiff


def _download_file(url):
    u = urlopen(url)
    result_dir = tempfile.mkdtemp()
    filename = os.path.join(result_dir, 'downloaded')
    with open(filename, 'wb') as tmpfile:
        tmpfile.write(u.read())
    return filename


def pytest_addoption(parser):
    group = parser.getgroup("general")
    group.addoption('--arraydiff', action='store_true',
                    help="Enable comparison of arrays to reference arrays stored in files")
    group.addoption('--arraydiff-generate-path',
                    help="directory to generate reference files in, relative to location where py.test is run", action='store')
    group.addoption('--arraydiff-reference-path',
                    help="directory containing reference files, relative to location where py.test is run", action='store')
    group.addoption('--arraydiff-default-format',
                    help="Default format for the reference arrays (can be 'fits' or 'text' currently)")


def pytest_configure(config):
    config.getini('markers').append(
        'array_compare: for functions using array comparison')

    if config.getoption("--arraydiff") or config.getoption("--arraydiff-generate-path") is not None:

        reference_dir = config.getoption("--arraydiff-reference-path")
        generate_dir = config.getoption("--arraydiff-generate-path")

        if reference_dir is not None and generate_dir is not None:
            warnings.warn("Ignoring --arraydiff-reference-path since --arraydiff-generate-path is set")

        if reference_dir is not None:
            reference_dir = os.path.abspath(reference_dir)
        if generate_dir is not None:
            reference_dir = os.path.abspath(generate_dir)

        default_format = config.getoption("--arraydiff-default-format") or 'text'

        config.pluginmanager.register(ArrayComparison(config,
                                                      reference_dir=reference_dir,
                                                      generate_dir=generate_dir,
                                                      default_format=default_format))
    else:
        config.pluginmanager.register(ArrayInterceptor(config))


def generate_test_name(item):
    """
    Generate a unique name for this test.
    """
    if item.cls is not None:
        name = f"{item.module.__name__}.{item.cls.__name__}.{item.name}"
    else:
        name = f"{item.module.__name__}.{item.name}"
    return name


def wrap_array_interceptor(plugin, item):
    """
    Intercept and store arrays returned by test functions.
    """
    # Only intercept array on marked array tests
    if item.get_closest_marker('array_compare') is not None:

        # Guard against wrapping more than once (e.g. when pytest-run-parallel
        # runs the same item multiple times).
        if getattr(item.obj, '_arraydiff_wrapped', False):
            return

        # Use the full test name as a key to ensure correct array is being retrieved
        test_name = generate_test_name(item)

        def array_interceptor(store, obj):
            def wrapper(*args, **kwargs):
                store.return_value[test_name] = obj(*args, **kwargs)
            wrapper._arraydiff_wrapped = True
            return wrapper

        item.obj = array_interceptor(plugin, item.obj)


class ArrayComparison:

    def __init__(self, config, reference_dir=None, generate_dir=None, default_format='text'):
        self.config = config
        self.reference_dir = reference_dir
        self.generate_dir = generate_dir
        self.default_format = default_format
        self.return_value = {}

    def pytest_collection_modifyitems(self, items):
        for item in items:
            wrap_array_interceptor(self, item)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):

        compare = item.get_closest_marker('array_compare')

        if compare is None:
            yield
            return

        file_format = compare.kwargs.get('file_format', self.default_format)

        if file_format not in FORMATS:
            raise ValueError(f"Unknown format: {file_format}")

        if 'extension' in compare.kwargs:
            extension = compare.kwargs['extension']
        else:
            extension = FORMATS[file_format].extension

        atol = compare.kwargs.get('atol', 0.)
        rtol = compare.kwargs.get('rtol', 1e-7)

        single_reference = compare.kwargs.get('single_reference', False)

        write_kwargs = compare.kwargs.get('write_kwargs', {})

        reference_dir = compare.kwargs.get('reference_dir', None)
        if reference_dir is None:
            if self.reference_dir is None:
                reference_dir = os.path.join(os.path.dirname(item.fspath.strpath), 'reference')
            else:
                reference_dir = self.reference_dir
        else:
            if not reference_dir.startswith(('http://', 'https://')):
                reference_dir = os.path.join(os.path.dirname(item.fspath.strpath), reference_dir)

        baseline_remote = reference_dir.startswith('http')

        yield
        test_name = generate_test_name(item)
        if test_name not in self.return_value:
            # Test function did not complete successfully
            return
        array = self.return_value[test_name]

        # Find test name to use as plot name
        filename = compare.kwargs.get('filename', None)
        if filename is None:
            if single_reference:
                filename = item.originalname + '.' + extension
            else:
                filename = item.name + '.' + extension
                filename = filename.replace('[', '_').replace(']', '_')
                filename = filename.replace('_.' + extension, '.' + extension)

        # What we do now depends on whether we are generating the reference
        # files or simply running the test.
        if self.generate_dir is None:

            # Save the figure
            result_dir = tempfile.mkdtemp()
            test_array = os.path.abspath(os.path.join(result_dir, filename))

            FORMATS[file_format].write(test_array, array, **write_kwargs)

            # Find path to baseline array
            if baseline_remote:
                baseline_file_ref = _download_file(reference_dir + filename)
            else:
                baseline_file_ref = os.path.abspath(os.path.join(os.path.dirname(item.fspath.strpath), reference_dir, filename))

            if not os.path.exists(baseline_file_ref):
                raise Exception("""File not found for comparison test
                                Generated file:
                                \t{test}
                                This is expected for new tests.""".format(
                    test=test_array))

            # setuptools may put the baseline arrays in non-accessible places,
            # copy to our tmpdir to be sure to keep them in case of failure
            baseline_file = os.path.abspath(os.path.join(result_dir, 'reference-' + filename))
            shutil.copyfile(baseline_file_ref, baseline_file)

            identical, msg = FORMATS[file_format].compare(baseline_file, test_array, atol=atol, rtol=rtol)

            if identical:
                shutil.rmtree(result_dir)
            else:
                raise Exception(msg)

        else:

            if not os.path.exists(self.generate_dir):
                os.makedirs(self.generate_dir)

            FORMATS[file_format].write(os.path.abspath(os.path.join(self.generate_dir, filename)), array, **write_kwargs)

            pytest.skip("Skipping test, since generating data")


class ArrayInterceptor:
    """
    This is used in place of ArrayComparison when the array comparison option is not used,
    to make sure that we still intercept arrays returned by tests.
    """

    def __init__(self, config):
        self.config = config
        self.return_value = {}

    def pytest_collection_modifyitems(self, items):
        for item in items:
            wrap_array_interceptor(self, item)
