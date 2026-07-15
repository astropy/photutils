# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for working with aperture photometry and statistics flags,
including centralized flag definitions and decoding utilities.
"""

from typing import ClassVar

from photutils.utils._flags import (FlagDefinition, FlagRegistry, decode_flags,
                                    update_flag_docstring)

__all__ = ['APERTURE_FLAGS', 'decode_aperture_flags']


class _ApertureFlags(FlagRegistry):
    """
    Centralized definition of aperture photometry and statistics
    flags.

    This class provides a single source of truth for all
    aperture flag definitions, including bit values,
    names, and descriptions. The same flag definitions are
    used by `~photutils.aperture.aperture_photometry`,
    `~photutils.aperture.PixelAperture.photometry`, and
    `~photutils.aperture.ApertureStats`, so a given bit always has the
    same meaning.

    Examples
    --------
    >>> from photutils.aperture.flags import _ApertureFlags
    >>> flags = _ApertureFlags()
    >>> flags.NO_OVERLAP
    1
    >>> flags.get_name(1)
    'no_overlap'
    >>> flags.get_description(8)
    'masked pixels within the aperture'
    """

    # Define all aperture flags with their properties
    FLAG_DEFINITIONS: ClassVar = [
        FlagDefinition(
            bit_value=1,
            name='no_overlap',
            description='aperture fully outside the data',
            detailed_description=('The aperture is fully outside the '
                                  'data array: no pixel with nonzero '
                                  'aperture weight falls inside the '
                                  'data'),
        ),
        FlagDefinition(
            bit_value=2,
            name='partial_overlap',
            description='aperture partially outside the data',
            detailed_description=('The aperture is partially outside '
                                  'the data array: one or more pixels '
                                  'with nonzero aperture weight fall '
                                  'outside the data'),
        ),
        FlagDefinition(
            bit_value=4,
            name='no_pixels',
            description='no aperture pixels within the data',
            detailed_description=('The aperture contains zero pixels '
                                  'with nonzero weight inside the data, '
                                  'e.g., a fully off-image aperture or '
                                  'a tiny aperture that contains no '
                                  'pixel (or subpixel) centers with '
                                  'the "center" or "subpixel" methods'),
        ),
        FlagDefinition(
            bit_value=8,
            name='masked_pixels',
            description='masked pixels within the aperture',
            detailed_description=('One or more input-masked pixels '
                                  '(``mask`` keyword) have nonzero '
                                  'aperture weight'),
        ),
        FlagDefinition(
            bit_value=16,
            name='all_masked',
            description='no valid pixels within the aperture',
            detailed_description=('The aperture contains pixels, but '
                                  'none are valid: every nonzero-weight '
                                  'pixel inside the data is masked, '
                                  'non-finite, or excluded by '
                                  'segmentation masking'),
        ),
        FlagDefinition(
            bit_value=32,
            name='non_finite_data',
            description='non-finite data values within the aperture',
            detailed_description=('One or more unmasked data values '
                                  '(NaN or inf) with nonzero aperture '
                                  'weight are non-finite'),
        ),
        FlagDefinition(
            bit_value=64,
            name='non_finite_error',
            description='non-finite error values within the aperture',
            detailed_description=('One or more unmasked error values '
                                  '(NaN or inf) with nonzero aperture '
                                  'weight are non-finite'),
        ),
        FlagDefinition(
            bit_value=128,
            name='neighbor_pixels',
            description='segmentation-masked pixels within the aperture',
            detailed_description=('One or more pixels within the '
                                  'aperture were excluded, restricted, '
                                  'or corrected due to neighboring '
                                  'sources in the segmentation image'),
        ),
        FlagDefinition(
            bit_value=256,
            name='uncorrected_pixels',
            description='uncorrectable neighbor pixels within the aperture',
            detailed_description=('With ``mask_method="correct"``, one '
                                  'or more neighbor-source pixels could '
                                  'not be corrected (the mirror pixel '
                                  'was unavailable) and were excluded '
                                  'instead'),
        ),
        FlagDefinition(
            bit_value=512,
            name='sigma_clipped',
            description='sigma-clipped pixels within the aperture',
            detailed_description=('One or more pixels within the '
                                  'aperture were rejected by sigma '
                                  'clipping'),
        ),
        FlagDefinition(
            bit_value=1024,
            name='all_clipped',
            description='all pixels within the aperture were sigma clipped',
            detailed_description=('All valid pixels within the aperture '
                                  'were rejected by sigma clipping'),
        ),
        FlagDefinition(
            bit_value=2048,
            name='too_few_pixels',
            description='too few pixels to compute a statistic',
            detailed_description=('There are too few valid pixels '
                                  'within the aperture to compute a '
                                  'requested statistic (e.g., the '
                                  'variance and standard deviation are '
                                  'undefined when the number of valid '
                                  'pixels is not larger than ``ddof``)'),
        ),
    ]

    domain: ClassVar = 'aperture'


# Create a singleton instance for global use
APERTURE_FLAGS = _ApertureFlags()


def _update_decode_docstring(func):
    """
    Decorator to update a function docstring with the aperture flag
    documentation.

    The ``<flag_descriptions>`` placeholder in the function docstring is
    replaced with a bullet list generated from ``APERTURE_FLAGS`` (see
    `photutils.utils._flags.update_flag_docstring`).

    Parameters
    ----------
    func : function
        The function to decorate.

    Returns
    -------
    func : function
        The decorated function with updated docstring.
    """
    return update_flag_docstring(func, APERTURE_FLAGS)


@_update_decode_docstring
def decode_aperture_flags(flags, *, return_bit_values=False):
    # numpydoc ignore: RT05
    """
    Decode aperture bit flags into individual components.

    This function takes integer flag values from aperture photometry or
    aperture statistics results and returns a list of human-readable
    names of the conditions that were detected. This is useful for
    understanding what problems were encountered without needing to
    manually perform bitwise operations.

    Parameters
    ----------
    flags : int or array-like of int
        Integer flag value(s) to decode. Each bit in the flag represents
        a specific condition that was detected when measuring the
        aperture.

    return_bit_values : bool, optional
        If `True`, return the decoded bit flags (integers) instead of
        the flag names (strings). Default is `False`.

    Returns
    -------
    decoded : list of str, list of int, list of list of str, or \
            list of list of int
        List of active flag names (or bit values), or list of lists
        if the input is an array. Each string (or integer) represents
        a specific condition that was detected. If no flags are
        set, an empty list is returned. Possible flag names are:
        <flag_descriptions>

    Examples
    --------
    Decode a single flag value:

    >>> from photutils.aperture import decode_aperture_flags
    >>> issues = decode_aperture_flags(10)  # bits 2 and 8 set
    >>> print(issues)
    ['partial_overlap', 'masked_pixels']
    >>> 'partial_overlap' in issues
    True
    >>> 'no_overlap' in issues
    False

    Decode multiple flag values:

    >>> flags = [0, 8, 24]  # 0, bit 8, bits 8+16
    >>> decoded_list = decode_aperture_flags(flags)
    >>> decoded_list[0]  # No issues
    []
    >>> decoded_list[1]
    ['masked_pixels']
    >>> decoded_list[2]
    ['masked_pixels', 'all_masked']

    Return the bit values instead of the names:

    >>> decode_aperture_flags(10, return_bit_values=True)
    [2, 8]
    """
    return decode_flags(flags, APERTURE_FLAGS,
                        return_bit_values=return_bit_values)
