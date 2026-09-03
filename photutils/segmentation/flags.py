# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for working with segmentation quality flags, including
centralized flag definitions and decoding utilities.
"""

from typing import ClassVar

from photutils.utils._flags import (FlagDefinition, FlagRegistry, decode_flags,
                                    update_flag_docstring)

__all__ = ['SEGMENTATION_FLAGS', 'decode_segmentation_flags']


class _SegmentationFlags(FlagRegistry):
    """
    Centralized definition of segmentation quality flags.

    This class provides a single source of truth for all
    segmentation flag definitions, including bit values,
    names, and descriptions. The same flag definitions are
    used by `~photutils.segmentation.SegmentationImage`,
    `~photutils.segmentation.deblend_sources`, and
    `~photutils.segmentation.SourceCatalog`, so a given bit always has
    the same meaning.

    Flags that describe the same condition as an aperture flag (see
    `~photutils.aperture.decode_aperture_flags`), with the source
    segment as the measurement region, use the same name as the aperture
    flag. The bit values are independent between packages. Always decode
    a flag column with the decoder of the package that produced it.
    Flags describing the Kron aperture use a ``kron_`` name prefix.

    Examples
    --------
    >>> from photutils.segmentation.flags import _SegmentationFlags
    >>> flags = _SegmentationFlags()
    >>> flags.DEBLENDED
    32
    >>> flags.get_name(32)
    'deblended'
    """

    # Define all segmentation flags with their properties
    FLAG_DEFINITIONS: ClassVar = [
        FlagDefinition(
            bit_value=1,
            name='masked_pixels',
            description='masked pixels within the source segment',
            detailed_description=('One or more input-masked pixels '
                                  '(``mask`` keyword) are within the '
                                  'source segment.'),
        ),
        FlagDefinition(
            bit_value=2,
            name='non_finite_data',
            description=('non-finite data values within the source '
                         'segment'),
            detailed_description=('One or more data values (NaN or '
                                  'inf) within the source segment '
                                  'are non-finite.'),
        ),
        FlagDefinition(
            bit_value=4,
            name='non_finite_error',
            description=('non-finite error values within the source '
                         'segment'),
            detailed_description=('One or more error values (NaN or '
                                  'inf) within the source segment '
                                  'are non-finite.'),
        ),
        FlagDefinition(
            bit_value=8,
            name='all_masked',
            description='no valid pixels within the source segment',
            detailed_description=('Every pixel within the source '
                                  'segment is masked or non-finite, '
                                  'so the measured source properties '
                                  'are undefined.'),
        ),
        FlagDefinition(
            bit_value=16,
            name='edge_touch',
            description='source touches an image boundary',
            detailed_description=('The source segment touches one or '
                                  'more image boundaries, so the '
                                  'source may be truncated and its '
                                  'measured properties may be '
                                  'unreliable.'),
        ),
        FlagDefinition(
            bit_value=32,
            name='deblended',
            description='source was produced by deblending',
            detailed_description=('The source was produced by '
                                  'deblending a parent source with '
                                  '`~photutils.segmentation'
                                  '.deblend_sources`.'),
        ),
        FlagDefinition(
            bit_value=64,
            name='deblend_nonposmin',
            description=('deblending mode changed to linear: '
                         'non-positive minimum'),
            detailed_description=('The deblending mode for the '
                                  'parent source was changed to '
                                  '"linear" because of a '
                                  'non-positive minimum data value '
                                  'within the parent source '
                                  'segment.'),
        ),
        FlagDefinition(
            bit_value=128,
            name='deblend_n_markers',
            description=('deblending mode changed to linear: too '
                         'many markers'),
            detailed_description=('The deblending mode for the '
                                  'parent source was changed to '
                                  '"linear" because the parent '
                                  'source had too many potential '
                                  'deblended sources.'),
        ),
        FlagDefinition(
            bit_value=256,
            name='undefined_shape',
            description=('non-positive net flux (shape properties '
                         'undefined)'),
            detailed_description=('The net source flux (the zeroth '
                                  'image moment over the source '
                                  'segment) is not positive, so the '
                                  'centroid and the '
                                  'covariance-derived shape '
                                  'properties (e.g., ``centroid``, '
                                  '``semimajor_sigma``, '
                                  '``orientation``) are undefined or '
                                  'unreliable.'),
        ),
        FlagDefinition(
            bit_value=512,
            name='singular_covariance',
            description=('singular or nearly singular source '
                         'covariance'),
            detailed_description=('The source covariance matrix is '
                                  'singular or nearly singular (the '
                                  'minor-axis variance is below '
                                  '``1/12``, the variance of a '
                                  'uniform distribution across a '
                                  'single pixel), so '
                                  'covariance-derived shape '
                                  'properties (e.g., '
                                  '``semimajor_sigma``, '
                                  '``orientation``, '
                                  '``eccentricity``) are '
                                  'ill-defined.'),
        ),
        FlagDefinition(
            bit_value=1024,
            name='centroid_win_fallback',
            description=('windowed centroid failed or fell back to '
                         'the isophotal centroid'),
            detailed_description=('The windowed centroid fell '
                                  'outside the 1-sigma moment '
                                  'ellipse, the windowed flux was '
                                  'non-positive, the windowed '
                                  '2nd-order moments or covariance '
                                  'determinant were negative, or '
                                  'the iterated centroid was NaN. '
                                  'In each of these cases, the '
                                  'isophotal ``centroid`` value '
                                  'was used instead. If the '
                                  'half-light radius was not '
                                  'finite, the windowed centroid '
                                  'could not be computed and is '
                                  'NaN. See ``centroid_win`` for '
                                  'algorithm details.'),
        ),
        FlagDefinition(
            bit_value=2048,
            name='centroid_quad_failed',
            description='quadratic-fit centroid is non-finite',
            detailed_description=('The quadratic-fit centroid '
                                  '(``centroid_quad``) could not be '
                                  'computed and is NaN.'),
        ),
        FlagDefinition(
            bit_value=4096,
            name='kron_undefined',
            description='Kron aperture undefined or skipped',
            detailed_description=('The Kron aperture could not be '
                                  'defined (e.g., the source is '
                                  'fully masked) or the Kron '
                                  'photometry was skipped (e.g., the '
                                  'aperture was too large), so the '
                                  'Kron flux is NaN. The '
                                  'photometry-loop flags '
                                  '(``kron_no_overlap``, '
                                  '``kron_partial_overlap``, '
                                  '``kron_masked_pixels``, '
                                  '``kron_neighbor_pixels``, '
                                  '``kron_uncorrected_pixels``) are '
                                  'not evaluated for these sources, '
                                  'but ``kron_minimum_radius`` is '
                                  'computed independently from the '
                                  'radii and can still be set.'),
        ),
        FlagDefinition(
            bit_value=8192,
            name='kron_minimum_radius',
            description=('minimum Kron radius or minimum circular '
                         'radius applied'),
            detailed_description=('The measured unscaled Kron radius '
                                  'fell below the minimum unscaled '
                                  'Kron radius (``kron_params[1]``) '
                                  'and was clipped to it, or the '
                                  'minimum circular aperture '
                                  '(``kron_params[2]``) was used '
                                  'instead of the Kron ellipse.'),
        ),
        FlagDefinition(
            bit_value=16384,
            name='kron_no_overlap',
            description='Kron aperture fully outside the data',
            detailed_description=('The Kron aperture is fully '
                                  'outside the data array: no pixel '
                                  'with nonzero aperture weight '
                                  'falls inside the data.'),
        ),
        FlagDefinition(
            bit_value=32768,
            name='kron_partial_overlap',
            description='Kron aperture partially outside the data',
            detailed_description=('The Kron aperture is partially '
                                  'outside the data array: one or '
                                  'more pixels with nonzero aperture '
                                  'weight fall outside the data.'),
        ),
        FlagDefinition(
            bit_value=65536,
            name='kron_masked_pixels',
            description=('masked or non-finite pixels within the '
                         'Kron aperture'),
            detailed_description=('One or more input-masked or '
                                  'non-finite pixels have nonzero '
                                  'aperture weight within the Kron '
                                  'aperture.'),
        ),
        FlagDefinition(
            bit_value=131072,
            name='kron_neighbor_pixels',
            description=('neighbor-source pixels within the Kron '
                         'aperture'),
            detailed_description=('One or more pixels within the '
                                  'Kron aperture were excluded or '
                                  'corrected due to neighboring '
                                  'sources in the segmentation image '
                                  '(``aperture_mask_method`` of '
                                  '"mask" or "correct").'),
        ),
        FlagDefinition(
            bit_value=262144,
            name='kron_uncorrected_pixels',
            description=('uncorrectable neighbor pixels within the '
                         'Kron aperture'),
            detailed_description=('With '
                                  '``aperture_mask_method="correct"``'
                                  ', one or more neighbor-source '
                                  'pixels within the Kron aperture '
                                  'could not be corrected (the '
                                  'mirror pixel was unavailable) and '
                                  'were set to zero instead.'),
        ),
    ]

    domain: ClassVar = 'segmentation'


# Create a singleton instance for global use
SEGMENTATION_FLAGS = _SegmentationFlags()


def _update_decode_docstring(func):
    """
    Decorator to update a function docstring with the segmentation flag
    documentation.

    The ``<flag_descriptions>`` placeholder in the function docstring is
    replaced with a bullet list generated from ``SEGMENTATION_FLAGS``
    (see `photutils.utils._flags.update_flag_docstring`).

    Parameters
    ----------
    func : function
        The function to decorate.

    Returns
    -------
    func : function
        The decorated function with updated docstring.
    """
    return update_flag_docstring(func, SEGMENTATION_FLAGS, indent=4)


@_update_decode_docstring
def decode_segmentation_flags(flags, *, return_bit_values=False):
    # numpydoc ignore: RT05
    """
    Decode segmentation bitwise flag values into individual components.

    This function takes integer flag values from segmentation operations
    and returns a list of human-readable names of the conditions that
    were detected. This is useful for understanding what problems were
    encountered without needing to manually perform bitwise operations.

    Parameters
    ----------
    flags : int or array-like of int
        Integer flag value(s) to decode. Each bit in the flag represents
        a specific condition that was detected when processing the
        segmentation.

    return_bit_values : bool, optional
        If `True`, return the decoded bit flags (integers) instead of
        the flag names (strings). Default is `False`.

    Returns
    -------
    decoded : list of str, list of int, or nested list
        List of active flag names (or bit values) for a scalar input.
        For an array input, a nested list with the same shape as the
        input is returned, where each innermost element is the list of
        active flag names (or bit values) for the corresponding flag. If
        no flags are set, an empty list is returned. Possible flags are:
        <flag_descriptions>

    Examples
    --------
    Decode a single flag value:

    >>> from photutils.segmentation import decode_segmentation_flags
    >>> issues = decode_segmentation_flags(48)  # bits 16 and 32 set
    >>> print(issues)
    ['edge_touch', 'deblended']
    >>> 'edge_touch' in issues
    True
    >>> 'deblended' in issues
    True

    Decode multiple flag values:

    >>> flags = [0, 1, 33]  # 0, bit 1, bits 1+32
    >>> decoded_list = decode_segmentation_flags(flags)
    >>> decoded_list[0]  # No issues
    []
    >>> decoded_list[1]
    ['masked_pixels']
    >>> decoded_list[2]
    ['masked_pixels', 'deblended']

    Return the bit values instead of the names:

    >>> decode_segmentation_flags(48, return_bit_values=True)
    [16, 32]
    """
    return decode_flags(flags, SEGMENTATION_FLAGS,
                        return_bit_values=return_bit_values)
