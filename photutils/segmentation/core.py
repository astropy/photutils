# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides classes for a segmentation image and a single
segment within a segmentation image.
"""

from copy import deepcopy

from astropy.utils import lazyproperty
import numpy as np

from ..aperture import BoundingBox
from ..utils.colormaps import make_random_cmap

__all__ = ['SegmentationImage', 'Segment']

__doctest_requires__ = {('SegmentationImage', 'SegmentationImage.*'):
                        ['scipy']}


class SegmentationImage:
    """
    Class for a segmentation image.

    Parameters
    ----------
    data : array_like (int)
        A segmentation array where source regions are labeled by
        different positive integer values.  A value of zero is reserved
        for the background.  The segmentation image must contain at
        least one non-zero pixel and must not contain any non-finite
        values (e.g., NaN, inf).
    """

    def __init__(self, data):
        self.data = data

    def __str__(self):
        cls_name = f'<{self.__class__.__module__}.{self.__class__.__name__}>'

        cls_info = []
        params = ['shape', 'nlabels']
        for param in params:
            cls_info.append((param, getattr(self, param)))
        cls_info.append(('labels', self.labels))
        with np.printoptions(threshold=25, edgeitems=5):
            fmt = [f'{key}: {val}' for key, val in cls_info]

        return f'{cls_name}\n' + '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()

    def __array__(self):
        """
        Array representation of the segmentation array (e.g., for
        matplotlib).
        """
        return self._data

    @lazyproperty
    def _cmap(self):
        """
        A matplotlib colormap consisting of (random) muted colors.

        This is very useful for plotting the segmentation array.
        """
        return self.make_cmap(background_color='#000000', seed=0)

    @staticmethod
    def _get_labels(data):
        """
        Return a sorted array of the non-zero labels in the segmentation
        image.

        Parameters
        ----------
        data : array_like (int)
            A segmentation array where source regions are labeled by
            different positive integer values.  A value of zero is
            reserved for the background.

        Returns
        -------
        result : `~numpy.ndarray`
            An array of non-zero label numbers.

        Notes
        -----
        This is a static method so it can be used in
        :meth:`remove_masked_labels` on a masked version of the
        segmentation array.
        """
        # np.unique also sorts elements
        return np.unique(data[data != 0])

    @lazyproperty
    def segments(self):
        """
        A list of `Segment` objects.

        The list starts with the *non-zero* label.  The returned list
        has a length equal to the number of labels and matches the order
        of the ``labels`` attribute.
        """
        segments = []
        for label, slc, bbox, area in zip(self.labels, self.slices, self.bbox,
                                          self.areas):
            segments.append(Segment(self.data, label, slc, bbox, area))
        return segments

    @property
    def data(self):
        """The segmentation array."""
        return self._data

    @data.setter
    def data(self, value):
        if np.any(~np.isfinite(value)):
            raise ValueError('data must not contain any non-finite values '
                             '(e.g., NaN, inf)')

        value = np.asarray(value, dtype=int)
        if not np.any(value):
            raise ValueError('The segmentation image must contain at least '
                             'one non-zero pixel.')

        if np.min(value) < 0:
            raise ValueError('The segmentation image cannot contain '
                             'negative integers.')

        if '_data' in self.__dict__:
            # needed only when data is reassigned, not on init
            self.__dict__ = {}

        self._data = value  # pylint: disable=attribute-defined-outside-init

    @lazyproperty
    def data_ma(self):
        """
        A `~numpy.ma.MaskedArray` version of the segmentation array
        where the background (label = 0) has been masked.
        """
        return np.ma.masked_where(self.data == 0, self.data)

    @lazyproperty
    def shape(self):
        """The shape of the segmentation array."""
        return self._data.shape

    @lazyproperty
    def _ndim(self):
        """The number of array dimensions of the segmentation array."""
        return self._data.ndim

    @lazyproperty
    def labels(self):
        """The sorted non-zero labels in the segmentation array."""
        return self._get_labels(self.data)

    @lazyproperty
    def nlabels(self):
        """The number of non-zero labels in the segmentation array."""
        return len(self.labels)

    @lazyproperty
    def max_label(self):
        """The maximum non-zero label in the segmentation array."""
        return np.max(self.labels)

    def get_index(self, label):
        """
        Find the index of the input ``label``.

        Parameters
        ----------
        label : int
            The label number to find.

        Returns
        -------
        index : int
            The array index.

        Raises
        ------
        ValueError
            If ``label`` is invalid.
        """
        self.check_labels(label)
        return np.searchsorted(self.labels, label)

    def get_indices(self, labels):
        """
        Find the indices of the input ``labels``.

        Parameters
        ----------
        labels : int, array-like (1D, int)
            The label numbers(s) to find.

        Returns
        -------
        indices : int `~numpy.ndarray`
            An integer array of indices with the same shape as
            ``labels``.  If ``labels`` is a scalar, then the returned
            index will also be a scalar.

        Raises
        ------
        ValueError
            If any input ``labels`` are invalid.
        """
        self.check_labels(labels)
        return np.searchsorted(self.labels, labels)

    @lazyproperty
    def slices(self):
        """
        A list of tuples, where each tuple contains two slices
        representing the minimal box that contains the labeled region.

        The list starts with the *non-zero* label.  The returned list
        has a length equal to the number of labels and matches the order
        of the ``labels`` attribute.
        """
        from scipy.ndimage import find_objects

        return [slc for slc in find_objects(self._data) if slc is not None]

    @lazyproperty
    def bbox(self):
        """
        A list of `~photutils.aperture.BoundingBox` of the minimal
        bounding boxes containing the labeled regions.
        """
        if self._ndim != 2:
            raise ValueError('The "bbox" attribute requires a 2D '
                             'segmentation image.')

        return [BoundingBox(ixmin=slc[1].start, ixmax=slc[1].stop,
                            iymin=slc[0].start, iymax=slc[0].stop)
                for slc in self.slices]

    @lazyproperty
    def background_area(self):
        """The area (in pixel**2) of the background (label=0) region."""
        return len(self.data[self.data == 0])

    @lazyproperty
    def areas(self):
        """
        A 1D array of areas (in pixel**2) of the non-zero labeled
        regions.

        The `~numpy.ndarray` starts with the *non-zero* label.  The
        returned array has a length equal to the number of labels and
        matches the order of the ``labels`` attribute.
        """
        return np.array([area
                         for area in np.bincount(self.data.ravel())[1:]
                         if area != 0])

    def get_area(self, label):
        """
        The area (in pixel**2) of the region for the input label.

        Parameters
        ----------
        label : int
            The label whose area to return.  Label must be non-zero.

        Returns
        -------
        area : `~numpy.ndarray`
            The area of the labeled region.
        """
        return self.get_areas(label)

    def get_areas(self, labels):
        """
        The areas (in pixel**2) of the regions for the input labels.

        Parameters
        ----------
        labels : int, 1D array-like (int)
            The label(s) for which to return areas.  Label must be
            non-zero.

        Returns
        -------
        areas : `~numpy.ndarray`
            The areas of the labeled regions.
        """
        idx = self.get_indices(labels)
        return self.areas[idx]

    @lazyproperty
    def is_consecutive(self):
        """
        Determine whether or not the non-zero labels in the segmentation
        array are consecutive and start from 1.
        """
        return ((self.labels[-1] - self.labels[0] + 1) == self.nlabels and
                self.labels[0] == 1)

    @lazyproperty
    def missing_labels(self):
        """
        A 1D `~numpy.ndarray` of the sorted non-zero labels that are
        missing in the consecutive sequence from one to the maximum
        label number.
        """
        return np.array(sorted(set(range(0, self.max_label + 1))
                               .difference(np.insert(self.labels, 0, 0))))

    def copy(self):
        """Return a deep copy of this class instance."""
        return deepcopy(self)

    def check_label(self, label):
        """
        Check that the input label is a valid label number within the
        segmentation array.

        Parameters
        ----------
        label : int
            The label number to check.

        Raises
        ------
        ValueError
            If the input ``label`` is invalid.
        """
        self.check_labels(label)

    def check_labels(self, labels):
        """
        Check that the input label(s) are valid label numbers within the
        segmentation array.

        Parameters
        ----------
        labels : int, 1D array-like (int)
            The label(s) to check.

        Raises
        ------
        ValueError
            If any input ``labels`` are invalid.
        """
        labels = np.atleast_1d(labels)
        bad_labels = set()

        # check for positive label numbers
        idx = np.where(labels <= 0)[0]
        if idx.size > 0:
            bad_labels.update(labels[idx])

        # check if label is in the segmentation array
        bad_labels.update(np.setdiff1d(labels, self.labels))

        if bad_labels:
            if len(bad_labels) == 1:
                raise ValueError(f'label {bad_labels} is invalid')
            else:
                raise ValueError(f'labels {bad_labels} are invalid')

    def make_cmap(self, background_color='#000000', seed=None):
        """
        Define a matplotlib colormap consisting of (random) muted
        colors.

        This is very useful for plotting the segmentation array.

        Parameters
        ----------
        background_color : str or `None`, optional
            A hex string in the "#rrggbb" format defining the first
            color in the colormap.  This color will be used as the
            background color (label = 0) when plotting the segmentation
            array.  The default is black ('#000000').

        seed : int, optional
            A seed to initialize the `numpy.random.BitGenerator`. If
            `None`, then fresh, unpredictable entropy will be pulled
            from the OS.  Separate function calls with the same ``seed``
            will generate the same colormap.

        Returns
        -------
        cmap : `matplotlib.colors.ListedColormap`
            The matplotlib colormap.
        """
        from matplotlib import colors

        cmap = make_random_cmap(self.max_label + 1, seed=seed)

        if background_color is not None:
            cmap.colors[0] = colors.hex2color(background_color)

        return cmap

    def reassign_label(self, label, new_label, relabel=False):
        """
        Reassign a label number to a new number.

        If ``new_label`` is already present in the segmentation array,
        then it will be combined with the input ``label`` number.

        Parameters
        ----------
        label : int
            The label number to reassign.

        new_label : int
            The newly assigned label number.

        relabel : bool, optional
            If `True`, then the segmentation array will be relabeled
            such that the labels are in consecutive order starting from
            1.

        Examples
        --------
        >>> from photutils.segmentation import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.reassign_label(label=1, new_label=2)
        >>> segm.data
        array([[2, 2, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 3, 3, 0, 0],
               [7, 0, 0, 0, 0, 5],
               [7, 7, 0, 5, 5, 5],
               [7, 7, 0, 0, 5, 5]])

        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.reassign_label(label=1, new_label=4)
        >>> segm.data
        array([[4, 4, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 3, 3, 0, 0],
               [7, 0, 0, 0, 0, 5],
               [7, 7, 0, 5, 5, 5],
               [7, 7, 0, 0, 5, 5]])

        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.reassign_label(label=1, new_label=4, relabel=True)
        >>> segm.data
        array([[2, 2, 0, 0, 2, 2],
               [0, 0, 0, 0, 0, 2],
               [0, 0, 1, 1, 0, 0],
               [4, 0, 0, 0, 0, 3],
               [4, 4, 0, 3, 3, 3],
               [4, 4, 0, 0, 3, 3]])
        """
        self.reassign_labels(label, new_label, relabel=relabel)

    def reassign_labels(self, labels, new_label, relabel=False):
        """
        Reassign one or more label numbers.

        Multiple input ``labels`` will all be reassigned to the same
        ``new_label`` number.  If ``new_label`` is already present in
        the segmentation array, then it will be combined with the input
        ``labels``.

        Parameters
        ----------
        labels : int, array-like (1D, int)
            The label numbers(s) to reassign.

        new_label : int
            The reassigned label number.

        relabel : bool, optional
            If `True`, then the segmentation array will be relabeled
            such that the labels are in consecutive order starting from
            1.

        Examples
        --------
        >>> from photutils.segmentation import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.reassign_labels(labels=[1, 7], new_label=2)
        >>> segm.data
        array([[2, 2, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 3, 3, 0, 0],
               [2, 0, 0, 0, 0, 5],
               [2, 2, 0, 5, 5, 5],
               [2, 2, 0, 0, 5, 5]])

        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.reassign_labels(labels=[1, 7], new_label=4)
        >>> segm.data
        array([[4, 4, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 3, 3, 0, 0],
               [4, 0, 0, 0, 0, 5],
               [4, 4, 0, 5, 5, 5],
               [4, 4, 0, 0, 5, 5]])

        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.reassign_labels(labels=[1, 7], new_label=2, relabel=True)
        >>> segm.data
        array([[1, 1, 0, 0, 3, 3],
               [0, 0, 0, 0, 0, 3],
               [0, 0, 2, 2, 0, 0],
               [1, 0, 0, 0, 0, 4],
               [1, 1, 0, 4, 4, 4],
               [1, 1, 0, 0, 4, 4]])
        """
        self.check_labels(labels)

        labels = np.atleast_1d(labels)
        if labels.size == 0:
            return

        idx = np.zeros(self.max_label + 1, dtype=int)
        idx[self.labels] = self.labels
        idx[labels] = new_label  # reassign labels

        if relabel:
            labels = np.unique(idx[idx != 0])
            if not len(labels) == 0:
                idx2 = np.zeros(max(labels) + 1, dtype=int)
                idx2[labels] = np.arange(len(labels)) + 1
                idx = idx2[idx]

        data_new = idx[self.data]
        self.__dict__ = {}  # reset all cached properties
        self._data = data_new  # use _data to avoid validation

    def relabel_consecutive(self, start_label=1):
        """
        Reassign the label numbers consecutively starting from a given
        label number.

        Parameters
        ----------
        start_label : int, optional
            The starting label number, which should be a strictly
            positive integer.  The default is 1.

        Examples
        --------
        >>> from photutils.segmentation import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.relabel_consecutive()
        >>> segm.data
        array([[1, 1, 0, 0, 3, 3],
               [0, 0, 0, 0, 0, 3],
               [0, 0, 2, 2, 0, 0],
               [5, 0, 0, 0, 0, 4],
               [5, 5, 0, 4, 4, 4],
               [5, 5, 0, 0, 4, 4]])
        """
        if start_label <= 0:
            raise ValueError('start_label must be > 0.')

        if ((self.labels[0] == start_label) and
                (self.labels[-1] - self.labels[0] + 1) == self.nlabels):
            return

        new_labels = np.zeros(self.max_label + 1, dtype=int)
        new_labels[self.labels] = np.arange(self.nlabels) + start_label

        data_new = new_labels[self.data]
        self.__dict__ = {}  # reset all cached properties
        self._data = data_new  # use _data to avoid validation

    def keep_label(self, label, relabel=False):
        """
        Keep only the specified label.

        Parameters
        ----------
        label : int
            The label number to keep.

        relabel : bool, optional
            If `True`, then the single segment will be assigned a label
            value of 1.

        Examples
        --------
        >>> from photutils.segmentation import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.keep_label(label=3)
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 3, 3, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]])

        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.keep_label(label=3, relabel=True)
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 1, 1, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]])
        """
        self.keep_labels(label, relabel=relabel)

    def keep_labels(self, labels, relabel=False):
        """
        Keep only the specified labels.

        Parameters
        ----------
        labels : int, array-like (1D, int)
            The label number(s) to keep.

        relabel : bool, optional
            If `True`, then the segmentation array will be relabeled
            such that the labels are in consecutive order starting from
            1.

        Examples
        --------
        >>> from photutils.segmentation import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.keep_labels(labels=[5, 3])
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 3, 3, 0, 0],
               [0, 0, 0, 0, 0, 5],
               [0, 0, 0, 5, 5, 5],
               [0, 0, 0, 0, 5, 5]])

        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.keep_labels(labels=[5, 3], relabel=True)
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 1, 1, 0, 0],
               [0, 0, 0, 0, 0, 2],
               [0, 0, 0, 2, 2, 2],
               [0, 0, 0, 0, 2, 2]])
        """
        self.check_labels(labels)

        labels = np.atleast_1d(labels)
        labels_tmp = list(set(self.labels) - set(labels))
        self.remove_labels(labels_tmp, relabel=relabel)

    def remove_label(self, label, relabel=False):
        """
        Remove the label number.

        The removed label is assigned a value of zero (i.e.,
        background).

        Parameters
        ----------
        label : int
            The label number to remove.

        relabel : bool, optional
            If `True`, then the segmentation array will be relabeled
            such that the labels are in consecutive order starting from
            1.

        Examples
        --------
        >>> from photutils.segmentation import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.remove_label(label=5)
        >>> segm.data
        array([[1, 1, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 3, 3, 0, 0],
               [7, 0, 0, 0, 0, 0],
               [7, 7, 0, 0, 0, 0],
               [7, 7, 0, 0, 0, 0]])

        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.remove_label(label=5, relabel=True)
        >>> segm.data
        array([[1, 1, 0, 0, 3, 3],
               [0, 0, 0, 0, 0, 3],
               [0, 0, 2, 2, 0, 0],
               [4, 0, 0, 0, 0, 0],
               [4, 4, 0, 0, 0, 0],
               [4, 4, 0, 0, 0, 0]])
        """
        self.remove_labels(label, relabel=relabel)

    def remove_labels(self, labels, relabel=False):
        """
        Remove one or more labels.

        Removed labels are assigned a value of zero (i.e., background).

        Parameters
        ----------
        labels : int, array-like (1D, int)
            The label number(s) to remove.

        relabel : bool, optional
            If `True`, then the segmentation array will be relabeled
            such that the labels are in consecutive order starting from
            1.

        Examples
        --------
        >>> from photutils.segmentation import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.remove_labels(labels=[5, 3])
        >>> segm.data
        array([[1, 1, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 0, 0, 0, 0],
               [7, 0, 0, 0, 0, 0],
               [7, 7, 0, 0, 0, 0],
               [7, 7, 0, 0, 0, 0]])

        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.remove_labels(labels=[5, 3], relabel=True)
        >>> segm.data
        array([[1, 1, 0, 0, 2, 2],
               [0, 0, 0, 0, 0, 2],
               [0, 0, 0, 0, 0, 0],
               [3, 0, 0, 0, 0, 0],
               [3, 3, 0, 0, 0, 0],
               [3, 3, 0, 0, 0, 0]])
        """
        self.check_labels(labels)
        self.reassign_labels(labels, new_label=0, relabel=relabel)

    def remove_border_labels(self, border_width, partial_overlap=True,
                             relabel=False):
        """
        Remove labeled segments near the array border.

        Labels within the defined border region will be removed.

        Parameters
        ----------
        border_width : int
            The width of the border region in pixels.

        partial_overlap : bool, optional
            If this is set to `True` (the default), a segment that
            partially extends into the border region will be removed.
            Segments that are completely within the border region are
            always removed.

        relabel : bool, optional
            If `True`, then the segmentation array will be relabeled
            such that the labels are in consecutive order starting from
            1.

        Examples
        --------
        >>> from photutils.segmentation import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.remove_border_labels(border_width=1)
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 3, 3, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]])

        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.remove_border_labels(border_width=1,
        ...                           partial_overlap=False)
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 3, 3, 0, 0],
               [7, 0, 0, 0, 0, 5],
               [7, 7, 0, 5, 5, 5],
               [7, 7, 0, 0, 5, 5]])
        """
        if border_width >= min(self.shape) / 2:
            raise ValueError('border_width must be smaller than half the '
                             'array size in any dimension')

        border_mask = np.zeros(self.shape, dtype=bool)
        for i in range(border_mask.ndim):
            border_mask = border_mask.swapaxes(0, i)
            border_mask[:border_width] = True
            border_mask[-border_width:] = True
            border_mask = border_mask.swapaxes(0, i)

        self.remove_masked_labels(border_mask,
                                  partial_overlap=partial_overlap,
                                  relabel=relabel)

    def remove_masked_labels(self, mask, partial_overlap=True,
                             relabel=False):
        """
        Remove labeled segments located within a masked region.

        Parameters
        ----------
        mask : array_like (bool)
            A boolean mask, with the same shape as the segmentation
            array, where `True` values indicate masked pixels.

        partial_overlap : bool, optional
            If this is set to `True` (default), a segment that partially
            extends into a masked region will also be removed.  Segments
            that are completely within a masked region are always
            removed.

        relabel : bool, optional
            If `True`, then the segmentation array will be relabeled
            such that the labels are in consecutive order starting from
            1.

        Examples
        --------
        >>> from photutils.segmentation import SegmentationImage
        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> mask = np.zeros(segm.data.shape, dtype=bool)
        >>> mask[0, :] = True  # mask the first row
        >>> segm.remove_masked_labels(mask)
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 3, 3, 0, 0],
               [7, 0, 0, 0, 0, 5],
               [7, 7, 0, 5, 5, 5],
               [7, 7, 0, 0, 5, 5]])

        >>> segm = SegmentationImage([[1, 1, 0, 0, 4, 4],
        ...                           [0, 0, 0, 0, 0, 4],
        ...                           [0, 0, 3, 3, 0, 0],
        ...                           [7, 0, 0, 0, 0, 5],
        ...                           [7, 7, 0, 5, 5, 5],
        ...                           [7, 7, 0, 0, 5, 5]])
        >>> segm.remove_masked_labels(mask, partial_overlap=False)
        >>> segm.data
        array([[0, 0, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 3, 3, 0, 0],
               [7, 0, 0, 0, 0, 5],
               [7, 7, 0, 5, 5, 5],
               [7, 7, 0, 0, 5, 5]])
        """
        if mask.shape != self.shape:
            raise ValueError('mask must have the same shape as the '
                             'segmentation array')
        remove_labels = self._get_labels(self.data[mask])
        if not partial_overlap:
            interior_labels = self._get_labels(self.data[~mask])
            remove_labels = list(set(remove_labels) - set(interior_labels))
        self.remove_labels(remove_labels, relabel=relabel)

    def outline_segments(self, mask_background=False):
        """
        Outline the labeled segments.

        The "outlines" represent the pixels *just inside* the segments,
        leaving the background pixels unmodified.

        Parameters
        ----------
        mask_background : bool, optional
            Set to `True` to mask the background pixels (labels = 0) in
            the returned array.  This is useful for overplotting the
            segment outlines.  The default is `False`.

        Returns
        -------
        boundaries : `~numpy.ndarray` or `~numpy.ma.MaskedArray`
            An array with the same shape of the segmentation array
            containing only the outlines of the labeled segments.  The
            pixel values in the outlines correspond to the labels in the
            segmentation array.  If ``mask_background`` is `True`, then
            a `~numpy.ma.MaskedArray` is returned.

        Examples
        --------
        >>> from photutils.segmentation import SegmentationImage
        >>> segm = SegmentationImage([[0, 0, 0, 0, 0, 0],
        ...                           [0, 2, 2, 2, 2, 0],
        ...                           [0, 2, 2, 2, 2, 0],
        ...                           [0, 2, 2, 2, 2, 0],
        ...                           [0, 2, 2, 2, 2, 0],
        ...                           [0, 0, 0, 0, 0, 0]])
        >>> segm.outline_segments()
        array([[0, 0, 0, 0, 0, 0],
               [0, 2, 2, 2, 2, 0],
               [0, 2, 0, 0, 2, 0],
               [0, 2, 0, 0, 2, 0],
               [0, 2, 2, 2, 2, 0],
               [0, 0, 0, 0, 0, 0]])
        """
        from scipy.ndimage import (generate_binary_structure, grey_dilation,
                                   grey_erosion)

        # mode='constant' ensures outline is included on the array borders
        selem = generate_binary_structure(self._ndim, 1)  # edge connectivity
        eroded = grey_erosion(self.data, footprint=selem, mode='constant',
                              cval=0.)
        dilated = grey_dilation(self.data, footprint=selem, mode='constant',
                                cval=0.)

        outlines = ((dilated != eroded) & (self.data != 0)).astype(int)
        outlines *= self.data

        if mask_background:
            outlines = np.ma.masked_where(outlines == 0, outlines)

        return outlines


class Segment:
    """
    Class for a single labeled region (segment) within a segmentation
    image.

    Parameters
    ----------
    segment_data : int `~numpy.ndarray`
        A segmentation array where source regions are labeled by
        different positive integer values.  A value of zero is reserved
        for the background.

    label : int
        The segment label number.

    slices : tuple of two slices
        A tuple of two slices representing the minimal box that contains
        the labeled region.

    bbox : `~photutils.aperture.BoundingBox`
        The minimal bounding box that contains the labeled region.

    area : float
        The area of the segment in pixels**2.
    """

    def __init__(self, segment_data, label, slices, bbox, area):
        self._segment_data = segment_data
        self.label = label
        self.slices = slices
        self.bbox = bbox
        self.area = area

    def __str__(self):
        cls_name = f'<{self.__class__.__module__}.{self.__class__.__name__}>'

        cls_info = []
        params = ['label', 'slices', 'area']
        for param in params:
            cls_info.append((param, getattr(self, param)))
        fmt = [f'{key}: {val}' for key, val in cls_info]

        return f'{cls_name}\n' + '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()

    def __array__(self):
        """
        Array representation of the labeled region (e.g., for
        matplotlib).
        """
        return self.data

    @lazyproperty
    def data(self):
        """
        A cutout array of the segment using the minimal bounding box,
        where pixels outside of the labeled region are set to zero
        (i.e., neighboring segments within the rectangular cutout array
        are not shown).
        """
        cutout = np.copy(self._segment_data[self.slices])
        cutout[cutout != self.label] = 0

        return cutout

    @lazyproperty
    def data_ma(self):
        """
        A `~numpy.ma.MaskedArray` cutout array of the segment using the
        minimal bounding box.

        The mask is `True` for pixels outside of the source segment
        (i.e., neighboring segments within the rectangular cutout array
        are masked).
        """
        mask = (self._segment_data[self.slices] != self.label)
        return np.ma.masked_array(self._segment_data[self.slices], mask=mask)

    def make_cutout(self, data, masked_array=False):
        """
        Create a (masked) cutout array from the input ``data`` using the
        minimal bounding box of the segment (labeled region).

        If ``masked_array`` is `False` (default), then the returned
        cutout array is simply a `~numpy.ndarray`.  The returned cutout
        is a view (not a copy) of the input ``data``.  No pixels are
        altered (e.g., set to zero) within the bounding box.

        If ``masked_array` is `True`, then the returned cutout array is
        a `~numpy.ma.MaskedArray`, where the mask is `True` for pixels
        outside of the segment (labeled region).  The data part of the
        masked array is a view (not a copy) of the input ``data``.

        Parameters
        ----------
        data : array-like
            The data array from which to create the masked cutout array.
            ``data`` must have the same shape as the segmentation array.

        masked_array : bool, optional
            If `True` then a `~numpy.ma.MaskedArray` will be created
            where the mask is `True` for pixels outside of the segment
            (labeled region).  If `False`, then a `~numpy.ndarray` will
            be generated.

        Returns
        -------
        result : `~numpy.ndarray` or `~numpy.ma.MaskedArray`
            The cutout array.
        """
        data = np.asanyarray(data)
        if data.shape != self._segment_data.shape:
            raise ValueError('data must have the same shape as the '
                             'segmentation array.')

        if masked_array:
            mask = (self._segment_data[self.slices] != self.label)
            return np.ma.masked_array(data[self.slices], mask=mask)
        else:
            return data[self.slices]
