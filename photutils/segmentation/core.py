# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides classes for a segmentation image and a single
segment within a segmentation image.
"""

import inspect
import warnings
from copy import copy, deepcopy

import numpy as np
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning

from photutils.aperture import BoundingBox
from photutils.utils._optional_deps import HAS_RASTERIO, HAS_SHAPELY
from photutils.utils._parameters import as_pair
from photutils.utils.colormaps import make_random_cmap

__all__ = ['SegmentationImage', 'Segment']

__doctest_requires__ = {('SegmentationImage', 'SegmentationImage.*'):
                        ['scipy']}


class SegmentationImage:
    """
    Class for a segmentation image.

    Parameters
    ----------
    data : 2D int `~numpy.ndarray`
        A 2D segmentation array where source regions are labeled by
        different positive integer values. A value of zero is reserved
        for the background. The segmentation image must have integer
        type.

    Notes
    -----
    The `SegmentationImage` instance may be sliced, but note that the
    sliced `SegmentationImage` data array will be a view into the
    original `SegmentationImage` array (this is the same behavior as
    `numpy.ndarray`). Explicitly use the :meth:`SegmentationImage.copy`
    method to create a copy of the sliced `SegmentationImage`.
    """

    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError('input data must be a numpy ndarray')
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

    def __getitem__(self, key):
        """
        Slice the segmentation image, returning a new SegmentationImage
        object.
        """
        if (isinstance(key, tuple) and len(key) == 2
                and all(isinstance(key[i], slice)
                        and (key[i].start != key[i].stop) for i in (0, 1))):
            return SegmentationImage(self.data[key])
        else:
            raise TypeError(f'{key!r} is not a valid 2D slice object')

    def __array__(self):
        """
        Array representation of the segmentation array (e.g., for
        matplotlib).
        """
        return self._data

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

        if HAS_RASTERIO and HAS_SHAPELY:
            for label, slc, bbox, area, polygon in zip(self.labels,
                                                       self.slices,
                                                       self.bbox,
                                                       self.areas,
                                                       self.polygons):
                segments.append(Segment(self.data, label, slc, bbox, area,
                                        polygon=polygon))
        else:
            for label, slc, bbox, area in zip(self.labels, self.slices,
                                              self.bbox, self.areas):
                segments.append(Segment(self.data, label, slc, bbox, area))

        return segments

    @property
    def data(self):
        """The segmentation array."""
        return self._data

    @property
    def _lazyproperties(self):
        """
        A list of all class lazyproperties (even in superclasses).
        """
        def islazyproperty(obj):
            return isinstance(obj, lazyproperty)

        return [i[0] for i in inspect.getmembers(self.__class__,
                                                 predicate=islazyproperty)]

    def _reset_lazyproperties(self):
        for key in self._lazyproperties:
            self.__dict__.pop(key, None)

    @data.setter
    def data(self, value):
        if not np.issubdtype(value.dtype, np.integer):
            raise TypeError('data must be have integer type')

        labels = self._get_labels(value)  # array([]) if value all zeros
        if labels.shape != (0,) and np.min(labels) < 0:
            raise ValueError('The segmentation image cannot contain '
                             'negative integers.')

        if '_data' in self.__dict__:
            # reset cached properties when data is reassigned, but not on init
            self._reset_lazyproperties()

        self._data = value  # pylint: disable=attribute-defined-outside-init
        self.__dict__['labels'] = labels

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
        if '_raw_slices' in self.__dict__:
            labels_all = np.arange(len(self._raw_slices)) + 1
            labels = []
            # if a label is missing, raw_slices will be None instead of a slice
            for label, slc in zip(labels_all, self._raw_slices):
                if slc is not None:
                    labels.append(label)
            return np.array(labels)
        else:
            return self._get_labels(self.data)

    @lazyproperty
    def nlabels(self):
        """The number of non-zero labels in the segmentation array."""
        return len(self.labels)

    @lazyproperty
    def max_label(self):
        """The maximum label in the segmentation array."""
        if self.nlabels == 0:
            return 0
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
        # self.labels is always sorted
        return np.searchsorted(self.labels, label)

    def get_indices(self, labels):
        """
        Find the indices of the input ``labels``.

        Parameters
        ----------
        labels : int, array_like (1D, int)
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
        # self.labels is always sorted
        return np.searchsorted(self.labels, labels)

    @lazyproperty
    def _raw_slices(self):
        from scipy.ndimage import find_objects
        return find_objects(self.data)

    @lazyproperty
    def slices(self):
        """
        A list of tuples, where each tuple contains two slices
        representing the minimal box that contains the labeled region.

        The list starts with the *non-zero* label.  The returned list
        has a length equal to the number of labels and matches the order
        of the ``labels`` attribute.
        """
        return [slc for slc in self._raw_slices if slc is not None]

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
        return self._data.size - np.count_nonzero(self._data)

    @lazyproperty
    def areas(self):
        """
        A 1D array of areas (in pixel**2) of the non-zero labeled
        regions.

        The `~numpy.ndarray` starts with the *non-zero* label.  The
        returned array has a length equal to the number of labels and
        matches the order of the ``labels`` attribute.
        """
        areas = []
        for label, slices in zip(self.labels, self.slices):
            areas.append(np.count_nonzero(self._data[slices] == label))
        return np.array(areas)

    def get_area(self, label):
        """
        The area (in pixel**2) of the region for the input label.

        Parameters
        ----------
        label : int
            The label whose area to return.  Label must be non-zero.

        Returns
        -------
        area : float
            The area of the labeled region.
        """
        return self.get_areas(label)[0]

    def get_areas(self, labels):
        """
        The areas (in pixel**2) of the regions for the input labels.

        Parameters
        ----------
        labels : int, 1D array_like (int)
            The label(s) for which to return areas.  Label must be
            non-zero.

        Returns
        -------
        areas : `~numpy.ndarray`
            The areas of the labeled regions.
        """
        idx = self.get_indices(np.atleast_1d(labels))
        return self.areas[idx]

    @lazyproperty
    def is_consecutive(self):
        """
        Boolean value indicating whether or not the non-zero labels in
        the segmentation array are consecutive and start from 1.
        """
        if self.nlabels == 0:
            return False
        return ((self.labels[-1] - self.labels[0] + 1) == self.nlabels
                and self.labels[0] == 1)

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
        labels : int, 1D array_like (int)
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

        if bad_labels:  # bad_labels is a set
            if len(bad_labels) == 1:
                raise ValueError(f'label {bad_labels} is invalid')
            raise ValueError(f'labels {bad_labels} are invalid')

    def _make_cmap(self, ncolors, background_color='#000000ff', seed=None):
        """
        Define a matplotlib colormap consisting of (random) muted
        colors.

        This is useful for plotting the segmentation array.

        Parameters
        ----------
        ncolors : int
            The number of the colors in the colormap.

        background_color : Matplotlib color, optional
            The color of the first color in the colormap.
            The color may be specified using any of
            the `Matplotlib color formats
            <https://matplotlib.org/stable/tutorials/colors/colors.html>`_.
            This color will be used as the background color (label = 0)
            when plotting the segmentation image. The default color is
            black with alpha=1.0 ('#000000ff').

        seed : int, optional
            A seed to initialize the `numpy.random.BitGenerator`. If
            `None`, then fresh, unpredictable entropy will be pulled
            from the OS. Separate function calls with the same ``seed``
            will generate the same colormap.

        Returns
        -------
        cmap : `matplotlib.colors.ListedColormap`
            The matplotlib colormap with colors in RGBA format.
        """
        if self.nlabels == 0:
            return None

        from matplotlib import colors

        cmap = make_random_cmap(ncolors, seed=seed)

        if background_color is not None:
            cmap.colors[0] = colors.to_rgba(background_color)

        return cmap

    def make_cmap(self, background_color='#000000ff', seed=None):
        """
        Define a matplotlib colormap consisting of (random) muted
        colors.

        This is useful for plotting the segmentation array.

        Parameters
        ----------
        background_color : Matplotlib color, optional
            The color of the first color in the colormap.
            The color may be specified using any of
            the `Matplotlib color formats
            <https://matplotlib.org/stable/tutorials/colors/colors.html>`_.
            This color will be used as the background color (label = 0)
            when plotting the segmentation image. The default color is
            black with alpha=1.0 ('#000000ff').

        seed : int, optional
            A seed to initialize the `numpy.random.BitGenerator`. If
            `None`, then fresh, unpredictable entropy will be pulled
            from the OS. Separate function calls with the same ``seed``
            will generate the same colormap.

        Returns
        -------
        cmap : `matplotlib.colors.ListedColormap`
            The matplotlib colormap with colors in RGBA format.
        """
        return self._make_cmap(self.max_label + 1,
                               background_color=background_color,
                               seed=seed)

    def reset_cmap(self, seed=None):
        """
        Reset the colormap (`cmap` attribute) to a new random colormap.

        Parameters
        ----------
        seed : int, optional
            A seed to initialize the `numpy.random.BitGenerator`. If
            `None`, then fresh, unpredictable entropy will be pulled
            from the OS. Separate function calls with the same ``seed``
            will generate the same colormap.
        """
        self.cmap = self.make_cmap(background_color='#000000ff', seed=seed)

    @lazyproperty
    def cmap(self):
        """
        A matplotlib colormap consisting of (random) muted colors.

        This is useful for plotting the segmentation array.
        """
        return self.make_cmap(background_color='#000000ff', seed=0)

    def reassign_label(self, label, new_label, relabel=False):
        """
        Reassign a label number to a new number.

        If ``new_label`` is already present in the segmentation array,
        then it will be combined with the input ``label`` number.
        Note that this can result in a label that is no longer pixel
        connected.

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
        >>> data = np.array([[1, 1, 0, 0, 4, 4],
        ...                  [0, 0, 0, 0, 0, 4],
        ...                  [0, 0, 3, 3, 0, 0],
        ...                  [7, 0, 0, 0, 0, 5],
        ...                  [7, 7, 0, 5, 5, 5],
        ...                  [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage(data)
        >>> segm.reassign_label(label=1, new_label=2)
        >>> segm.data
        array([[2, 2, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 3, 3, 0, 0],
               [7, 0, 0, 0, 0, 5],
               [7, 7, 0, 5, 5, 5],
               [7, 7, 0, 0, 5, 5]])

        >>> data = np.array([[1, 1, 0, 0, 4, 4],
        ...                  [0, 0, 0, 0, 0, 4],
        ...                  [0, 0, 3, 3, 0, 0],
        ...                  [7, 0, 0, 0, 0, 5],
        ...                  [7, 7, 0, 5, 5, 5],
        ...                  [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage(data)
        >>> segm.reassign_label(label=1, new_label=4)
        >>> segm.data
        array([[4, 4, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 3, 3, 0, 0],
               [7, 0, 0, 0, 0, 5],
               [7, 7, 0, 5, 5, 5],
               [7, 7, 0, 0, 5, 5]])

        >>> data = np.array([[1, 1, 0, 0, 4, 4],
        ...                  [0, 0, 0, 0, 0, 4],
        ...                  [0, 0, 3, 3, 0, 0],
        ...                  [7, 0, 0, 0, 0, 5],
        ...                  [7, 7, 0, 5, 5, 5],
        ...                  [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage(data)
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
        ``new_label`` number. If ``new_label`` is already present in
        the segmentation array, then it will be combined with the input
        ``labels``. Note that both of these can result in a label that
        is no longer pixel connected.

        Parameters
        ----------
        labels : int, array_like (1D, int)
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
        >>> data = np.array([[1, 1, 0, 0, 4, 4],
        ...                  [0, 0, 0, 0, 0, 4],
        ...                  [0, 0, 3, 3, 0, 0],
        ...                  [7, 0, 0, 0, 0, 5],
        ...                  [7, 7, 0, 5, 5, 5],
        ...                  [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage(data)
        >>> segm.reassign_labels(labels=[1, 7], new_label=2)
        >>> segm.data
        array([[2, 2, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 3, 3, 0, 0],
               [2, 0, 0, 0, 0, 5],
               [2, 2, 0, 5, 5, 5],
               [2, 2, 0, 0, 5, 5]])

        >>> data = np.array([[1, 1, 0, 0, 4, 4],
        ...                  [0, 0, 0, 0, 0, 4],
        ...                  [0, 0, 3, 3, 0, 0],
        ...                  [7, 0, 0, 0, 0, 5],
        ...                  [7, 7, 0, 5, 5, 5],
        ...                  [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage(data)
        >>> segm.reassign_labels(labels=[1, 7], new_label=4)
        >>> segm.data
        array([[4, 4, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 3, 3, 0, 0],
               [4, 0, 0, 0, 0, 5],
               [4, 4, 0, 5, 5, 5],
               [4, 4, 0, 0, 5, 5]])

        >>> data = np.array([[1, 1, 0, 0, 4, 4],
        ...                  [0, 0, 0, 0, 0, 4],
        ...                  [0, 0, 3, 3, 0, 0],
        ...                  [7, 0, 0, 0, 0, 5],
        ...                  [7, 7, 0, 5, 5, 5],
        ...                  [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage(data)
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
        self._reset_lazyproperties()  # reset all cached properties
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
        >>> data = np.array([[1, 1, 0, 0, 4, 4],
        ...                  [0, 0, 0, 0, 0, 4],
        ...                  [0, 0, 3, 3, 0, 0],
        ...                  [7, 0, 0, 0, 0, 5],
        ...                  [7, 7, 0, 5, 5, 5],
        ...                  [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage(data)
        >>> segm.relabel_consecutive()
        >>> segm.data
        array([[1, 1, 0, 0, 3, 3],
               [0, 0, 0, 0, 0, 3],
               [0, 0, 2, 2, 0, 0],
               [5, 0, 0, 0, 0, 4],
               [5, 5, 0, 4, 4, 4],
               [5, 5, 0, 0, 4, 4]])
        """
        if self.nlabels == 0:
            warnings.warn('Cannot relabel a segmentation image of all zeros',
                          AstropyUserWarning)
            return

        if start_label <= 0:
            raise ValueError('start_label must be > 0.')

        if ((self.labels[0] == start_label)
                and (self.labels[-1] - self.labels[0] + 1) == self.nlabels):
            return

        old_slices = self.__dict__.get('slices', None)
        new_labels = np.arange(self.nlabels) + start_label
        new_label_map = np.zeros(self.max_label + 1, dtype=int)
        new_label_map[self.labels] = new_labels

        data_new = new_label_map[self.data]
        self._reset_lazyproperties()  # reset all cached properties
        self._data = data_new  # use _data to avoid validation
        self.__dict__['labels'] = new_labels
        if old_slices is not None:
            self.__dict__['slices'] = old_slices  # slice order is unchanged

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
        >>> data = np.array([[1, 1, 0, 0, 4, 4],
        ...                  [0, 0, 0, 0, 0, 4],
        ...                  [0, 0, 3, 3, 0, 0],
        ...                  [7, 0, 0, 0, 0, 5],
        ...                  [7, 7, 0, 5, 5, 5],
        ...                  [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage(data)
        >>> segm.keep_label(label=3)
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 3, 3, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]])

        >>> data = np.array([[1, 1, 0, 0, 4, 4],
        ...                  [0, 0, 0, 0, 0, 4],
        ...                  [0, 0, 3, 3, 0, 0],
        ...                  [7, 0, 0, 0, 0, 5],
        ...                  [7, 7, 0, 5, 5, 5],
        ...                  [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage(data)
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
        labels : int, array_like (1D, int)
            The label number(s) to keep.

        relabel : bool, optional
            If `True`, then the segmentation array will be relabeled
            such that the labels are in consecutive order starting from
            1.

        Examples
        --------
        >>> from photutils.segmentation import SegmentationImage
        >>> data = np.array([[1, 1, 0, 0, 4, 4],
        ...                  [0, 0, 0, 0, 0, 4],
        ...                  [0, 0, 3, 3, 0, 0],
        ...                  [7, 0, 0, 0, 0, 5],
        ...                  [7, 7, 0, 5, 5, 5],
        ...                  [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage(data)
        >>> segm.keep_labels(labels=[5, 3])
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 3, 3, 0, 0],
               [0, 0, 0, 0, 0, 5],
               [0, 0, 0, 5, 5, 5],
               [0, 0, 0, 0, 5, 5]])

        >>> data = np.array([[1, 1, 0, 0, 4, 4],
        ...                  [0, 0, 0, 0, 0, 4],
        ...                  [0, 0, 3, 3, 0, 0],
        ...                  [7, 0, 0, 0, 0, 5],
        ...                  [7, 7, 0, 5, 5, 5],
        ...                  [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage(data)
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
        >>> data = np.array([[1, 1, 0, 0, 4, 4],
        ...                  [0, 0, 0, 0, 0, 4],
        ...                  [0, 0, 3, 3, 0, 0],
        ...                  [7, 0, 0, 0, 0, 5],
        ...                  [7, 7, 0, 5, 5, 5],
        ...                  [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage(data)
        >>> segm.remove_label(label=5)
        >>> segm.data
        array([[1, 1, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 3, 3, 0, 0],
               [7, 0, 0, 0, 0, 0],
               [7, 7, 0, 0, 0, 0],
               [7, 7, 0, 0, 0, 0]])

        >>> data = np.array([[1, 1, 0, 0, 4, 4],
        ...                  [0, 0, 0, 0, 0, 4],
        ...                  [0, 0, 3, 3, 0, 0],
        ...                  [7, 0, 0, 0, 0, 5],
        ...                  [7, 7, 0, 5, 5, 5],
        ...                  [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage(data)
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
        labels : int, array_like (1D, int)
            The label number(s) to remove.

        relabel : bool, optional
            If `True`, then the segmentation array will be relabeled
            such that the labels are in consecutive order starting from
            1.

        Examples
        --------
        >>> from photutils.segmentation import SegmentationImage
        >>> data = np.array([[1, 1, 0, 0, 4, 4],
        ...                  [0, 0, 0, 0, 0, 4],
        ...                  [0, 0, 3, 3, 0, 0],
        ...                  [7, 0, 0, 0, 0, 5],
        ...                  [7, 7, 0, 5, 5, 5],
        ...                  [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage(data)
        >>> segm.remove_labels(labels=[5, 3])
        >>> segm.data
        array([[1, 1, 0, 0, 4, 4],
               [0, 0, 0, 0, 0, 4],
               [0, 0, 0, 0, 0, 0],
               [7, 0, 0, 0, 0, 0],
               [7, 7, 0, 0, 0, 0],
               [7, 7, 0, 0, 0, 0]])

        >>> data = np.array([[1, 1, 0, 0, 4, 4],
        ...                  [0, 0, 0, 0, 0, 4],
        ...                  [0, 0, 3, 3, 0, 0],
        ...                  [7, 0, 0, 0, 0, 5],
        ...                  [7, 7, 0, 5, 5, 5],
        ...                  [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage(data)
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
        >>> data = np.array([[1, 1, 0, 0, 4, 4],
        ...                  [0, 0, 0, 0, 0, 4],
        ...                  [0, 0, 3, 3, 0, 0],
        ...                  [7, 0, 0, 0, 0, 5],
        ...                  [7, 7, 0, 5, 5, 5],
        ...                  [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage(data)
        >>> segm.remove_border_labels(border_width=1)
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 3, 3, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]])

        >>> data = np.array([[1, 1, 0, 0, 4, 4],
        ...                  [0, 0, 0, 0, 0, 4],
        ...                  [0, 0, 3, 3, 0, 0],
        ...                  [7, 0, 0, 0, 0, 5],
        ...                  [7, 7, 0, 5, 5, 5],
        ...                  [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage(data)
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
        >>> data = np.array([[1, 1, 0, 0, 4, 4],
        ...                  [0, 0, 0, 0, 0, 4],
        ...                  [0, 0, 3, 3, 0, 0],
        ...                  [7, 0, 0, 0, 0, 5],
        ...                  [7, 7, 0, 5, 5, 5],
        ...                  [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage(data)
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

        >>> data = np.array([[1, 1, 0, 0, 4, 4],
        ...                  [0, 0, 0, 0, 0, 4],
        ...                  [0, 0, 3, 3, 0, 0],
        ...                  [7, 0, 0, 0, 0, 5],
        ...                  [7, 7, 0, 5, 5, 5],
        ...                  [7, 7, 0, 0, 5, 5]])
        >>> segm = SegmentationImage(data)
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

    def make_source_mask(self, *, size=None, footprint=None):
        """
        Make a source mask from the segmentation image.

        Use the ``size`` or ``footprint`` keyword to perform binary
        dilation on the segmentation image mask.

        Parameters
        ----------
        size : int or tuple of int, optional
            The size along each axis of the rectangular footprint used
            for the source dilation. If ``size`` is a scalar, then a
            square footprint of ``size`` will be used. If ``size`` has
            two elements, they must be in ``(ny, nx)`` order. ``size``
            should have odd values for each axis. To perform source
            dilation, either ``size`` or ``footprint`` must be defined.
            If they are both defined, then ``footprint`` overrides
            ``size``.

        footprint : 2D `~numpy.ndarray`, optional
            The local footprint used for the source dilation. Non-zero
            elements are considered `True`. ``size=(n, m)`` is
            equivalent to ``footprint=np.ones((n, m))``. To perform
            source dilation, either ``size`` or ``footprint`` must
            be defined. If they are both defined, then ``footprint``
            overrides ``size``.

        Returns
        -------
        mask : 2D bool `~numpy.ndarray`
            A 2D boolean image containing the source mask.

        Notes
        -----
        When performing source dilation, using a square footprint
        will be much faster than using other shapes (e.g., a circular
        footprint). Source dilation also is slower for larger images and
        larger footprints.

        Examples
        --------
        >>> import numpy as np
        >>> from photutils.segmentation import SegmentationImage
        >>> from photutils.utils import circular_footprint
        >>> data = np.zeros((7, 7), dtype=int)
        >>> data[3, 3] = 1
        >>> segm = SegmentationImage(data)
        >>> segm.data
        array([[0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0]])

        >>> mask0 = segm.make_source_mask()
        >>> mask0
        array([[False, False, False, False, False, False, False],
               [False, False, False, False, False, False, False],
               [False, False, False, False, False, False, False],
               [False, False, False,  True, False, False, False],
               [False, False, False, False, False, False, False],
               [False, False, False, False, False, False, False],
               [False, False, False, False, False, False, False]])

        >>> mask1 = segm.make_source_mask(size=3)
        >>> mask1
        array([[False, False, False, False, False, False, False],
               [False, False, False, False, False, False, False],
               [False, False,  True,  True,  True, False, False],
               [False, False,  True,  True,  True, False, False],
               [False, False,  True,  True,  True, False, False],
               [False, False, False, False, False, False, False],
               [False, False, False, False, False, False, False]])

        >>> footprint = circular_footprint(radius=3)
        >>> mask2 = segm.make_source_mask(footprint=footprint)
        >>> mask2
        array([[False, False, False,  True, False, False, False],
               [False,  True,  True,  True,  True,  True, False],
               [False,  True,  True,  True,  True,  True, False],
               [ True,  True,  True,  True,  True,  True,  True],
               [False,  True,  True,  True,  True,  True, False],
               [False,  True,  True,  True,  True,  True, False],
               [False, False, False,  True, False, False, False]])
        """
        mask = self._data.astype(bool)

        if footprint is None:
            if size is None:
                return mask
            else:
                size = as_pair('size', size, check_odd=False)
                footprint = np.ones(size, dtype=bool)
        footprint = footprint.astype(bool)

        if np.all(footprint):
            # With a rectangular footprint, scipy grey_dilation is
            # currently much faster than binary_dilation (separable
            # footprint). grey_dilation and binary_dilation are identical
            # for binary inputs (equivalent to a 2D maximum filter).
            from scipy.ndimage import grey_dilation
            return grey_dilation(mask, footprint=footprint)
        else:
            # Binary dilation is very slow, especially for large
            # footprints. The following is a faster implementation
            # using fast Fourier transforms (FFTs) that gives identical
            # results to binary_dilation. Based on the following paper:
            # "Dilation and Erosion of Gray Images with Spherical
            # Masks", J. Kukal, D. Majerova, A. Prochazka (Jan 2007).
            # https://www.researchgate.net/publication/238778666_DILATION_AND_EROSION_OF_GRAY_IMAGES_WITH_SPHERICAL_MASKS
            from scipy.signal import fftconvolve
            return fftconvolve(mask, footprint, 'same') > 0.5

    @lazyproperty
    def _geo_polygons(self):
        """
        A list of polygons representing each source segment.

        Each item in the list is tuple of (polygon, value) where the
        polygon is a GeoJSON-like dict and the value is the label from
        the segmentation image.

        Note that the coordinates of these polygon vertices are in a
        reference frame with the (0, 0) origin at the *lower-left*
        corner of the lower-left pixel.
        """
        from rasterio.features import shapes

        polygons = list(shapes(self.data.astype('int32'), connectivity=8))
        polygons.sort(key=lambda x: x[1])  # sort in label order

        # do not include polygons for background (label = 0)
        return polygons[1:]

    @lazyproperty
    def polygons(self):
        """
        A list of `Shapely <https://shapely.readthedocs.io/en/stable/>`_
        polygons representing each source segment.
        """
        from shapely import transform
        from shapely.geometry import shape

        polygons = []
        for geo_poly in self._geo_polygons:
            polygons.append(shape(geo_poly[0]))
        # shift the vertices so that the (0, 0) origin is at the
        # center of the lower-left pixel
        polygons = transform(polygons, lambda x: x - [0.5, 0.5])

        return polygons

    def to_patches(self, *, origin=(0, 0), scale=1.0, **kwargs):
        """
        Return a list of `~matplotlib.patches.Polygon` objects
        representing each source segment.

        By default, the polygon patch will have a white edge color and
        no face color.

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image. This effectively translates the position of the
            polygons.

        scale : float, optional
            The scale factor applied to the polygon vertices.

        **kwargs : `dict`
            Any keyword arguments accepted by
            `matplotlib.patches.Polygon`.
        """
        from matplotlib.patches import Polygon

        origin = np.array(origin)
        patch_kwargs = {'edgecolor': 'white', 'facecolor': 'none'}
        patch_kwargs.update(kwargs)

        patches = []
        for poly in self.polygons:
            xy = np.array(poly.exterior.coords)
            xy = scale * (xy + 0.5) - 0.5
            xy -= origin
            patches.append(Polygon(xy, **patch_kwargs))

        return patches

    def plot_patches(self, *, ax=None, origin=(0, 0), scale=1.0, labels=None,
                     **kwargs):
        """
        Plot the `~matplotlib.patches.Polygon` objects for the source
        segments on a matplotlib `~matplotlib.axes.Axes` instance.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes` or `None`, optional
            The matplotlib axes on which to plot.  If `None`, then the
            current `~matplotlib.axes.Axes` instance is used.

        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        scale : float, optional
            The scale factor applied to the polygon vertices.

        labels : int or array of int, optional
            The label numbers whose polygons are to be plotted. If
            `None`, the polygons for all labels will be plotted.

        **kwargs : `dict`
            Any keyword arguments accepted by
            `matplotlib.patches.Polygon`.

        Returns
        -------
        patches : list of `~matplotlib.patches.Polygon`
            A list of matplotlib polygon patches for the plotted
            polygons. The patches can be used, for example, when adding
            a plot legend.

        Examples
        --------
        .. plot::
            :include-source:

            import numpy as np
            from photutils.segmentation import SegmentationImage

            data = np.array([[1, 1, 0, 0, 4, 4],
                             [0, 0, 0, 0, 0, 4],
                             [0, 0, 3, 3, 0, 0],
                             [7, 0, 0, 0, 0, 5],
                             [7, 7, 0, 5, 5, 5],
                             [7, 7, 0, 0, 5, 5]])
            segm = SegmentationImage(data)
            segm.imshow(figsize=(5, 5))
            segm.plot_patches(edgecolor='white', lw=2)
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        patches = self.to_patches(origin=origin, scale=scale, **kwargs)
        if labels is not None:
            patches = np.array(patches)
            indices = self.get_indices(labels)
            patches = patches[indices]
            if np.isscalar(labels):
                patches = [patches]

        for patch in patches:
            patch = copy(patch)
            ax.add_patch(patch)

        if labels is not None:
            patches = list(patches)

        return patches

    def imshow(self, ax=None, figsize=None, dpi=None, cmap=None, alpha=None):
        """
        Display the segmentation image in a matplotlib
        `~matplotlib.axes.Axes` instance.

        The segmentation image will be displayed with "nearest"
        interpolation and with the origin set to "lower".

        Parameters
        ----------
        ax : `matplotlib.axes.Axes` or `None`, optional
            The matplotlib axes on which to plot. If `None`, then a new
            `~matplotlib.axes.Axes` instance will be created.

        figsize : 2-tuple of floats or `None`, optional
            The figure dimension (width, height) in inches when creating
            a new Axes. This keyword is ignored if ``axes`` is input.

        dpi : float or `None`, optional
            The figure dots per inch when creating a new Axes. This
            keyword is ignored if ``axes`` is input.

        cmap : `matplotlib.colors.Colormap`, str, or `None`, optional
            The `~matplotlib.colors.Colormap` instance or a registered
            matplotlib colormap name used to map scalar data to colors.
            If `None`, then the colormap defined by the `cmap` attribute
            will be used.

        alpha : float, array_like, or `None`, optional
            The alpha blending value, between 0 (transparent) and 1
            (opaque). If alpha is an array, the alpha blending values
            are applied pixel by pixel, and alpha must have the same
            shape as the segmentation image.

        Returns
        -------
        result : `matplotlib.image.AxesImage`
            An image attached to an `matplotlib.axes.Axes`.

        Examples
        --------
        .. plot::
            :include-source:

            import numpy as np
            from photutils.segmentation import SegmentationImage

            data = np.array([[1, 1, 0, 0, 4, 4],
                             [0, 0, 0, 0, 0, 4],
                             [0, 0, 3, 3, 0, 0],
                             [7, 0, 0, 0, 0, 5],
                             [7, 7, 0, 5, 5, 5],
                             [7, 7, 0, 0, 5, 5]])
            segm = SegmentationImage(data)

            fig, ax = plt.subplots()
            im = segm.imshow(ax=ax)
            fig.colorbar(im, ax=ax)
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=figsize, dpi=dpi)
        if cmap is None:
            cmap = self.cmap

        return ax.imshow(self.data, cmap=cmap, interpolation='nearest',
                         origin='lower', alpha=alpha, vmin=-0.5,
                         vmax=self.max_label + 0.5)

    def imshow_map(self, ax=None, figsize=None, dpi=None, cmap=None,
                   alpha=None, max_labels=25, cbar_labelsize=None):
        """
        Display the segmentation image in a matplotlib
        `~matplotlib.axes.Axes` instance with a colorbar.

        This method is useful for displaying segmentation images that
        have a small number of labels (e.g., from a cutout) that are
        not consecutive. It maps the labels to be consecutive integers
        starting from 1 before plotting. The plotted image values are
        not the label values, but the colorbar tick labels are used to
        show the original labels.

        The segmentation image will be displayed with "nearest"
        interpolation and with the origin set to "lower".

        Parameters
        ----------
        ax : `matplotlib.axes.Axes` or `None`, optional
            The matplotlib axes on which to plot. If `None`, then a new
            `~matplotlib.axes.Axes` instance will be created.

        figsize : 2-tuple of floats or `None`, optional
            The figure dimension (width, height) in inches when creating
            a new Axes. This keyword is ignored if ``axes`` is input.

        dpi : float or `None`, optional
            The figure dots per inch when creating a new Axes. This
            keyword is ignored if ``axes`` is input.

        cmap : `matplotlib.colors.Colormap`, str, or `None`, optional
            The `~matplotlib.colors.Colormap` instance or a registered
            matplotlib colormap name used to map scalar data to colors.
            If `None`, then the colormap defined by the `cmap` attribute
            will be used.

        alpha : float, array_like, or `None`, optional
            The alpha blending value, between 0 (transparent) and 1
            (opaque). If alpha is an array, the alpha blending values
            are applied pixel by pixel, and alpha must have the same
            shape as the segmentation image.

        max_labels : int, optional
            The maximum number of labels to display in the colorbar. If
            the number of labels is greater than ``max_labels``, then
            the colorbar will not be displayed.

        cbar_labelsize : `None` or float, optional
            The font size of the colorbar tick labels.

        Returns
        -------
        result : `matplotlib.image.AxesImage`
            An image attached to an `matplotlib.axes.Axes`.
        cbar_info : tuple or `None`
            The colorbar information as a tuple containing the
            `~matplotlib.colorbar.Colorbar` instance, a `~numpy.ndarray`
            of tick positions, and a `~numpy.ndarray` of tick labels.
            `None` is returned if the colorbar was not plotted.

        Examples
        --------
        .. plot::
            :include-source:

            import numpy as np
            from photutils.segmentation import SegmentationImage

            data = np.array([[1, 1, 0, 0, 4, 4],
                             [0, 0, 0, 0, 0, 4],
                             [0, 0, 3, 3, 0, 0],
                             [7, 0, 0, 0, 0, 5],
                             [7, 7, 0, 5, 5, 5],
                             [7, 7, 0, 0, 5, 5]])
            data *= 1000
            segm = SegmentationImage(data)

            fig, ax = plt.subplots()
            im, cbar = segm.imshow_map(ax=ax)
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        if ax is None:
            _, ax = plt.subplots(figsize=figsize, dpi=dpi)

        data, idx = np.unique(self.data, return_inverse=True)
        idx = idx.reshape(self.data.shape)
        vmin = -0.5
        vmax = np.max(idx) + 0.5

        # keep the original cmap colors for the labels
        if cmap is None:
            cmap = ListedColormap(self.cmap.colors[data])

        im = ax.imshow(idx, cmap=cmap, interpolation='nearest', origin='lower',
                       alpha=alpha, vmin=vmin, vmax=vmax)

        cbar_info = None
        cbar_labels = np.hstack((0, self.labels))
        if len(cbar_labels) <= max_labels:
            cbar_ticks = np.arange(len(cbar_labels))
            cbar = plt.colorbar(im, ax=ax, ticks=cbar_ticks)
            cbar.ax.set_yticklabels(cbar_labels)
            if cbar_labelsize is not None:
                cbar.ax.yaxis.set_tick_params(labelsize=cbar_labelsize)
            cbar_info = (cbar, cbar_ticks, cbar_labels)
        else:
            warnings.warn('The colorbar was not plotted because the number of '
                          f'labels is greater than {max_labels=}.',
                          AstropyUserWarning)

        return im, cbar_info


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

    polygon : Shapely polygon, optional
        The outline of the segment as a `Shapely
        <https://shapely.readthedocs.io/en/stable/>`_ polygon.
    """

    def __init__(self, segment_data, label, slices, bbox, area, *,
                 polygon=None):
        self._segment_data = segment_data
        self.label = label
        self.slices = slices
        self.bbox = bbox
        self.area = area
        self.polygon = polygon

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

    def _repr_svg_(self):  # pragma: no cover
        if self.polygon is not None:
            print(repr(self))
            return self.polygon._repr_svg_()
        return None

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

        If ``masked_array`` is `True`, then the returned cutout array is
        a `~numpy.ma.MaskedArray`, where the mask is `True` for pixels
        outside of the segment (labeled region). The data part of the
        masked array is a view (not a copy) of the input ``data``.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The data array from which to create the masked cutout array.
            ``data`` must have the same shape as the segmentation array.

        masked_array : bool, optional
            If `True` then a `~numpy.ma.MaskedArray` will be created
            where the mask is `True` for pixels outside of the segment
            (labeled region).  If `False`, then a `~numpy.ndarray` will
            be generated.

        Returns
        -------
        result : 2D `~numpy.ndarray` or `~numpy.ma.MaskedArray`
            The cutout array.
        """
        if data.shape != self._segment_data.shape:
            raise ValueError('data must have the same shape as the '
                             'segmentation array.')

        if masked_array:
            mask = (self._segment_data[self.slices] != self.label)
            return np.ma.masked_array(data[self.slices], mask=mask)
        else:
            return data[self.slices]
