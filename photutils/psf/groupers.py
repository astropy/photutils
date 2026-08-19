# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for performing grouping of stars.
"""

from collections import defaultdict
from functools import cached_property

import numpy as np
from scipy.spatial import cKDTree

from photutils.aperture import CircularAperture
from photutils.utils import make_random_cmap
from photutils.utils._deprecation import deprecated_positional_kwargs
from photutils.utils._repr import make_repr

__all__ = ['SourceGrouper', 'SourceGroups']


class SourceGroups:
    """
    Container for source grouping results with analysis methods.

    This class stores the results of grouping sources and provides
    methods to analyze and query the grouping.

    Parameters
    ----------
    x, y : 1D float `~numpy.ndarray`
        The 1D arrays of the x and y coordinates of the sources.

    groups : 1D int `~numpy.ndarray`
        A 1D array of the group IDs, in the same order as the input x
        and y coordinates.

    Attributes
    ----------
    x : `~numpy.ndarray`
        The x coordinates of the sources.

    y : `~numpy.ndarray`
        The y coordinates of the sources.

    groups : `~numpy.ndarray`
        The group IDs for each source.

    n_sources : int
        Total number of sources.

    n_groups : int
        Total number of groups.

    See Also
    --------
    SourceGrouper

    Examples
    --------
    Create a SourceGroups object:

    >>> from photutils.psf import SourceGroups
    >>> import numpy as np
    >>> x = np.array([10, 15, 50, 55, 100])
    >>> y = np.array([20, 25, 60, 65, 90])
    >>> groups = np.array([1, 1, 2, 2, 3])
    >>> source_groups = SourceGroups(x, y, groups)
    >>> print(source_groups)
    <SourceGroups(n_sources=5, n_groups=3)>

    Access properties of the SourceGroups object:

    >>> print(f'Number of groups: {source_groups.n_groups}')
    Number of groups: 3
    >>> source_groups.size_map
    {1: 2, 2: 2, 3: 1}
    >>> source_groups.sizes
    array([2, 2, 2, 2, 1])
    >>> source_groups.group_centers
    {1: (12.5, 22.5), 2: (52.5, 62.5), 3: (100.0, 90.0)}
    >>> x_group1, y_group1 = source_groups.get_group_sources(1)
    >>> print(x_group1, y_group1)
    [10 15] [20 25]
    """

    def __init__(self, x, y, groups):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.groups = np.asarray(groups)

        if self.x.shape != self.y.shape or self.x.shape != self.groups.shape:
            msg = 'x, y, and groups must have the same shape'
            raise ValueError(msg)

        if not np.isfinite(self.x).all():
            msg = 'x coordinates must be finite (no NaN or inf values)'
            raise ValueError(msg)
        if not np.isfinite(self.y).all():
            msg = 'y coordinates must be finite (no NaN or inf values)'
            raise ValueError(msg)

        self.n_sources = len(self.groups)

        unique_groups, counts = np.unique(self.groups, return_counts=True)
        self._unique_groups = unique_groups
        self._group_counts = counts
        self.n_groups = len(unique_groups)

    def __repr__(self):
        params = ['n_sources', 'n_groups']
        return make_repr(self, params, brackets=True)

    def __len__(self):
        """
        Return the number of sources.
        """
        return self.n_sources

    @cached_property
    def size_map(self):
        """
        Mapping of group ID to group size.

        Returns
        -------
        size_map : dict
            A dictionary where keys are group IDs and values are the
            corresponding group sizes.
        """
        return dict(zip(self._unique_groups.tolist(),
                        self._group_counts.tolist(), strict=True))

    @cached_property
    def sizes(self):
        """
        Size of each group for each source.

        Returns
        -------
        sizes : 1D int `~numpy.ndarray`
            A 1D array of the group sizes, in the same order as the
            sources. Each element indicates how many sources are in the
            same group as the corresponding source.
        """
        return np.array([self.size_map[group] for group in self.groups])

    @cached_property
    def group_centers(self):
        """
        Centroid coordinates of each group.

        Returns
        -------
        group_centers : dict
            A dictionary where keys are group IDs and values are tuples
            of (x_center, y_center) representing the centroid of each
            group.
        """
        group_centers = {}
        for group_id in self._unique_groups.tolist():
            mask = self.groups == group_id
            x_center = np.mean(self.x[mask]).item()
            y_center = np.mean(self.y[mask]).item()
            group_centers[group_id] = (x_center, y_center)

        return group_centers

    def get_group_sources(self, group_id):
        """
        Get the coordinates of all sources in a specific group.

        Parameters
        ----------
        group_id : int
            The group ID to retrieve sources for.

        Returns
        -------
        x, y : `~numpy.ndarray`
            Arrays of x and y coordinates for all sources in the
            specified group.
        """
        if group_id not in self.groups:
            msg = f'Group ID {group_id} not found in groups'
            raise ValueError(msg)

        mask = self.groups == group_id
        return self.x[mask], self.y[mask]

    def plot(self, radius, *, ax=None, cmap=None, seed=0, label_groups=False,
             label_kwargs=None, label_offset=(0, 0), **kwargs):
        """
        Plot circular apertures around sources, color-coded by group.

        Parameters
        ----------
        radius : float
            The radius of the circles to plot around each source (in
            pixels).

        ax : `~matplotlib.axes.Axes`, optional
            The matplotlib axes on which to plot. If None, uses the
            current axes.

        cmap : `~matplotlib.colors.Colormap` or str, optional
            The colormap to use for group colors. If None, a random
            colormap is generated.

        seed : int, optional
            Random seed for generating the colormap if ``cmap`` is None.

        label_groups : bool, optional
            Whether to label each group with its group ID at the group
            center.

        label_kwargs : dict, optional
            Keyword arguments passed to ``ax.text`` for plotting group ID
            labels.

        label_offset : tuple of float, optional
            Offset (dx, dy) in pixels for positioning labels relative to
            group centers. Positive values move right/up, negative values
            move left/down. Default is (0, 0).

        **kwargs
            Additional keyword arguments passed to
            `~photutils.aperture.CircularAperture.plot`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        import matplotlib.pyplot as plt
        from matplotlib import colormaps

        if ax is None:
            ax = plt.gca()

        if cmap is None:
            cmap = make_random_cmap(n_colors=self.n_groups, seed=seed)
        elif isinstance(cmap, str):
            cmap = colormaps[cmap]

        # Sample the colormap over its full range so that the group
        # colors are distinct. Indexing a colormap by the group number
        # would map every group to the low end of the colormap.
        fractions = np.arange(self.n_groups) / max(self.n_groups - 1, 1)
        colors = cmap(fractions)

        # Merge user label kwargs with the defaults
        default_label_kwargs = {'ha': 'center',
                                'va': 'center',
                                'zorder': 10}
        label_kwargs = {**default_label_kwargs, **(label_kwargs or {})}

        # A user-supplied color overrides the per-group colors
        user_color = kwargs.pop('color', None)

        # Get label offset
        label_dx, label_dy = label_offset

        for i, group_id in enumerate(self._unique_groups):
            mask = self.groups == group_id
            xypos = zip(self.x[mask], self.y[mask], strict=True)
            ap = CircularAperture(xypos, r=radius)
            color = colors[i] if user_color is None else user_color
            ap.plot(ax=ax, color=color, **kwargs)

            if label_groups:
                # Add group ID label with offset
                x_center, y_center = self.group_centers[group_id]
                label_x = x_center + label_dx
                label_y = y_center + label_dy
                ax.text(label_x, label_y, f'{group_id}', color=color,
                        **label_kwargs)

        return ax


class SourceGrouper:
    """
    Class to group sources into clusters based on a minimum separation
    distance.

    The groups are the connected components of the graph linking pairs
    of sources separated by less than ``min_separation``. This is
    identical to single-linkage hierarchical agglomerative clustering
    with a distance criterion, but is computed using a KD-tree so that
    it scales to large numbers of sources.

    Parameters
    ----------
    min_separation : float
        The minimum distance (in pixels) such that any two sources
        separated by less than this distance will be placed in the same
        group.

    See Also
    --------
    SourceGroups

    Examples
    --------
    Create a SourceGrouper with a minimum separation of 10 pixels:

    >>> from photutils.psf import SourceGrouper
    >>> import numpy as np
    >>> grouper = SourceGrouper(min_separation=10)

    Group sources and get group IDs as an array (default behavior):

    >>> x = np.array([10, 15, 50, 55, 100])
    >>> y = np.array([20, 25, 60, 65, 90])
    >>> group_ids = grouper(x, y)
    >>> print(group_ids)
    [1 1 2 2 3]

    Optionally, get a SourceGroups object with additional analysis methods:

    >>> groups = grouper(x, y, return_groups_object=True)
    >>> print(groups)
    <SourceGroups(n_sources=5, n_groups=3)>

    Access properties of the SourceGroups object:

    >>> print(f'Number of groups: {groups.n_groups}')
    Number of groups: 3
    >>> groups.size_map
    {1: 2, 2: 2, 3: 1}

    Retrieve the (x, y) positions of sources from a specific group:

    >>> x_group1, y_group1 = groups.get_group_sources(1)
    >>> print(x_group1, y_group1)
    [10 15] [20 25]
    """

    def __init__(self, min_separation):
        if not np.isfinite(min_separation) or min_separation <= 0:
            msg = 'min_separation must be a positive finite value'
            raise ValueError(msg)
        self.min_separation = min_separation

    def __repr__(self):
        return make_repr(self, 'min_separation')

    def _compute_groups(self, x, y):
        """
        Group sources into clusters based on a minimum distance
        criterion.

        Parameters
        ----------
        x, y : 1D float `~numpy.ndarray`
            The 1D arrays of the x and y coordinates of the sources.

        Returns
        -------
        result : 1D int `~numpy.ndarray`
            A 1D array of the groups, in the same order as the input x
            and y coordinates.
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        if x.ndim != 1 or y.ndim != 1:
            msg = 'x and y must be 1D arrays'
            raise ValueError(msg)
        if x.shape != y.shape:
            msg = (f'x and y must have the same shape, got x.shape={x.shape} '
                   f'and y.shape={y.shape}')
            raise ValueError(msg)
        if x.shape == (0,):
            msg = 'x and y must not be empty'
            raise ValueError(msg)
        if not np.isfinite(x).all():
            msg = 'x coordinates must be finite (no NaN or inf values)'
            raise ValueError(msg)
        if not np.isfinite(y).all():
            msg = 'y coordinates must be finite (no NaN or inf values)'
            raise ValueError(msg)

        # A single source forms its own group
        if x.shape == (1,):
            return np.array([1])

        # Find all pairs of sources separated by min_separation or less.
        # query_pairs includes pairs at exactly min_separation, matching
        # the inclusive fclusterdata distance criterion.
        xypos = np.column_stack((x, y))
        tree = cKDTree(xypos)
        pairs = tree.query_pairs(self.min_separation,
                                 output_type='ndarray')

        # Union-find over the pairs to get the connected components,
        # which for single linkage are identical to the clusters from
        # hierarchical clustering with a distance criterion.
        parent = np.arange(len(x))

        def find(idx):
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        for i, j in pairs:
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i

        cluster_labels = [find(i) for i in range(len(x))]

        # Reorder cluster labels to start from 1 and increase
        # sequentially (this matches the behavior of DBSCAN and other
        # algorithms). Use defaultdict for efficient mapping by order of
        # first appearance.
        mapping = defaultdict(lambda: len(mapping) + 1)
        return np.array([mapping[label] for label in cluster_labels])

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def __call__(self, x, y, return_groups_object=False):
        """
        Group sources into clusters based on a minimum distance
        criterion.

        Parameters
        ----------
        x, y : 1D float `~numpy.ndarray`
            The 1D arrays of the x and y coordinates of the sources.

        return_groups_object : bool, optional
            If `False` (default), return a 1D array of group IDs.
            If `True`, return a `SourceGroups` object containing the
            grouping results along with analysis methods.

        Returns
        -------
        result : `~numpy.ndarray` or `SourceGroups`
            If ``return_groups_object=False`` (default), returns a 1D
            integer array of group IDs for each source, in the same
            order as the input coordinates.

            If ``return_groups_object=True``, returns a `SourceGroups`
            object containing the grouping results. The object provides:

            - ``groups`` : array of group IDs for each source
            - ``n_sources`` : total number of sources
            - ``n_groups`` : total number of groups
            - ``sizes`` : group size for each source
            - ``group_centers`` : centroid coordinates for each group
            - ``get_group_sources(group_id)`` : retrieve sources in a
              specific group
            - ``plot()`` : visualize the grouping with color-coded
              apertures

        Examples
        --------
        Get group IDs as an array (default behavior):

        >>> from photutils.psf import SourceGrouper
        >>> import numpy as np
        >>> x = np.array([10, 15, 50])
        >>> y = np.array([20, 25, 60])
        >>> grouper = SourceGrouper(min_separation=10)
        >>> group_ids = grouper(x, y)
        >>> print(group_ids)
        [1 1 2]

        Get a SourceGroups object with additional analysis methods:

        >>> groups = grouper(x, y, return_groups_object=True)
        >>> print(groups.n_groups)
        2
        >>> print(groups.groups)
        [1 1 2]
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        groups = self._compute_groups(x, y)
        if return_groups_object:
            return SourceGroups(x, y, groups)
        return groups
