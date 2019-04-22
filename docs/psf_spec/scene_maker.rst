SceneMaker
==========

SceneMaker does not yet exist in the framework of PSF Photometry. This API
is therefore a work in progress and should be considered as such.

However, the block will likely involve an additional column or variable attributed
to each source in the input `~astropy.table.Table` indicating whether sources are
stars or more extended objects. This column will be used in conjunction with
``SingleObjectModel``, which already has the framework to accept ``object_type`` which
allows for individual extended sources to be handled, depending on the specific
class used for the specific purpose. The given list of single object models allowed
by, and available to, ``SingleObjectModel`` must be the same list of physical
source classes (stars, galaxies, etc.) as ``SceneMaker`` uses to group and merge
detected sources into point sources or extended objects. The block must also handle
the possibility of merging several point sources into an extended object and vice
versa.

SceneMaker should be considered an extension of GroupMaker, and will likely accept
outputs from that block. Sources grouped together by GroupMaker will subsequently
be assigned as multiple single sources or one larger, extended source erroneously split
up by the Finder, or some combination of the two. This information will then be used by
SingleObjectModel to fit the grouped sources as either individual point sources or
extended sources, based on ``object_type``.
