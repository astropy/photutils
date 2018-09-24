SceneMaker
==========

SceneMaker does not yet exist in the framework of PSF Photometry. However,
it likely will involve an additional column or variable attributed to each
source in the input `~astropy.table.Table` indicating whether sources are
stars or more extended objects. This must handle the possibility of merging
several point sources into an extended object and vice versa. This will
then be used in conjunction with SingleObjectModel, which already has the
framework of ``object_type`` which allows for individual extended sources
to be handled.
