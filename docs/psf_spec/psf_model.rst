PsfModel
========

EJT: the PSF models in the current ``photutils.psf`` are not explicitly
defined as a specific class, but any 2D model (inputs: x,y, output: flux) can
be considered a PSF Model.  The `~photutils.psf.PRFAdapter` class is the
clearest application specific example, however, and demonstrates the required
convention for *names* of the PSF model's parameters.  Hopefully that class can
be used *directly* as this block, as it is meant to wrap any arbitrary other
models to make it compatible with the machinery.

A single sentence summarizing this block.

A longer description. Can be multiple paragraphs. You can link to other
things like `photutils.background`.

Parameters
----------

first_parameter_name : `~astropy.table.Table`
    Description of first input

second_parameter_name : SomeOtherType
    Description of second input (if any)

Returns
-------

first_return : `~astropy.table.Table`
    Description of the first thing this block outputs.

second_return
    Many blocks will only return one object, but if more things are returned
    they can be described here (e.g., in python this is
    ``first, second = some_function(...)``)


Methods
-------

Not all blocks will have these, but if desired some blocks can have methods that
let you do something other than just running the block.  E.g::

    some_block = BlockClassName()
    output = some_block(input1, input2, ...)  # this is what is documented above
    result = some_block.method_name(...)  #this is documented here

method_name
^^^^^^^^^^^

Description of method

Parameters
""""""""""

first_parameter : type
    Description ...

second_parameter : type
    Description ...

Returns
"""""""

first_return : type
    Description ...


Example Usage
-------------

An example of *using* the block should be provided.  This needs to be after a
``::`` in the rst and indented::

    print("This is example code")
