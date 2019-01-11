SingleObjectModel
=================

EJT: This does not exist in the current `photutils.psf` model, because there is not
an explicit separate single object model.  Instead the psf_model is used
directly, as the "single object model" is implicitly a delta function.  To
maintain backwards-compatibility, the new ``SingleObjectModel`` will need to
default to the "point source" object model, and behave the same as the current
behavior of a model with shape parameters is provided as the "psf model".  But
arguable that is *not* the desired behavior in the "new" paradigm that combines
the ``SceneMaker``and the ``SingleObjectModel``.

A single sentence summarizing this block.

A longer descrption.  Can be multiple paragraphs.  You can link to other things
like `photutils.background`.

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
