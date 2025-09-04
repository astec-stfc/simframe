.. _simcodes:

SimCodes
=============

.. warning::
    | In case of any issues arising during installation or running :mod:`SimFrame`, contact `Alex Brynes <mailto:alexander.brynes@stfc.ac.uk>`_.

``SimCodes`` is a container for the particle accelerator tracking codes used by ``SimFrame``.

While most of the accelerator code executables are open-source, we prefer not to provide these as an installable
package.

``SimFrame`` does, however, require these codes to be accessible. This functionality is provided in various ways.

Creating a SimCodes Directory
-----------------------------

One can create a top-level directory containing sub-folders for each tracking code, and instantiate ``SimFrame``
with a ``simcodes_location`` argument:

.. code-block:: python

    import SimulationFramework.Framework as fw
    directory = "/path/to/working_directory"
    simcodes_location = "/path/to/simcodes/folder"

    fw = Framework(
        directory=directory,
        simcodes_location=simcodes_location,
    )

Alternatively, one can set up ``SimFrame`` without this argument and set up the ``SimCodes`` location afterwards:

.. code-block:: python

    import SimulationFramework.Framework as fw
    directory = "/path/to/working_directory"
    simcodes_location = "/path/to/simcodes/folder"

    fw = Framework(directory=directory)

    fw.setSimCodesLocation(simcodes_location)

These executables are then accessible to the ``run()`` function of the ``frameworkLattice`` object.

In `Executables.yaml <../../SimulationFramework/Executables.yaml>`_ the required structure is provided for this
schema to work for different hardware architectures, either by the OS type or the computer name.

Editing the Executables.yaml file
---------------------------------

If the user already has these executables installed, they can point directly to them in
`Executables.yaml <../../SimulationFramework/Executables.yaml>`_.

Pointing to a specific location
-------------------------------

An instance of ``SimFrame`` has access to these executables via the ``executables`` attribute, and these
can be modified once ``SimFrame`` is instantiated.

For example, in order to point to a local install of the ``ELEGANT`` code, the user can run the following code:

.. code-block:: python

    import SimulationFramework.Framework as fw
    directory = "/path/to/working_directory"
    elegant_location = "/path/to/elegant/binary"

    fw = Framework(
        directory=directory,
    )

    fw.executables.define_elegant_command(location=elegant_location)

This will then allow ``SimFrame`` to call the correct version of ``ELEGANT``.