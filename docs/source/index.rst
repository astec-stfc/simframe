.. SimFrame documentation master file, created by
   sphinx-quickstart on Tue Sep 24 10:00:24 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The Accelerator Simulation Framework
====================================

The **Accelerator Simulation Framework** otherwise known as **SimFrame** is a ``python`` package for performing start-to-end (S2E) simulations of linear particle accelerators.

It provides a wrapper for several well-known particle tracking codes:

* `ASTRA <https://www.desy.de/~mpyflo/>`_
* `GPT <https://www.pulsar.nl/gpt/>`_
* `Elegant <https://www.aps.anl.gov/Accelerator-Operations-Physics/Software#elegant>`_
* `CSRTrack <https://www.desy.de/xfel-beam/csrtrack/>`_
* `Ocelot <https://github.com/ocelot-collab/ocelot>`_

The primary use for SimFrame has been for simulating the `CLARA <https://www.astec.stfc.ac.uk/Pages/CLARA.aspx>`_ particle accelerator.

Setup
-----
.. warning::
   | This site is currently **under construction**.
   | Some pages may have missing or incomplete reference documentation.

.. toctree::
   :maxdepth: 2
   
   installation
   getting-started
   MasterLattice
   SimCodes
   

.. Examples
   --------

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/notebooks/getting_started
   examples/notebooks/beams_example
   examples/notebooks/utility_functions
   
Participation
-------------

We welcome contributions and suggestions from the community! :mod:`SimFrame` is currently under active development,
and as such certain features may be missing or not working as expected. If you find any issues, please
raise it `here <https://github.com/astec-stfc/simframe/issues>`_ or contact
`Alex Brynes <mailto:alexander.brynes@stfc.ac.uk>`_ or `James Jones <mailto:james.jones@stfc.ac.uk>`_.

We are also happy to help with installation and setting up your accelerator lattice. 
   
.. API
   ---

.. toctree::
   :maxdepth: 2
   :caption: API
   
   Framework_objects
   Framework_elements
   SimulationFramework.Codes
   SimulationFramework.Modules
   



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
