SimCodes
=============

:mod:`SimCodes` is a container for the particle accelerator tracking codes used by :mod:`SimFrame`.

Install via pip
-------------------

Install :mod:`SimCodes` from ``pypi``:

.. code-block:: bash

    pip install SimCodes

This will allow ``python`` to access the binaries for ``ASTRA`` and ``CSRTrack``.
Windows binaries for ``ELEGANT`` are also added; we are working on including a stable version of
``ELEGANT`` for Linux machines. For now, it is recommended to install ``ELEGANT`` on Linux
following the instructions on the
`OAG software page <https://www.aps.anl.gov/Accelerator-Operations-Physics/Software>`_

Proprietary software such as ``GPT`` are not included; visit `Pular Physics <https://pulsar.nl>`_
for more information.

In case of any issues arising during installation or running :mod:`SimFrame`, contact
`Alex Brynes <mailto:alexander.brynes@stfc.ac.uk>`_.