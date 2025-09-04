.. _installation:

Installation
============

.. note::
   | **SimFrame** is compatible only with python `3.12` and above.
   | Contact `Alex Brynes <mailto:alexander.brynes@stfc.ac.uk>`_ in case of any issues during installation / testing / etc.

Cloning from Github
-------------------

Clone ``SimFrame`` from Github:

.. code-block:: bash

    git clone https://github.com/astec-stfc/simframe.git

Install via pip
-------------------

(It is recommended to activate a ``python3.12`` virtual environment to run ``SimFrame``.)

The package and its dependencies can be installed using the following command in the ``SimFrame`` directory:

```bash
pip install .
```

To install the ``MasterLattice`` and ``SimCodes`` packages along with ``SimFrame``, use this command from
the ``SimFrame`` directory:

```bash
pip install .[test]
```

To check that the install was completed successfully, run this command from the top level:

```bash
pytest --cov
```


Install from pypi / conda-force
-------------------

WIP.....

Required Dependencies
---------------------

Check out the ``pyproject.toml`` file for a full list of dependencies for ``SimFrame``.

Optional Dependencies
---------------------

The following dependencies are optional, but are generally required for running ``CLARA`` simulations:

* `MasterLattice <https://github.com/astec-stfc/masterlattice.git>`__
* `SimCodes <https://github.com/astec-stfc/simcodes.git>`__

These can be installed via `pip`:

.. code-block:: bash

    pip install MasterLattice SimCodes
