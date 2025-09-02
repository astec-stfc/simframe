Installation
============

.. note::
   | **SimFrame** is compatible only with python `3.10` and above.
   | Installing the package may require authentication with an STFC Federal ID or valid SSH key.

Cloning from Github
-------------------

Clone :mod:`SimFrame` from Github:

.. code-block:: bash

    git clone https://github.com/astec-stfc/simframe.git

Install via pip
-------------------

Install :mod:`SimFrame` from the ``apclara1.dl.ac.uk`` pypi server (requires STFC user ID):

.. code-block:: bash

    pip install --extra-index-url http://apclara1.dl.ac.uk:8090/simple/ --trusted-host apclara1.dl.ac.uk AcceleratorSimFrame


Required Dependencies
---------------------

:mod:`SimFrame` has the following required dependencies:

* `fastKDE>=2.1.5`
* `h5py>=3.4.0`
* `matplotlib>=3.4.3`
* `mpl_axes_aligner>=1.3`
* `munch>=2.5.0`
* `numpy>=2.2.6`
* `progressbar2==4.0.0`
* `pyyaml==6.0`
* `pyzmq==22.3.0`
* `scipy>=1.7.1`
* `deepdiff>=5`
* `tqdm>=4`
* `PyQt5>=5.15`
* `PyYAML>=5.3`
* `lox>=0.11`
* `pydantic>=2.5.3`
* `attrs>=23.2.0`
* `ocelot-desy>=25.06.0`
* `easygdf>=2.1.1`
* `soliday.sdds`
* `numexpr>=2.11.0`
* `numba>=0.61.2`
* `pyFFTW==0.15.0`
* `sphinx>8.1`
* `sphinx-rtd-theme`
* `myst-nb`
* `sphinx_autodoc_typehints>3`


You can also install via the ``requirements.txt`` file from the git repo:

.. code-block:: bash

    pip install -r requirements.txt

Optional Dependencies
---------------------

The following dependencies are optional, but are generally required for running ``CLARA`` simulations:

* `MasterLattice <https://github.com/astec-stfc/masterlattice.git>`__
* `SimCodes <https://github.com/astec-stfc/simcodes.git>`__

These can be installed via `pip`:

.. code-block:: bash

    pip install MasterLattice SimCodes
