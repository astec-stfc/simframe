Installation
============

.. note::
   | **SimFrame** is compatible only with python `3.10` and above.
   | Installing the package may require authentication with an STFC Federal ID or valid SSH key.

Cloning from GitLab
-------------------

Clone :mod:`SimFrame` from GitLab:

.. code-block:: bash

    git clone https://gitlab.stfc.ac.uk/jkj62/simframe.git

Install via pip
-------------------

Install :mod:`SimFrame` from the ``apclara1.dl.ac.uk`` pypi server:

.. code-block:: bash

    pip install --extra-index-url http://apclara1.dl.ac.uk:8090/simple/ --trusted-host apclara1.dl.ac.uk AcceleratorSimFrame


Required Dependencies
---------------------

:mod:`SimFrame` has the following required dependencies:

* `fastkde==1.0.19`
* `h5py==3.4.0`
* `matplotlib==3.4.3`
* `mpl_axes_aligner==1.3`
* `munch==2.5.0`
* `numpy==1.21.2`
* `progressbar2==4.0.0`
* `pyyaml==6.0`
* `pyzmq==22.3.0`
* `scipy==1.7.1`


You can also install via the ``requirements.txt`` file from the git repo:

.. code-block:: bash

    pip install -r requirements.txt

Optional Dependencies
---------------------

The following dependencies are optional, but are generally required for running ``CLARA`` simulations:

* `MasterLattice`
* `SimCodes`

The can all be installed from the ``apclara1.dl.ac.uk`` pypi server:

.. code-block:: bash

    pip install --extra-index-url http://apclara1.dl.ac.uk:8090/simple/ --trusted-host apclara1.dl.ac.uk MasterLattice SimCodes