# SimFrame
*Improving accelerators through consistent, complete, and easy to use simulations.*

## Mission
`SimFrame` is a framework for performing simulations of particle accelerators (particularly linac-based FELs) that aims to be simple to use, and complete.

Our mission statement is *"To create a start to end framework for consistent, transparent simulations of particle accelerators and FELs that anybody can use, and that everybody trusts."*

By leveraging a [standard accelerator lattice format](https://github.com/astec-stfc/masterlattice.git), `SimFrame` is able to generate and run input files for a range of accelerator simulation codes, enabling seamless transfer of input and output distributions. 

The codes currently supported by `SimFrame` are:

* [ASTRA](https://www.desy.de/~mpyflo/)
* [GPT](https://pulsar.nl/)
* [ELEGANT](https://www.aps.anl.gov/Accelerator-Operations-Physics/Software)
* [Ocelot](https://github.com/ocelot-collab/ocelot)
* [CSRTrack](https://www.desy.de/xfel-beam/csrtrack/)

A range of other codes are also currently under active development. 

**`Documentation`** |
------------------- |
[![Documentation](https://img.shields.io/badge/simframe-documentation-blue.svg)](https://acceleratorsimframe.readthedocs.io/)  |

## Installation

Cloning from Github
-------------------

Clone `SimFrame` from Github:

```bash
    git clone https://github.com/astec-stfc/simframe.git
```

The package and its dependencies can be installed using the following command in the `simframe` directory:

```bash
    pip install .
```
    
Optional Dependencies
---------------------

The following dependencies are optional, but are generally required for running ``CLARA`` simulations and for testing the code:

* [MasterLattice](https://github.com/astec-stfc/masterlattice.git)
* [SimCodes](https://github.com/astec-stfc/simcodes.git)

## Example Lattice and Simulation

Getting started with SimFrame
=============================

The first step in starting a new `SimFrame` simulation is to define the lattice, using the `MasterLattice`.

The following is part of the ``CLARA`` Injector, in a file  called ``YAML/CLA_Gun400.yaml`` for example:

```yaml
elements:
    CLA-HRG1-SIM-APER-01:
        centre: [0, 0, 0]
        datum: [0, 0, 0]
        global_rotation: [0, 0, 0]
        horizontal_size: 0.017
        shape: rectangular
        type: aperture
        vertical_size: 0.017
    CLA-HRG1-GUN-CAV-01:
        Structure_Type: StandingWave
        centre: [0.0, 0.0, 0.16]
        crest: 145.789
        datum: [0.0, 0.0, 0.32]
        field_amplitude: 120000000.0
        field_definition: $master_lattice_location$Data_Files/HRRG_1D_RF.dat
        field_definition_gdf: $master_lattice_location$Data_Files/HRRG_1D_RF.gdf
        frequency: 2998500000.0
        global_rotation: [0, 0, 0]
        length: 0.32
        lsc_cutoff_high: [0, 0]
        n_cells: 1.5
        phase: 9
        type: cavity
            sub_elements:
                EBT-HRG1-MAG-SOL-01:
                    centre: [0.0, 0.0, 0.16241]
                    datum: [0.0, 0.0, 0.32]
                    field_amplitude: 0.345
                    field_definition: $master_lattice_location$Data_Files/HRRG_combined_sols_100mm_onaxis.dat
                    field_definition_gdf: $master_lattice_location$Data_Files/HRRG_combined_sols_100mm_onaxis.gdf
                    global_rotation: [0, 0, 0]
                    length: 0.32
                    type: solenoid
    CLA-S01-SIM-APER-01:
        centre: [0.0, 0.0, 0.32]
        datum: [0.0, 0.0, 0.32]
        global_rotation: [0, 0, 0]
        horizontal_size: 0.017
        shape: rectangular
        type: aperture
```

Note the use of ``sub_elements`` to define elements that overlap an existing element (in this case, a solenoid placed around the gun). 
We make extensive use of `substitutions` to define the locations of field definition files.

Element specific options (such as RF parameters) are also specified.

Defining the Lattice Simulation
-------------------------------

The simulation of the lattice is defined in a separate ``YAML`` file, for example ``CLA-Injector.def``:

```yaml
generator:
    default: clara_400_3ps
files:
    injector400:
        code: ASTRA
        charge:
            cathode: True
            space_charge_mode: 2D
            mirror_charge: True
        input:
            particle_definition: 'initial_distribution'
        output:
            zstart: 0
            end_element: CLA-L01-SIM-APER-01
    Linac:
        code: elegant
        starting_rotation: 0
        charge:
            cathode: False
            space_charge_mode: 3D
        input:
            {}
        output:
            start_element: CLA-L01-SIM-APER-01
            end_element: CLA-S02-SIM-APER-01
elements:
    filename: YAML/CLA_Gun400.yaml
```

This lattice definition would produce several output files (called ``injector400.in`` and ``Linac.lte``) for running in the **ASTRA** and **Elegant** beam tracking codes.
The elements are loaded from the file ``YAML/CLA_Gun400.yaml`` defined above. Element definitions can also be defined directly in the ``.def`` file.

As this simulation starts from the cathode, the ``input`` definiton is required for the first `injector400` ``file`` block. 

For `follow-on` lattice runs, it is sufficient to define the ``output: start_element``, which should match the ``output: end_element`` definition 
from the previous ``file`` block.


Running SimFrame
----------------

```python
import SimulationFramework.Framework as fw


# Define a new framework instance, in directory 'example'.
#       "clean" will empty (delete everything!) in the directory if true
#       "verbose" will print a progressbar if true
framework = fw.Framework("example", clean=True, verbose=True)
# Load a lattice definition file. These can be found in Masterlattice/Lattices by default.
framework.loadSettings("Lattices/clara400_v13.def")
# Change all lattice codes to ASTRA/Elegant/GPT with exclusions (injector can not be done in Elegant)
framework.change_Lattice_Code("All", "ASTRA", exclude=["VBC"])
# Again, but put the VBC in Elegant for CSR
framework.change_Lattice_Code("VBC", "Elegant")
# This is the code that generates the laser distribution (ASTRA or GPT)
framework.change_generator("ASTRA")
# Load a starting laser distribution setting
framework.generator.load_defaults("clara_400_2ps_Gaussian")
# Set the thermal emittance for the generator
framework.generator.thermal_emittance = 0.0005
# This is a scaling parameter
# This defines the number of particles to create at the gun (this is "ASTRA generator" which creates distributions)
framework.generator.number_of_particles = 2 ** (3 * scaling)
```

Example Notebooks
-----------------

Some further examples on `SimFrame` usage can be found in the following notebooks:

* [getting_started.ipynb](./examples/notebooks/getting_started.ipynb)
* [beams_example.ipynb](./examples/notebooks/beams_example.ipynb)
* [utility_functions.ipynb](./examples/notebooks/utility_functions.ipynb)
