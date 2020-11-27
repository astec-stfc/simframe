import sys, os
sys.path.append('../../')
import SimulationFramework.Framework as fw
import SimulationFramework.Modules.Beams as rtf
import SimulationFramework.Modules.Twiss as rbf
import numpy as np

# Define a new framework instance, in directory 'example_ASTRA'.
#       "clean" will empty (delete everything!) the directory if true
#       "verbose" will print a progressbar if true
framework = fw.Framework('example_ASTRA', clean=False, verbose=True)
# Load a lattice definition file. These can be found in Masterlattice/Lattices by default.
framework.loadSettings('Lattices/clara400_v12_v3.def')
# Change all lattice codes to ASTRA/Elegant/GPT with exclusions (injector can not be done in Elegant)
framework.change_Lattice_Code('All','ASTRA', exclude=['injector400','VBC'])
# Again, but put the VBC in Elegant for CSR
framework.change_Lattice_Code('VBC','ASTRA')
# This is the code that generates the laser distribution (ASTRA or GPT)
# framework.change_generator('GPT')
# Load a starting laser distribution setting
framework.generator.load_defaults('clara_400_2ps_Gaussian')
# Set the thermal emittance for the generator
# framework.generator.thermal_emittance = 0.0005
# This is a scaling parameter
scaling = 3
# This defines the number of particles to create at the gun (this is "ASTRA generator" which creates distributions)
framework.generator.number_of_particles = 2**(3*scaling)
# Track the whole lattice
framework.track()
exit()
# for lattice in framework.latticesObjects.values():
#     lsc = False
#     elements = lattice.elements.values()
#     for e in elements:
#         e.lsc_enable = False
#         e.csr_enable = False
#     lattice.lscDrifts = False
#     lattice.csrDrifts = False

# lattice = fw.Framework('example_ASTRA_OM', clean=False, verbose=True)
# framework.loadSettings('Lattices/CLA10-BA1_OM.def')
# framework.change_Lattice_Code('CLA-C2V','ASTRA')
# framework.generator.load_defaults('clara_400_2ps_Gaussian')
# scaling = 3
# framework.generator.number_of_particles = 2**(3*scaling)
# framework.track(files=['CLA-C2V'])

# This time we will use CSRTrack for the VBC
framework = fw.Framework('example_ASTRA_CSRTrack', clean=False, verbose=True)
framework.loadSettings('Lattices/clara400_v12_v3.def')
framework.generator.number_of_particles = 2**(3*scaling)
framework.change_Lattice_Code('All','ASTRA', exclude=['injector400','VBC'])
framework.change_Lattice_Code('VBC','csrtrack')
# We want to start from the VBC so we don't run the injector again.
# Here we tell SimFrame where to look for the starting files (the "prefix" parameter) for the lattice we want to run
framework.set_lattice_prefix('S02', '../example_ASTRA/')
# Here we ony run from the VBC
# framework.track(startfile='VBC',endfile='S07')

# This time we will use Elegant for everything except the injector
framework = fw.Framework('example_Elegant', clean=False, verbose=True)
framework.loadSettings('Lattices/clara400_v12_v3.def')
framework.generator.number_of_particles = 2**(3*scaling)
framework.change_Lattice_Code('All','elegant', exclude=['injector400'])
# Set the prefix for S02 (where we will start)
framework.set_lattice_prefix('S02', '../example_ASTRA/')
# Run from S02 onwards
# framework.track(startfile='S02', endfile='S07')

# Unless you have GPT installed, don't run this.
framework = fw.Framework('example_GPT', clean=False, verbose=True)
framework.loadSettings('Lattices/CLA10-BA1_OM.def')
framework.generator.number_of_particles = 2**(3*scaling)
framework.change_Lattice_Code('All','GPT', exclude=[])
framework.track()
