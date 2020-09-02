import sys, os
sys.path.append('../../')
import SimulationFramework.Framework as fw
import SimulationFramework.Modules.read_twiss_file as rtf
import SimulationFramework.Modules.read_beam_file as rbf
import numpy as np

# Define a new framework instance, in directory 'C2V'.
#       "clean" will empty (delete everything!) the directory if true
#       "verbose" will print a progressbar is true
lattice = fw.Framework('example_ASTRA', clean=False, verbose=True)
# Load a lattice definition file. These can be found in Masterlattice/Lattices by default.
lattice.loadSettings('Lattices/clara400_v12_v3.def')
# Change all lattice codes to ASTRA/Elegant/GPT with exclusions (injector can not be done in Elegant)
lattice.change_Lattice_Code('All','ASTRA', exclude=['injector400','VBC'])
# Again, but put the VBC in Elegant for CSR
lattice.change_Lattice_Code('VBC','elegant')
# This is the code that generates the laser distribution (ASTRA or GPT)
lattice.change_generator('ASTRA')
# Load a starting laser distribution setting
lattice.generator.load_defaults('clara_400_2ps_Gaussian')
# Set the thermal emittance for the generator
lattice.generator.thermal_emittance = 0.0005
# This is a scaling parameter
scaling = 5
# This defines the number of particles to create at the gun (this is "ASTRA generator" which creates distributions)
lattice.generator.number_of_particles = 2**(3*scaling)
# Track the whole lattice
# lattice.track()

# This time we will use CSRTrack for the VBC
lattice = fw.Framework('example_ASTRA_CSRTrack', clean=False, verbose=True)
lattice.loadSettings('Lattices/clara400_v12_v3.def')
lattice.generator.number_of_particles = 2**(3*4)
lattice.change_Lattice_Code('All','ASTRA', exclude=['injector400','VBC'])
lattice.change_Lattice_Code('VBC','csrtrack')
# We want to start from the VBC so we don't run the injector again.
# Here we tell SimFrame where to look for the starting files (the "prefix" parameter) for the lattice we want to run
lattice.set_lattice_prefix('S02', '../example_ASTRA/')
# Here we ony run from the VBC
lattice.track(startfile='VBC',endfile='S07')

# This time we will use Elegant for everything except the injector
lattice = fw.Framework('example_Elegant', clean=False, verbose=True)
lattice.loadSettings('Lattices/clara400_v12_v3.def')
lattice.generator.number_of_particles = 2**(3*4)
lattice.change_Lattice_Code('All','elegant', exclude=['injector400'])
# Set the prefix for S02 (where we will start)
lattice.set_lattice_prefix('S02', '../example_ASTRA/')
# Run from S02 onwards
lattice.track(startfile='S02', endfile='S07')

# Unless you have GPT installed, don't run this.
lattice = fw.Framework('example_GPT', clean=False, verbose=True)
lattice.loadSettings('Lattices/clara400_v12_v3.def')
lattice.generator.number_of_particles = 2**(3*4)
lattice.change_Lattice_Code('All','elegant', exclude=['injector400'])
lattice.change_Lattice_Code('All','GPT', exclude=['injector400','L02','L03','L04','L4H','VBC','S07'])
lattice['S02'].prefix = '../example_ASTRA/'
# lattice.track(startfile='S07', endfile='S07')
