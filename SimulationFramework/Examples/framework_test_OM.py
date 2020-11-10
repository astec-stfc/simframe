import sys, os
sys.path.append('../../')
import SimulationFramework.Framework as fw
import SimulationFramework.Modules.read_twiss_file as rtf
import SimulationFramework.Modules.read_beam_file as rbf
import numpy as np
from scipy.optimize import minimize
from functools import partial
beamobject = rbf.beam()

# Define a new framework instance, in directory 'example_ASTRA'.
#       "clean" will empty (delete everything!) the directory if true
#       "verbose" will print a progressbar if true
framework = fw.Framework('example_ASTRA_OM', clean=False, verbose=True)
# Load a lattice definition file. These can be found in Masterlattice/Lattices by default.
framework.loadSettings('Lattices/CLA10-BA1_OM.def')
# Change all lattice codes to ASTRA/Elegant/GPT with exclusions (injector can not be done in Elegant)
framework.change_Lattice_Code('All','ASTRA')
# This is a scaling parameter
scaling = 3
framework.change_generator('GPT')
# This defines the number of particles to create at the gun (this is "ASTRA generator" which creates distributions)
framework.generator.number_of_particles = 2**(3*scaling)
framework.generator.charge = 100e-12
# Track the whole lattice
# framework.track(endfile='CLA-S02')

# Unless you have GPT installed, don't run this.
framework.setSubDirectory('example_GPT_OM')
gun_crest = 140.47
linac1_crest = 152.7
framework.change_Lattice_Code('All','GPT', exclude=[])
framework['Gun'].prefix = '../example_ASTRA_OM/'

def crest_gun(framework, guess=None):
    framework['Gun'].file_block['charge']['space_charge_mode'] = None
    framework['CLA-LRG1-GUN-CAV'].phase = 0
    if guess is None:
        x0 = np.array([framework['CLA-LRG1-GUN-CAV'].crest])
    else:
        x0 = np.array([guess])
    optFunc = partial(cavity_momentum, framework=framework, cavity='CLA-LRG1-GUN-CAV', lattice_section='Gun', beam_file='CLA-L01-APER.hdf5')
    res = minimize(optFunc, x0, method='powell')
    return res.x

def crest_linac1(framework, guess=None):
    framework['Linac'].file_block['charge']['space_charge_mode'] = None
    framework['CLA-L01-CAV'].phase = 0
    if guess is None:
        x0 = np.array([framework['CLA-L01-CAV'].crest])
    else:
        x0 = np.array([guess])
    optFunc = partial(cavity_momentum, framework=framework, cavity='CLA-L01-CAV', lattice_section='Linac', beam_file='CLA-S02-APER-01.hdf5')
    res = minimize(optFunc, x0, method='powell')
    return res.x

def cavity_momentum(phase, framework=None, cavity=None, lattice_section=None, beam_file=None):
    framework[cavity].crest = phase[0]
    framework.track(files=[lattice_section])
    beamobject.read_HDF5_beam_file(os.path.join(framework.subdirectory, beam_file))
    print('Phase = ', phase, 1e-6*float(np.mean(beamobject.cpz)))
    return -1*float(np.mean(beamobject.cpz))

verbose = framework.verbose
framework.verbose = False
gunSC = framework['Gun'].file_block['charge']['space_charge_mode']
gunPhase = framework['CLA-LRG1-GUN-CAV'].phase
# gun_crest = crest_gun(framework, guess=gun_crest) - 0.5 ## This is a fudge!
framework['Gun'].file_block['charge']['space_charge_mode'] = gunSC
framework['CLA-LRG1-GUN-CAV'].crest = gun_crest
framework['CLA-LRG1-GUN-CAV'].phase = gunPhase
print('Gun crest =', gun_crest)
# framework.track(files=['Gun'])

linacSC = framework['Linac'].file_block['charge']['space_charge_mode']
linacPhase = framework['CLA-L01-CAV'].phase
# linac1_crest = crest_linac1(framework, linac1_crest)
framework['Linac'].file_block['charge']['space_charge_mode'] = linacSC
framework['CLA-L01-CAV'].crest = linac1_crest
framework['CLA-L01-CAV'].phase = linacPhase
print('Linac1 crest =', linac1_crest)
# framework.track(files=['Linac'])
framework.verbose = verbose

framework['Gun'].screen_step_size = 0.1
framework['Linac'].screen_step_size = 0.1
framework['CLA-LRG1-GUN-CAV'].crest = gun_crest
framework['CLA-L01-CAV'].crest = linac1_crest
framework.track(startfile='Gun', endfile='CLA-S02')
