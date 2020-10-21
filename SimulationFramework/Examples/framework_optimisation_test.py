import sys, os, time
sys.path.append('../../')
import SimulationFramework.Framework as fw
import SimulationFramework.Modules.read_twiss_file as rtf
import SimulationFramework.Modules.read_beam_file as rbf
from SimulationFramework.Modules.merge_two_dicts import merge_two_dicts
from SimulationFramework.Modules.constraints import *
import numpy as np
from scipy.optimize import minimize

beam = rbf.beam()
twiss = rtf.twiss()

#####################  Set-up base files for the injector  #####################
# Define a new framework instance, in directory 'example_optimisation'.
#       "clean" will empty (delete everything!) the directory if true
#       "verbose" will print a progressbar if true
lattice = fw.Framework('example_optimisation', clean=False, verbose=False)
# Load a lattice definition file. These can be found in Masterlattice/Lattices by default.
lattice.loadSettings('Lattices/CLA10-BA1_OM.def')
# This is a scaling parameter
scaling = 3
# This defines the number of particles to create at the gun (this is "ASTRA generator" which creates distributions)
lattice.generator.number_of_particles = 2**(3*scaling)
# Track the whole lattice - if this has already been run once, you do not need to run again!
# lattice.track()

def optFuncVELA(names, values):
    """ Evaluate the fitness of a set of quadrupole values """
    global bestdelta
    try:
        # Assign k1l values according to the input values - names is the list of quad names
        [lattice.modifyElement(name, 'k1l', val) for name, val in list(zip(names, values))]
        # Track the lattice, starting from S02 - i.e. do not re-run the injector!
        lattice.track(startfile='CLA-S02')
        # There is a small bug to do with file access in elegant - you may need a sleep command!
        time.sleep(0.01)
        # re-initialise the data objects
        constraintsList = {}
        twiss.reset_dicts()
        # Read in the twiss files for the different lattices
        twiss.read_elegant_twiss_files([lattice.subdirectory+'/'+l for l in  ['CLA-S02.twi','CLA-C2V.twi','EBT-INJ.twi','EBT-BA1.twi']])
        # These are position indices within the Twiss data object
        c2v1index = list(twiss['element_name']).index('CLA-C2V-MARK-01')
        c2v2index = list(twiss['element_name']).index('CLA-C2V-MARK-02')
        ipindex = list(twiss['element_name']).index('EBT-BA1-COFFIN-FOC')
        # Read the HDF5 beam file for the coffin focus
        beam.read_HDF5_beam_file(lattice.subdirectory+'/'+'EBT-BA1-COFFIN-FOC.hdf5')
        # Slice the beam into 25 slices
        beam.slices = 25
        beam.bin_time()
        # Get the peak current of the beam (at the coffin focus)
        current = beam.slice_peak_current
        peakI = max(current)
        # Here we define our constraints - each constraint must have a unique name!
        constraintsListBA1 = {
            'C2V_max_betax2': {'type': 'lessthan', 'value': twiss['beta_x'][:c2v2index], 'limit': 100, 'weight': 150},
            'C2V_max_betay2': {'type': 'lessthan', 'value': twiss['beta_y'][:c2v2index], 'limit': 100, 'weight': 150},

            'c2v_betax': {'type': 'equalto', 'value': twiss['beta_x'][c2v2index], 'limit': twiss['beta_x'][c2v1index], 'weight': 100},
            'c2v_betay': {'type': 'equalto', 'value': twiss['beta_y'][c2v2index], 'limit': twiss['beta_y'][c2v1index], 'weight': 100},
            'c2v_alphax': {'type': 'equalto', 'value': twiss['alpha_x'][c2v2index], 'limit': -1*twiss['alpha_x'][c2v1index], 'weight': 100},
            'c2v_alphay': {'type': 'equalto', 'value': twiss['alpha_y'][c2v2index], 'limit': -1*twiss['alpha_y'][c2v1index], 'weight': 100},

            'c2v_etax': {'type': 'lessthan', 'value': abs(twiss['eta_x_beam'][c2v2index]), 'limit': 3e-4, 'weight': 50},
            'c2v_etaxp': {'type': 'lessthan', 'value': abs(twiss['eta_xp_beam'][c2v2index]), 'limit': 3e-4, 'weight': 50},

            # 'ip_peakI': {'type': 'greaterthan', 'value': peakI, 'limit': 250, 'weight': 300},

            'ip_max_betax2': {'type': 'lessthan', 'value': twiss['beta_x'][c2v2index:], 'limit': 100, 'weight': 150},
            'ip_max_betay2': {'type': 'lessthan', 'value': twiss['beta_y'][c2v2index:], 'limit': 100, 'weight': 150},
            'ip_Sx': {'type': 'lessthan', 'value': 1e6*twiss['sigma_x'][ipindex], 'limit': 150, 'weight': 25},
            'ip_Sy': {'type': 'lessthan', 'value': 1e6*twiss['sigma_y'][ipindex], 'limit': 150, 'weight': 25},
            'ip_alphax': {'type': 'equalto', 'value': twiss['alpha_x_beam'][ipindex], 'limit': 0., 'weight': 2.5},
            'ip_alphay': {'type': 'equalto', 'value': twiss['alpha_y_beam'][ipindex], 'limit': 0., 'weight': 2.5},
            'ip_etax': {'type': 'lessthan', 'value': abs(twiss['eta_x_beam'][ipindex]), 'limit': 3e-4, 'weight': 50},
            'ip_etaxp': {'type': 'lessthan', 'value': abs(twiss['eta_xp_beam'][ipindex]), 'limit': 3e-4, 'weight': 50},
            'dump_etax': {'type': 'equalto', 'value': twiss['eta_x_beam'][-1], 'limit': 0.67, 'weight': 50},
            'dump_betax': {'type': 'lessthan', 'value': twiss['beta_x_beam'][-1], 'limit': 10, 'weight': 1.5},
            'dump_betay': {'type': 'lessthan', 'value': twiss['beta_y_beam'][-1], 'limit': 80, 'weight': 1.5},
        }
        constraintsList = constraintsListBA1
        # Instantiate a constraints object
        cons = constraintsClass()
        # Calculate the fitness
        delta = cons.constraints(constraintsList)
        # print some data - updateOutput() makes prettier output
        updateOutput('VELA delta = ' + str(delta))
        # if we have a new best solution
        if delta < bestdelta:
            bestdelta = delta
            # print some data, just in case
            print ('[',', '.join(map(str,values)),']')
            print(cons.constraintsList(constraintsList))
            # Save a new, best, changes file
            lattice.save_changes_file(filename=YAMLFILE)
            print('### New Best: ', delta)
        return delta
    except:
        # If something goes wrong, return a large fitness value
        return 1e18

def setVELA(quads):
    """ Optimise VELA lattice """
    global bestdelta
    bestdelta = 1e10
    names, best = list(zip(*quads))
    res = minimize(lambda x: optFuncVELA(names, x), best, method='nelder-mead', options={'disp': False, 'adaptive': True, 'fatol': 1e-3, 'maxiter': 250})
    return res

def optimise_Lattice(q=100, do_optimisation=False, quads=None):
    """ Perform lattice optimisation """
    if do_optimisation:
        output = setVELA(quads)
    return output

def get_quads(names):
    """ return quadrupole values """
    return [[name, lattice.getElement(name,'k1l')] for name in names]

def updateOutput(output):
    """ Write to stdout and flush """
    sys.stdout.write(output + '\r')
    sys.stdout.flush()
    # print(*output)

if __name__ == '__main__':
    global YAMLFILE
    # Define and load the SimFrame changes file we use as a starting point
    YAMLFILE = 'example_optimisation/changes_optimise_10pC_Hector.yaml'
    lattice.load_changes_file(filename=YAMLFILE)
    # Quadrupole names that will be used in the optimisation
    quad_names = [
        'CLA-S02-MAG-QUAD-01',
        'CLA-S02-MAG-QUAD-02',
        'CLA-S02-MAG-QUAD-03',
        'CLA-S02-MAG-QUAD-04',
        'CLA-C2V-MAG-QUAD-01',
        'CLA-C2V-MAG-QUAD-02',
        'CLA-C2V-MAG-QUAD-03',
        'EBT-INJ-MAG-QUAD-07',
        'EBT-INJ-MAG-QUAD-08',
        'EBT-INJ-MAG-QUAD-09',
        'EBT-INJ-MAG-QUAD-10',
        'EBT-INJ-MAG-QUAD-11',
        'EBT-INJ-MAG-QUAD-15',
        'EBT-BA1-MAG-QUAD-01',
        'EBT-BA1-MAG-QUAD-02',
        'EBT-BA1-MAG-QUAD-03',
        'EBT-BA1-MAG-QUAD-04',
        'EBT-BA1-MAG-QUAD-05',
        'EBT-BA1-MAG-QUAD-06',
        'EBT-BA1-MAG-QUAD-07',
    ]
    # Get quadrupole starting values
    best = get_quads(quad_names)
    # Optimise!
    output = optimise_Lattice(do_optimisation=True, quads=best)
    # This is the final fitness value
    fitness = output.fun
    # This is the final solution in [[quad, value]...] format
    # (which can be used as input to optimise_Lattice)
    best = list(zip(quad_names, output.x))
