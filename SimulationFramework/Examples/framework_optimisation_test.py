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

# Define a new framework instance, in directory 'example_ASTRA'.
#       "clean" will empty (delete everything!) the directory if true
#       "verbose" will print a progressbar if true
lattice = fw.Framework('example_optimisation', clean=False, verbose=False)
# Load a lattice definition file. These can be found in Masterlattice/Lattices by default.
lattice.loadSettings('Lattices/CLA10-BA1_OM.def')
# This is a scaling parameter
scaling = 3
# This defines the number of particles to create at the gun (this is "ASTRA generator" which creates distributions)
lattice.generator.number_of_particles = 2**(3*scaling)
# Track the whole lattice
# lattice.track()

def optFuncChicane(args):
    lattice.modifyElement('CLA-S02-MAG-QUAD-01','k1l', args[0])
    lattice.modifyElement('CLA-S02-MAG-QUAD-02','k1l', args[1])
    lattice.modifyElement('CLA-S02-MAG-QUAD-03','k1l', args[2])
    lattice.modifyElement('CLA-S02-MAG-QUAD-04','k1l', args[3])
    lattice.modifyElement('CLA-C2V-MAG-QUAD-01','k1l', args[4])
    lattice.modifyElement('CLA-C2V-MAG-QUAD-02','k1l', args[5])
    lattice.modifyElement('CLA-C2V-MAG-QUAD-03','k1l', args[6])
    lattice.track(startfile='CLA-S02')
    beam.read_HDF5_beam_file(lattice.subdirectory+'/'+'CLA-C2V-MARK-01.hdf5')
    emitStart = beam.normalized_horizontal_emittance
    betaXStart = beam.beta_x
    betaYStart = beam.beta_y
    beam.read_HDF5_beam_file(lattice.subdirectory+'/'+'CLA-C2V-MARK-02.hdf5')
    emitEnd = beam.normalized_horizontal_emittance
    betaXEnd = beam.beta_x
    betaYEnd = beam.beta_y
    betaYPenalty = betaYStart - 50 if betaYStart > 50 else 0
    etaXEnd = beam.eta_x
    etaXEnd = etaXEnd if abs(etaXEnd) > 1e-3 else 0
    etaXPEnd = beam.eta_xp
    etaXPEnd = etaXEnd if abs(etaXEnd) > 1e-3 else 0
    delta = np.sqrt((1e6*abs(emitStart-emitEnd))**2 + 1*abs(betaXStart-betaXEnd)**2 + 1*abs(betaYStart-betaYEnd)**2 + 10*abs(betaYPenalty)**2 + 100*abs(etaXEnd)**2 + 100*abs(etaXPEnd)**2)
    if delta < 0.4:
        delta = 0.4
    updateOutput('Chicane delta = ' + str(delta), args)
    evaluation_solutions[str(args)] = delta
    return delta

def setChicane(quads=None):
    evaluation_solutions = {}
    best = [lattice.getElement('CLA-S02-MAG-QUAD-01','k1l'),
            lattice.getElement('CLA-S02-MAG-QUAD-02','k1l'),
            lattice.getElement('CLA-S02-MAG-QUAD-03','k1l'),
            lattice.getElement('CLA-S02-MAG-QUAD-04','k1l'),
            lattice.getElement('CLA-C2V-MAG-QUAD-01','k1l'),
            lattice.getElement('CLA-C2V-MAG-QUAD-02','k1l'),
            lattice.getElement('CLA-C2V-MAG-QUAD-03','k1l'),
    ]
    if quads is not None:
        best = quads
    res = minimize(optFuncChicane, best, method='nelder-mead', options={'disp': False, 'adaptive': True, 'maxiter': 10, 'xatol': 1e-3})
    return res.x

def optFuncVELA(names, values):
    global bestdelta
    try:
        [lattice.modifyElement(name, 'k1l', val) for name, val in list(zip(names, values))]
        run_id = lattice.track(startfile='CLA-S02')
        time.sleep(0.1)
        constraintsList = {}
        twiss.reset_dicts()
        twiss.read_elegant_twiss_files([lattice.subdirectory+'/'+l for l in  ['CLA-S02.twi','CLA-C2V.twi','EBT-INJ.twi','EBT-BA1.twi']])
        c2v1index = list(twiss['element_name']).index('CLA-C2V-MARK-01')
        c2v2index = list(twiss['element_name']).index('CLA-C2V-MARK-02')
        ipindex = list(twiss['element_name']).index('EBT-BA1-COFFIN-FOC')
        beam.read_HDF5_beam_file(lattice.subdirectory+'/'+'EBT-BA1-COFFIN-FOC.hdf5')
        beam.slices = 25
        beam.bin_time()
        current = beam.slice_peak_current
        peakI = max(current)
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
        cons = constraintsClass()
        delta = cons.constraints(constraintsList)
        updateOutput('VELA delta = ' + str(delta))
        if delta < bestdelta:
            bestdelta = delta
            # print(*args, sep = ", ")
            print ('[',', '.join(map(str,values)),']')
            print(cons.constraintsList(constraintsList))
            lattice.save_changes_file(filename=YAMLFILE)
            print('### New Best: ', delta)
        return delta
    except:
        return 1e18

def setVELA(quads):
    global bestdelta
    bestdelta = 1e10
    names, best = list(zip(*quads))
    res = minimize(lambda x: optFuncVELA(names, x), best, method='nelder-mead', options={'disp': False, 'adaptive': True, 'fatol': 1e-3, 'maxiter': 250})
    return res

def optimise_Lattice(q=100, do_optimisation=False, quads=None):
    # if quadnamevalues is None:
    #     quads = np.array([
    #         1.775, -1.648, 2.219, -1.387, 5.797,-4.95, 5.714, 1.725, -1.587, 0.376, -0.39, 0.171, 0.123, -0.264, -0.959, 1.225, 1.15, 0.039, -1.334, 1.361
    #     ])
    # OM.modify_widget('generator:charge:value', q)
    if do_optimisation:
        output = setVELA(quads)
    # optFuncVELA(output.x)
    return output

def get_quads(names):
    return [[name, lattice.getElement(name,'k1l')] for name in names]

def updateOutput(output):
    sys.stdout.write(output + '\r')
    sys.stdout.flush()
    # print(*output)

if __name__ == '__main__':
    global YAMLFILE
    YAMLFILE = 'example_optimisation/changes_optimise_10pC_Hector.yaml'
    lattice.load_changes_file(filename=YAMLFILE)
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
    best = get_quads(quad_names)
    fitness = 1000
    output = optimise_Lattice(do_optimisation=True, quads=best)
    fitness = output.fun
    best = list(zip(quad_names, output.x))
