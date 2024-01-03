import os, shutil
import numpy as np
from SimulationFramework.Modules.constraints import constraintsClass
from SimulationFramework.Modules.nelder_mead import nelder_mead
import csv
from copy import copy
from SimulationFramework.Modules.merge_two_dicts import merge_two_dicts
from . import runElegant as runEle
from scipy.optimize import minimize

def saveState(dir, args, n, fitness):
    with open(dir+'/best_solutions_running.csv','a') as out:
        csv_out=csv.writer(out)
        args=list(args)
        args.append(n)
        args.append(fitness)
        csv_out.writerow(args)

def saveParameterFile(best, file='clara_elegant_best.yaml'):
    if POST_INJECTOR:
        allparams = list(zip(*(parameternames)))
    else:
        allparams = list(zip(*(injparameternames+parameternames)))
    output = {}
    for p, k, v in zip(allparams[0], allparams[1], best):
        if p not in output:
            output[p] = {}
        output[p][k] = v
        with open(file,"w") as yaml_file:
            yaml.dump(output, yaml_file, default_flow_style=False)

class Optimise_Elegant(runEle.fitnessFunc):

    injector_parameter_names = [
        ['CLA-HRG1-GUN-CAV', 'phase'],
        ['CLA-HRG1-GUN-SOL', 'field_amplitude'],
        ['CLA-L01-CAV', 'field_amplitude'],
        ['CLA-L01-CAV', 'phase'],
        ['CLA-L01-CAV-SOL-01', 'field_amplitude'],
        ['CLA-L01-CAV-SOL-02', 'field_amplitude'],
    ]
    parameter_names = [
        ['CLA-L02-CAV', 'field_amplitude'],
        ['CLA-L02-CAV', 'phase'],
        ['CLA-L03-CAV', 'field_amplitude'],
        ['CLA-L03-CAV', 'phase'],
        ['CLA-L4H-CAV', 'field_amplitude'],
        ['CLA-L4H-CAV', 'phase'],
        ['CLA-L04-CAV', 'field_amplitude'],
        ['CLA-L04-CAV', 'phase'],
        ['bunch_compressor', 'angle'],
        ['CLA-S07-DCP-01', 'factor'],
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cons = constraintsClass()
        self.changes = None
        self.lattice = None
        self.resultsDict = {}
        self.deleteFolders = True
        # ******************************************************************************
        self.CLARA_dir = os.path.relpath(__file__+'/../../CLARA')
        self.POST_INJECTOR = True
        CREATE_BASE_FILES = False
        self.scaling = 5
        self.verbose = False
        self.basefiles = '../../CLARA/basefiles_'+str(self.scaling)+'/'
        # ******************************************************************************
        if not self.POST_INJECTOR:
            best = injector_startingvalues + best
        elif CREATE_BASE_FILES:
            for i in [self.scaling]:
                pass
                # optfunc(injector_startingvalues + best, scaling=scaling, post_injector=False, verbose=False, runGenesis=False, dir='nelder_mead/basefiles_'+str(i))

    def calculate_constraints(self):
        pass

    def set_changes_file(self, changes):
        self.changes = changes

    def set_lattice_file(self, lattice):
        self.lattice_file = lattice

    def set_start_file(self, file):
        self.start_lattice = file

    def OptimisingFunction(self, inputargs, *args, endfile=None, **kwargs):
        if not self.POST_INJECTOR:
            parameternames = self.injector_parameter_names + self.parameter_names
        else:
            parameternames = copy(self.parameter_names)
        self.inputlist = [a[0]+[a[1]] for a in zip(parameternames, inputargs)]

        self.linac_names = np.array([i[0] for i in self.inputlist if i[1] == 'field_amplitude'])
        self.linac_fields = np.array([i[2] for i in self.inputlist if i[1] == 'field_amplitude'])
        self.linac_phases = np.array([i[2] for i in self.inputlist if i[1] == 'phase'])
        self.vbc_angle = np.array([i[2] for i in self.inputlist if i[1] == 'angle'])

        if 'iteration' in list(kwargs.keys()):
            self.opt_iteration = kwargs['iteration']
            del kwargs['iteration']

        if 'bestfit' in list(kwargs.keys()):
            self.bestfit = kwargs['bestfit']
            del kwargs['bestfit']

        if 'dir' in list(kwargs.keys()):
            dir = kwargs['dir']
            del kwargs['dir']
            save_state = False
        else:
            save_state = True
            dir = self.optdir+str(self.opt_iteration)

        # print('dir = ', dir)
        self.setup_lattice(self.inputlist, dir)
        print('New run = ', list(inputargs), dir)
        self.before_tracking()
        # if not 'track' in kwargs or ('track' in kwargs and not kwargs['track']):
        constraintsList = None
        try:
            fitvalue = self.track(endfile=endfile,  **kwargs)
            constraintsList = self.calculate_constraints()
            fitvalue = self.cons.constraints(constraintsList)
        except Exception as e:
            print(e)
            fitvalue = 1e26

        if isinstance(self.opt_iteration, int):
            self.opt_iteration += 1
            print('fitvalue[', self.opt_iteration-1, '] = ', fitvalue)
        elif constraintsList is None:
            pass
        else:
            print('fitvalue = ', fitvalue)

        if save_state:
            try:
                if isinstance(self.opt_iteration, int):
                    saveState(self.subdir, inputargs, self.opt_iteration-1, fitvalue)
                else:
                    saveState(self.subdir, inputargs, self.opt_iteration, fitvalue)
            except:
                pass
        if save_state and fitvalue < self.bestfit:
            print(self.cons.constraintsList(constraintsList))
            print('!!!!!!  New best = ', fitvalue, inputargs)
            self.bestfit = fitvalue
            try:
                shutil.copyfile(dir+'/changes.yaml', self.best_changes)
                self.framework.save_lattice()
            except:
                pass
        else:
            if self.deleteFolders:
                shutil.rmtree(dir, ignore_errors=True)
        return fitvalue

    def Nelder_Mead(self, best=None, step=0.1, best_changes='./nelder_mead_best_changes.yaml', subdir='nelder_mead', converged=None, **kwargs):
        best = np.array(self.best) if best is None else np.array(best)
        self.subdir = subdir
        self.optdir = self.subdir + '/iteration_'
        self.best_changes = best_changes
        print('best = ', best)
        self.bestfit = 1e26

        os.makedirs(subdir, exist_ok=True)
        with open(subdir+'/best_solutions_running.csv','w') as out:
            self.opt_iteration = 0
        res = nelder_mead(self.OptimisingFunction, best, step=step, max_iter=300, no_improv_break=100, converged=converged, **kwargs)
        print(res)

    def Simplex(self, best=None, best_changes='./simplex_best_changes.yaml', subdir='simplex', maxiter=300, **kwargs):
        best = self.best if best is None else best
        self.subdir = subdir
        self.optdir = self.subdir + '/iteration_'
        self.best_changes = best_changes
        print('best = ', best)
        self.bestfit = 1e26

        os.makedirs(subdir, exist_ok=True)
        with open(subdir+'/best_solutions_running.csv','w') as out:
            self.opt_iteration = 0
        res = minimize(self.OptimisingFunction, best, method='nelder-mead', options={'disp': True, 'maxiter': maxiter, 'adaptive': True}, args=kwargs)
        print(res.x)

    def Example(self, best=None, step=0.1, dir='example', **kwargs):
        best = np.array(self.best) if best is None else np.array(best)
        self.subdir = dir
        self.optdir = self.subdir
        self.best_changes = './example_best_changes.yaml'
        # print('best = ', best)
        self.bestfit = -1e26

        self.opt_iteration = ''
        # try:
        self.OptimisingFunction(best, **kwargs)
        # except:
            # pass
