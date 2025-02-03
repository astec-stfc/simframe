import os, errno, sys
import numpy as np
import random

sys.path.append("./../../")
from SimulationFramework.Modules.constraints import constraintsClass
from SimulationFramework.Modules.nelder_mead import nelder_mead
import time
import csv
from copy import copy

# sys.path.append(os.path.abspath(__file__+'/../../'))
import SimulationFramework.ClassFiles.genesisBeamFile as genesisBeamFile
from functools import partial
from collections import OrderedDict
from shutil import copyfile
from SimulationFramework.Modules.merge_two_dicts import merge_two_dicts


def saveState(args, n, fitness):
    with open("nelder_mead/best_solutions_running.csv", "a") as out:
        csv_out = csv.writer(out)
        args = list(args)
        args.append(n)
        args.append(fitness)
        csv_out.writerow(args)


def saveParameterFile(best, file="clara_elegantgenesis_best.yaml"):
    if POST_INJECTOR:
        allparams = zip(*(parameternames))
    else:
        allparams = zip(*(injparameternames + parameternames))
    output = {}
    for p, k, v in zip(allparams[0], allparams[1], best):
        if p not in output:
            output[p] = {}
        output[p][k] = v
        with open(file, "w") as yaml_file:
            yaml.dump(output, yaml_file, default_flow_style=False)


class Optimise_Genesis_Elegant(genesisBeamFile.genesisSimulation):

    injector_parameter_names = [
        ["CLA-HRG1-GUN-CAV", "phase"],
        ["CLA-HRG1-GUN-SOL", "field_amplitude"],
        ["CLA-L01-CAV", "field_amplitude"],
        ["CLA-L01-CAV", "phase"],
        ["CLA-L01-CAV-SOL-01", "field_amplitude"],
        ["CLA-L01-CAV-SOL-02", "field_amplitude"],
    ]
    parameter_names = [
        ["CLA-L02-CAV", "field_amplitude"],
        ["CLA-L02-CAV", "phase"],
        ["CLA-L03-CAV", "field_amplitude"],
        ["CLA-L03-CAV", "phase"],
        ["CLA-L4H-CAV", "field_amplitude"],
        ["CLA-L4H-CAV", "phase"],
        ["CLA-L04-CAV", "field_amplitude"],
        ["CLA-L04-CAV", "phase"],
        ["bunch_compressor", "set_angle"],
        ["CLA-S07-DCP-01", "factor"],
    ]

    def __init__(self):
        super(Optimise_Genesis_Elegant, self).__init__()
        self.cons = constraintsClass()
        self.changes = None
        self.lattice = None
        self.resultsDict = {}
        self.opt_iteration = 0
        self.bestfit = 1e26
        self.beam_file = "CLA-S07-APER-01.hdf5"
        # ******************************************************************************
        CLARA_dir = os.path.relpath(__file__ + "/../../")

    def calculate_constraints(self):
        pass

    def set_changes_file(self, changes):
        self.changes = changes

    def set_lattice_file(self, lattice):
        self.lattice_file = lattice

    def OptimisingFunction(self, inputargs, **kwargs):
        if not self.post_injector:
            parameternames = self.injector_parameter_names + self.parameter_names
        else:
            parameternames = copy(self.parameter_names)

        self.inputlist = list(
            map(lambda a: a[0] + [a[1]], zip(parameternames, inputargs))
        )

        self.linac_fields = np.array(
            [i[2] for i in self.inputlist if i[1] == "field_amplitude"]
        )
        self.linac_phases = np.array([i[2] for i in self.inputlist if i[1] == "phase"])

        if "dir" in kwargs.keys():
            dir = kwargs["dir"]
            del kwargs["dir"]
        else:
            dir = self.optdir + str(self.opt_iteration)

        e, b, ee, be, l, g = self.run_simulation(self.inputlist, dir, **kwargs)
        if e < 0.01:
            print("e too low! ", e)
            l = 500
        self.resultsDict.update(
            {
                "e": e,
                "b": b,
                "ee": ee,
                "be": be,
                "l": l,
                "g": g,
                "brightness": (1e-4 * e) / (1e-2 * b),
                "momentum": 1e-6 * np.mean(g.momentum),
            }
        )
        constraintsList = self.calculate_constraints()
        fitvalue = self.cons.constraints(constraintsList)
        print(self.cons.constraintsList(constraintsList))
        print("fitvalue[", self.opt_iteration, "] = ", fitvalue)
        saveState(inputargs, self.opt_iteration, fitvalue)
        if fitvalue < self.bestfit:
            print("!!!!!!  New best = ", fitvalue)
            copyfile(dir + "/changes.yaml", self.best_changes)
            self.bestfit = fitvalue
        if isinstance(self.opt_iteration, (int, float)):
            self.opt_iteration += 1
        return fitvalue

    def Nelder_Mead(self, best=None, step=0.1):
        best = np.array(self.best) if best is None else np.array(best)
        self.optdir = "nelder_mead/iteration_"
        self.best_changes = "./nelder_mead_best_changes.yaml"
        print("best = ", best)
        self.bestfit = 1e26

        with open("nelder_mead/best_solutions_running.csv", "w") as out:
            self.opt_iteration = 0
        res = nelder_mead(
            self.OptimisingFunction, best, step=step, max_iter=300, no_improv_break=100
        )
        print(res)

    def Simplex(self, best=None):
        best = self.best if best is None else best
        self.optdir = "simplex/iteration_"
        self.best_changes = "./simplex_best_changes.yaml"
        print("best = ", best)
        self.bestfit = 1e26

        with open("simplex/best_solutions_running.csv", "w") as out:
            self.opt_iteration = 0
        res = minimize(
            self.OptimisingFunction,
            best,
            method="nelder-mead",
            options={"disp": True, "maxiter": 300, "adaptive": True},
        )
        print(res.x)
