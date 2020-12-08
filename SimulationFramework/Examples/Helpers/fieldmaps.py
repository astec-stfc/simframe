"""
Tools for loading fieldmap data
"""

import sys
import os
from io import StringIO
sys.path.append('../../../')
import SimulationFramework.Framework as fw
import matplotlib.pyplot as plt
# import SimulationFramework.Modules.read_twiss_file as rtf
# import SimulationFramework.Modules.read_beam_file as rbf
import numpy as np
# beamobject = rbf.beam()

def fieldmap_data(element):
    """
    Loads the fieldmap in absolute coordinates.

    If a fieldmaps dict is given, thes will be used instead of loading the file.

    """

    # Position
    offset = element.position_start[2]

    # Scaling
    scale = element.get_field_amplitude

    # file
    element.update_field_definition()
    file = element.field_definition.strip('"')

    # print(f'loading from file {file}')
    with open(file) as f:
        firstline = f.readline()
        c = StringIO(firstline)
        header = np.loadtxt(c)
    if len(header) == 4:
        dat = np.loadtxt(file, skiprows=1)
        start, stop, n, p = header
        fielddat = dat[1:]
        zpos = list(1*fielddat[:,0])
        startpos = zpos.index(start)
        stoppos = zpos.index(stop)
        halfcell1 = 1*fielddat[:startpos]
        halfcell2 = 1*fielddat[stoppos:]
        rfcell = 1*dat[startpos-1:stoppos+1]
        # rfcell[:,0] -= start
        # rfcell[:,0] /= p
        # rfcell[:,0] += start
        n_cells = int(element.cells / p)
        cell_length = rfcell[-1,0] - rfcell[0,0]
        dat = list(halfcell1)
        for i in range(0,n_cells+1,1):
            dat += list(1.0*rfcell)
            rfcell[:,0] += cell_length
        halfcell2[:,0] += n_cells*cell_length
        dat += list(halfcell2)
        dat = np.array(dat)
    else:
        dat = np.loadtxt(file)
    dat[:,0] += offset
    dat[:,1] *= scale
    return dat

def load_fieldmaps(lattice, sections='All', types=['cavity', 'solenoid'], verbose=False):
    fmap = {}
    for t in types:
        fmap[t] = {}
        if sections == 'All':
            elements = framework.getElementType(t)
        else:
            elements = []
            for s in sections:
                elements += lattice[s].getElementType(t)
        for e in elements:
            fmap[t][e.objectname] = fieldmap_data(e)
    return fmap

def add_fieldmaps_to_axes(lattice, axes, bounds=None, sections='All',
                           types=['cavity', 'solenoid'],
                          include_labels=True, verbose=False):
    """
    Adds fieldmaps to an axes.

    """

    fmaps = load_fieldmaps(lattice, sections=sections, verbose=verbose)
    ax1 = axes

    ax1rhs = ax1.twinx()
    ax = [ax1, ax1rhs]

    ylabel = {'cavity': '$E_z$ (MV/m)', 'solenoid':'$B_z$ (T)'}
    color = {'cavity': 'green', 'solenoid':'blue'}

    for i, section in enumerate(types):
        a = ax[i]
        for name, data in fmaps[section].items():
            label = f'{section}_{name}'
            c = color[section]
            a.plot(*data.T, label=label, color=c)
        a.set_ylabel(ylabel[section])
    ax1.set_xlabel('$z$ (m)')

    if bounds:
        ax1.set_xlim(bounds[0], bounds[1])


def plot_fieldmaps(lattice, sections='All', include_labels=True,  xlim=None, figsize=(12,4), **kwargs):
    """
    Simple fieldmap plot
    """

    fig, axes = plt.subplots(figsize=figsize, **kwargs)

    add_fieldmaps_to_axes(lattice, axes, bounds=xlim, include_labels=include_labels,
                          sections=sections, types=['cavity', 'solenoid'])


if __name__ == 'main':
    framework = fw.Framework('./', clean=False, verbose=False)
    framework.loadSettings('Lattices/clara400_v12_v3.def')
    import matplotlib
    import matplotlib.pyplot as plt
    plot_fieldmaps(framework, sections=['injector400','S02','L02'])
    plt.show()
