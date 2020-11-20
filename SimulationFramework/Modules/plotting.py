import sys
import os
from io import StringIO
sys.path.append('../../')
import SimulationFramework.Framework as fw
import matplotlib.pyplot as plt
# import SimulationFramework.Modules.read_twiss_file as rtf
# import SimulationFramework.Modules.read_beam_file as rbf
import numpy as np
sys.path.append('../../../')
from openPMD.pmd_beamphysics.units import nice_array, nice_scale_prefix

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
    if element.objecttype == 'cavity':
        scale = scale / 1e6

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
    dat[:,1] *= scale / max(abs(dat[:,1]))
    return dat

def load_fieldmaps(lattice, sections='All', types=['cavity', 'solenoid'], verbose=False):
    fmap = {}
    for t in types:
        fmap[t] = {}
        if sections == 'All':
            elements = lattice.getElementType(t)
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


def plot_stats_with_layout(twiss_object, ykeys=['sigma_x', 'sigma_y'], ykeys2=['sigma_z'],
                           xkey='z', xlim=None,
                           nice=True,
                           include_layout=False,
                           include_labels=True,
                           include_legend=True, **kwargs):
    """
    Plots stat output multiple keys.

    If a list of ykeys2 is given, these will be put on the right hand axis. This can also be given as a single key.

    Logical switches, all default to True:
        nice: a nice SI prefix and scaling will be used to make the numbers reasonably sized.

        include_legend: The plot will include the legend

        include_layout: the layout plot will be displayed at the bottom

        include_labels: the layout will include element labels.

    Copied almost verbatim from lume-impact's Impact.plot.plot_stats_with_layout
    """
    I = twiss_object # convenience
    I.sort() # sort before plotting!

    if include_layout is not False:
        fig, all_axis = plt.subplots(2, gridspec_kw={'height_ratios': [4, 1]}, **kwargs)
        ax_layout = all_axis[-1]
        ax_plot = [all_axis[0]]
    else:
        fig, all_axis = plt.subplots( **kwargs)
        ax_plot = [all_axis]

    # collect axes
    if isinstance(ykeys, str):
        ykeys = [ykeys]

    if ykeys2:
        if isinstance(ykeys2, str):
            ykeys2 = [ykeys2]
        ax_plot.append(ax_plot[0].twinx())

    # No need for a legend if there is only one plot
    if len(ykeys)==1 and not ykeys2:
        include_legend=False

    X = I.stat(xkey)

    # Only get the data we need
    if xlim:
        good = np.logical_and(X >= xlim[0], X <= xlim[1])
        X = X[good]
    else:
        xlim = X.min(), X.max()
        good = slice(None,None,None) # everything

    # X axis scaling
    units_x = str(I.units(xkey))
    if nice:
        X, factor_x, prefix_x = nice_array(X)
        units_x  = prefix_x+units_x
    else:
        factor_x = 1

    # set all but the layout
    for ax in ax_plot:
        ax.set_xlim(xlim[0]/factor_x, xlim[1]/factor_x)
        ax.set_xlabel(f'{xkey} ({units_x})')


    # Draw for Y1 and Y2

    linestyles = ['solid','dashed']

    ii = -1 # counter for colors
    for ix, keys in enumerate([ykeys, ykeys2]):
        if not keys:
            continue
        ax = ax_plot[ix]
        linestyle = linestyles[ix]

        # Check that units are compatible
        ulist = [I.units(key) for key in keys]
        if len(ulist) > 1:
            for u2 in ulist[1:]:
                assert ulist[0] == u2, f'Incompatible units: {ulist[0]} and {u2}'
        # String representation
        unit = str(ulist[0])

        # Data
        data = [I.stat(key)[good] for key in keys]



        if nice:
            factor, prefix = nice_scale_prefix(np.ptp(data))
            unit = prefix+unit
        else:
            factor = 1

        # Make a line and point
        for key, dat in zip(keys, data):
            #
            ii += 1
            color = 'C'+str(ii)
            ax.plot(X, dat/factor, label=f'{key} ({unit})', color=color, linestyle=linestyle)

        ax.set_ylabel(', '.join(keys)+f' ({unit})')

    # Collect legend
    if include_legend:
        lines = []
        labels = []
        for ax in ax_plot:
            a, b = ax.get_legend_handles_labels()
            lines += a
            labels += b
        ax_plot[0].legend(lines, labels, loc='best')

    # Layout
    if include_layout is not False:

        # Gives some space to the top plot
        #ax_layout.set_ylim(-1, 1.5)

        if xkey == 'z':
            #ax_layout.set_axis_off()
            ax_layout.set_xlim(xlim[0], xlim[1])
        # else:
        #     ax_layout.set_xlabel('mean_z')
        #     xlim = (0, I.stop)
        add_fieldmaps_to_axes(include_layout,  ax_layout, bounds=xlim, include_labels=include_labels)
