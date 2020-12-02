import os
import sys
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from copy import copy
from ..units import nice_array, nice_scale_prefix

from fastkde import fastKDE


CMAP0 = copy(plt.get_cmap('viridis'))
CMAP0.set_under('white')
CMAP1 = copy(plt.get_cmap('plasma'))

# beamobject = rbf.beam()

def density_plot(particle_group, key='x', bins=None, **kwargs):
    """
    1D density plot. Also see: marginal_plot

    Example:

        density_plot(P, 'x', bins=100)

    """

    if not bins:
        n = len(particle_group)
        bins = int(n/100)
    # Scale to nice units and get the factor, unit prefix
    x, f1, p1 = nice_array(particle_group[key])
    if key is not 'charge':
        w = abs(particle_group['charge'])
    else:
        w = np.ones(len(particle_group[key]))
    u1 = ''#particle_group.units(key).unitSymbol
    ux = p1+u1

    labelx = f'{key} ({ux})'

    fig, ax = plt.subplots(**kwargs)
    hist, bin_edges = np.histogram(x, bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_width =  np.diff(bin_edges)
    hist_y, hist_f, hist_prefix = nice_array(hist/hist_width)
    ax.bar(hist_x, hist_y, hist_width, color='grey')
    # Special label for C/s = A
    if u1 == 's':
        _, hist_prefix = nice_scale_prefix(hist_f/f1)
        ax.set_ylabel(f'{hist_prefix}A')
    else:
        ax.set_ylabel(f'{hist_prefix}C/{ux}')


    ax.set_xlabel(labelx)

def slice_plot(particle_group, xkey='t', ykey='slice_current',  xlim=None, nice=True, include_legend=True, subtract_mean=True, bins=None, **kwargs):
    """
    slice plot. Also see: marginal_plot

    Example:

        slice plot(P, 'slice_current', bins=100)

    """

    P = particle_group

    fig, all_axis = plt.subplots( **kwargs)
    ax_plot = [all_axis]

    if not bins:
        n = len(particle_group)
        bins = int(n/100)
    P.slice.slices = bins

    X = getattr(P, 'slice_'+xkey)
    if subtract_mean:
        X = X - np.mean(X)

    if isinstance(ykey, str):
        ykey = [ykey]
    if not isinstance(ykey, (list, tuple)):
        ykey = [ykey]
    if len(ykey)==1:
        include_legend=False

    # Only get the data we need
    if xlim:
        good = np.logical_and(X >= xlim[0], X <= xlim[1])
        X = X[good]
    else:
        xlim = X.min(), X.max()
        good = slice(None,None,None) # everything

    # X axis scaling
    units_x = 's'#str(P.units(xkey))
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
    for ix, keys in enumerate([ykey]):
        if not keys:
            continue
        ax = ax_plot[ix]
        linestyle = linestyles[ix]

        # Check that units are compatible
        ulist = ['' for key in keys]#[I.units(key) for key in keys]
        if len(ulist) > 1:
            for u2 in ulist[1:]:
                assert ulist[0] == u2, f'Incompatible units: {ulist[0]} and {u2}'
        # String representation
        unit = str(ulist[0])

        # Data
        data = [np.array(getattr(P, key)[good]) for key in keys]

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

def marginal_plot(particle_group, key1='t', key2='p', bins=None, **kwargs):
    """
    Density plot and projections

    Example:

        marginal_plot(P, 't', 'energy', bins=200)

    """

    if not bins:
        n = len(particle_group)
        bins = int(np.sqrt(n/4) )

    # Scale to nice units and get the factor, unit prefix
    x, f1, p1 = nice_array(particle_group[key1])
    y, f2, p2 = nice_array(particle_group[key2])

    w = np.full(len(x), 1)#particle_group['charge']

    u1 = ''#particle_group.units(key1).unitSymbol
    u2 = ''#particle_group.units(key2).unitSymbol
    ux = p1+u1
    uy = p2+u2

    labelx = f'{key1} ({ux})'
    labely = f'{key2} ({uy})'

    fig = plt.figure(**kwargs)

    gs = GridSpec(4,4)

    ax_joint =  fig.add_subplot(gs[1:4,0:3])
    ax_marg_x = fig.add_subplot(gs[0,0:3])
    ax_marg_y = fig.add_subplot(gs[1:4,3])
    #ax_info = fig.add_subplot(gs[0, 3:4])
    #ax_info.table(cellText=['a'])

    # Proper weighting
    ax_joint.hexbin(x, y, C=w, reduce_C_function=np.sum, gridsize=bins, cmap=CMAP0, vmin=1e-20)

    # Manual histogramming version
    #H, xedges, yedges = np.histogram2d(x, y, weights=w, bins=bins)
    #extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #ax_joint.imshow(H.T, cmap=cmap, vmin=1e-16, origin='lower', extent=extent, aspect='auto')



    # Top histogram
    # Old method:
    #dx = x.ptp()/bins
    #ax_marg_x.hist(x, weights=w/dx/f1, bins=bins, color='gray')
    hist, bin_edges = np.histogram(x, bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_width =  np.diff(bin_edges)
    hist_y, hist_f, hist_prefix = nice_array(hist/hist_width)
    ax_marg_x.bar(hist_x, hist_y, hist_width, color='gray')
    # Special label for C/s = A
    if u1 == 's':
        _, hist_prefix = nice_scale_prefix(hist_f/f1)
        ax_marg_x.set_ylabel(f'{hist_prefix}A')
    else:
        ax_marg_x.set_ylabel(f'{hist_prefix}C/{ux}')


    # Side histogram
    # Old method:
    #dy = y.ptp()/bins
    #ax_marg_y.hist(y, orientation="horizontal", weights=w/dy, bins=bins, color='gray')
    hist, bin_edges = np.histogram(y, bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_width =  np.diff(bin_edges)
    hist_y, hist_f, hist_prefix = nice_array(hist/hist_width)
    ax_marg_y.barh(hist_x, hist_y, hist_width, color='gray')
    ax_marg_y.set_xlabel(f'{hist_prefix}C/{uy}')

    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Set labels on joint
    ax_joint.set_xlabel(labelx)
    ax_joint.set_ylabel(labely)

def plot(self, keys=None, bins=None, type='density', **kwargs):

    if keys is not None and ((isinstance(keys, (list, tuple)) and len(keys) == 1) or isinstance(keys, str)):
        if isinstance(keys, (list, tuple)):
            ykey = keys[0]
        if type == 'slice' or 'slice_' in ykey:
            return slice_plot(self, ykey=ykey, bins=bins, **kwargs)
        elif type == 'density':
            return density_plot(self, key=ykey, bins=bins, **kwargs)
    else:
        xkey, ykey = keys
        return marginal_plot(self, key1=xkey, key2=ykey, bins=bins, **kwargs)

def plotScreenImage(beam, scale=1, colormap=plt.cm.jet, size=15, **kwargs):
    #Do the self-consistent density estimate
    myPDF,axes = fastKDE.pdf(1e3*(beam.x-np.mean(beam.x)),1e3*(beam.y-np.mean(beam.y)), **kwargs)
    v1,v2 = axes
    myPDF=myPDF/myPDF.max()*scale

    fig, ax = plt.subplots()
    ax.set_aspect(1)
    #ax.xaxis.set_visible(False)
    #ax.yaxis.set_visible(False)
    draw_circle = plt.Circle((0,0), size+0.05, fill=True, ec='w', fc=colormap(0), zorder=-1)
    ax.add_artist(draw_circle)

    # Make a circle
    circ = plt.Circle((0,0), size, facecolor='none')
    ax.add_patch(circ) # Plot the outline

    ax.set_facecolor('k')
    plt.pcolormesh(v1,v2,myPDF, cmap=colormap, shading='auto', vmax=1, clip_path=circ);
    ax.set_xlim([-(size + 1), (size + 1)])
    ax.set_ylim([-(size + 1), (size + 1)])
    file, ext = os.path.splitext(os.path.basename(beam.filename))
    plt.title(file)
    plt.show()
