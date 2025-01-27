import sys
import os
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from copy import copy
import numpy as np
from .units import nice_array, nice_scale_prefix
from mpl_axes_aligner import align

# from units import nice_array, nice_scale_prefix

CMAP0 = copy(plt.get_cmap("viridis"))
CMAP0.set_under("white")
CMAP1 = copy(plt.get_cmap("plasma"))


def fieldmap_data(element, directory="."):
    """
    Loads the fieldmap in absolute coordinates.

    If a fieldmaps dict is given, thes will be used instead of loading the file.

    """

    # Position
    offset = element.position_start[2]

    # Scaling
    scale = element.get_field_amplitude
    if element.objecttype == "cavity":
        scale = scale / 1e6

    # file
    element.update_field_definition()
    file = os.path.abspath(os.path.join(directory, element.field_definition.strip('"')))

    # print(f'loading from file {file}')
    with open(file) as f:
        firstline = f.readline()
        c = StringIO(firstline)
        header = np.loadtxt(c)
    if len(header) == 4:
        dat = np.loadtxt(file, skiprows=1)
        start, stop, n, p = header
        fielddat = dat[1:]
        zpos = list(1 * fielddat[:, 0])
        startpos = zpos.index(start)
        stoppos = zpos.index(stop)
        halfcell1 = 1 * fielddat[:startpos]
        halfcell2 = 1 * fielddat[stoppos:]
        rfcell = 1 * dat[startpos - 1 : stoppos + 1]
        # rfcell[:,0] -= start
        # rfcell[:,0] /= p
        # rfcell[:,0] += start
        n_cells = int(element.cells / p)
        cell_length = rfcell[-1, 0] - rfcell[0, 0]
        dat = list(halfcell1)
        for i in range(0, n_cells + 1, 1):
            dat += list(1.0 * rfcell)
            rfcell[:, 0] += cell_length
        halfcell2[:, 0] += n_cells * cell_length
        dat += list(halfcell2)
        dat = np.array(dat)
    else:
        dat = np.loadtxt(file)
    dat[:, 0] += offset
    x = dat[:, 1]
    normalise = max(x.min(), x.max(), key=abs)
    dat[:, 1] *= scale / normalise
    return dat


class magnet_plotting_data:

    def half_rectangle(self, e, half_height):
        return np.array(
            [
                [e.position_start[2], 0],
                [e.position_start[2], half_height],
                [e.position_end[2], half_height],
                [e.position_end[2], 0],
            ]
        )

    def full_rectangle(self, e, half_height, width=0):
        return np.array(
            [
                [e.position_start[2] - width, -half_height],
                [e.position_start[2] - width, half_height],
                [e.position_end[2] + width, half_height],
                [e.position_end[2] + width, -half_height],
            ]
        )

    def quadrupole(self, e):
        strength = np.sign(e.k1l) * 0.5  # e.k1l / e.length
        return self.half_rectangle(e, strength), "red"

    def dipole(self, e):
        strength = np.sign(e.angle) * 0.4  # e.angle
        return self.half_rectangle(e, strength), "gray"

    def beam_position_monitor(self, e):
        strength = 0.1  # e.angle
        return self.full_rectangle(e, strength), "purple"

    def screen(self, e):
        strength = 0.33  # e.angle
        return self.full_rectangle(e, strength), "green"

    def aperture(self, e):
        strength = 0.15  # e.angle
        return self.full_rectangle(e, strength, width=0.01), "black"

    def wall_current_monitor(self, e):
        strength = 0.33  # e.angle
        return self.full_rectangle(e, strength), "brown"


def load_elements(
    lattice,
    bounds=None,
    sections="All",
    types=["cavity", "solenoid"],
    verbose=False,
    scale=1,
):
    fmap = {}
    for t in types:
        fmap[t] = {}
        if sections == "All":
            elements = lattice.getElementType(t)
        else:
            elements = []
            for s in sections:
                elements += lattice[s].getElementType(t)
        if bounds is not None:
            elements = [
                e
                for e in elements
                if e.position_start[2] <= bounds[1]
                and e.position_end[2] >= bounds[0] - 0.1
            ]
        for e in elements:
            if (
                (t == "cavity" or t == "solenoid")
                and hasattr(e, "field_definition")
                and e.field_definition is not None
            ):
                fmap[t][e.objectname] = fieldmap_data(e, directory=lattice.subdirectory)
            elif hasattr(magnet_plotting_data, t):
                fmap[t][e.objectname] = getattr(magnet_plotting_data(), t)(e)
            else:
                print("Missing drawings for", t)
    return fmap


def add_fieldmaps_to_axes(
    lattice,
    axes,
    bounds=None,
    sections="All",
    fields=["cavity", "solenoid"],
    magnets=["quadrupole", "dipole", "beam_position_monitor", "screen"],
    include_labels=True,
    verbose=False,
):
    """
    Adds fieldmaps to an axes.

    """

    max_scale = 0

    fmaps = load_elements(
        lattice, bounds=bounds, sections=sections, verbose=verbose, types=fields
    )
    ax1 = axes

    ax1rhs = ax1.twinx()
    ax = [ax1, ax1rhs]

    ylabel = {"cavity": "$E_z$ (MV/m)", "solenoid": "$B_z$ (T)"}
    color = {"cavity": "green", "solenoid": "blue"}

    for i, section in enumerate(fields):
        a = ax[i]
        for name, data in fmaps[section].items():
            label = f"{section}_{name}"
            c = color[section]
            # if section == 'cavity':# and not section == 'solenoid':
            if section == fields[0]:
                max_scale = (
                    max(abs(data[:, 1]))
                    if max(abs(data[:, 1])) > max_scale
                    else max_scale
                )
            a.plot(*data.T, label=label, color=c)
        a.set_ylabel(ylabel[section])
    ax1.set_xlabel("$z$ (m)")

    if len(fields) < 1:
        for a in ax:
            a.set_yticks([])

    max_scale = 1 if max_scale == 0 else max_scale
    a = ax[0]

    fmaps = load_elements(
        lattice,
        bounds=bounds,
        sections=sections,
        verbose=verbose,
        types=magnets,
        scale=max_scale,
    )
    for section in magnets:
        if len(fmaps[section].items()) > 0:
            section_scale = (
                max_scale  # / max([max(abs(d[:,1])) for d in fmaps[section].values()])
            )
        for name, (data, c) in fmaps[section].items():
            # c = color[section]
            data[:, 1] = data[:, 1] * section_scale
            a.fill(*data.T, color=c)

    data = np.array([[0, 0], [100, 0]])
    ax[0].plot(*data.T, color="black")
    if bounds:
        ax1.set_xlim(bounds[0], bounds[1])

    align.yaxes(ax[0], 0, ax[1], 0, 0.5)


def plot_fieldmaps(
    lattice,
    sections="All",
    include_labels=True,
    limits=None,
    figsize=(12, 4),
    fields=["cavity", "solenoid"],
    magnets=["quadrupole", "dipole", "beam_position_monitor", "screen"],
    **kwargs,
):
    """
    Simple fieldmap plot
    """

    fig, axes = plt.subplots(figsize=figsize, **kwargs)

    add_fieldmaps_to_axes(
        lattice,
        axes,
        bounds=limits,
        include_labels=include_labels,
        sections=sections,
        fields=fields,
        magnets=magnets,
    )


def plot(
    framework_object,
    ykeys=["sigma_x", "sigma_y"],
    ykeys2=["sigma_z"],
    xkey="z",
    limits=None,
    nice=True,
    include_layout=False,
    include_labels=True,
    include_legend=True,
    include_particles=False,
    fields=["cavity", "solenoid"],
    magnets=[
        "quadrupole",
        "dipole",
        "beam_position_monitor",
        "screen",
        "wall_current_monitor",
        "aperture",
    ],
    grid=False,
    **kwargs,
):
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
    I = framework_object.twiss  # convenience
    I.sort()  # sort before plotting!
    P = framework_object.beams

    if include_layout is not False:
        fig, all_axis = plt.subplots(2, gridspec_kw={"height_ratios": [4, 1]}, **kwargs)
        ax_layout = all_axis[-1]
        ax_plot = [all_axis[0]]
    else:
        fig, all_axis = plt.subplots(**kwargs)
        ax_plot = [all_axis]

    if grid:
        ax_plot[0].grid(visible=True, which="major", color="#666666", linestyle="-")

    # collect axes
    if isinstance(ykeys, str):
        ykeys = [ykeys]

    if ykeys2:
        if isinstance(ykeys2, str):
            ykeys2 = [ykeys2]
        ax_plot.append(ax_plot[0].twinx())

    # No need for a legend if there is only one plot
    if len(ykeys) == 1 and not ykeys2:
        include_legend = False

    X = I.stat(xkey)

    # Only get the data we need
    if limits:
        good = np.logical_and(X >= limits[0], X <= limits[1])
        idx = list(np.where(good == True)[0])
        if len(idx) > 0:
            if idx[0] > 0:
                good[idx[0] - 1] = True
            if (idx[-1] + 1) < len(good):
                good[idx[-1] + 1] = True
            X = X[good]
        if X.min() > limits[0]:
            limits[0] = X.min()
        if X.max() < limits[1]:
            limits[1] = X.max()
    else:
        limits = X.min(), X.max()
        good = slice(None, None, None)  # everything

    # Try particles within these bounds
    Pnames = []
    X_particles = []

    if include_particles:
        # try:
        for pname in range(len(P)):  # Modified from Impact
            xp = np.mean(np.array(P[pname][xkey]))
            if xp >= limits[0] and xp <= limits[1]:
                Pnames.append(pname)
                X_particles.append(xp)
        X_particles = np.array(X_particles)
    # except:
    #     Pnames = []
    else:
        Pnames = []

    # X axis scaling
    units_x = str(I.units(xkey))
    if nice:
        X, factor_x, prefix_x = nice_array(X)
        units_x = prefix_x + units_x
    else:
        factor_x = 1

    # set all but the layout
    for ax in ax_plot:
        ax.set_xlim(limits[0] / factor_x, limits[1] / factor_x)
        ax.set_xlabel(f"{xkey} ({units_x})")

    # Draw for Y1 and Y2

    linestyles = ["solid", "dashed"]

    ii = -1  # counter for colors
    for ix, keys in enumerate([ykeys, ykeys2]):
        if not keys:
            continue

        ax = ax_plot[ix]
        linestyle = linestyles[ix]

        # Check that units are compatible
        ulist = [I.units(key) for key in keys]
        if len(ulist) > 1:
            for u2 in ulist[1:]:
                assert ulist[0] == u2, f"Incompatible units: {ulist[0]} and {u2}"
        # String representation
        unit = str(ulist[0])

        # Data
        data = [I.stat(key)[good] for key in keys]

        if nice:
            factor, prefix = nice_scale_prefix(np.ptp(data))
            unit = prefix + unit
        else:
            factor = 1

        # Make a line and point
        for key, dat in zip(keys, data):
            #
            ii += 1
            color = "C" + str(ii)
            ax.plot(
                X,
                dat / factor,
                label=f"{key} ({unit})",
                color=color,
                linestyle=linestyle,
            )

            # Particles
            if Pnames:
                # try:
                # print(Pnames, [key in P._parameters['data'] for key in Pnames])
                Y_particles = np.array(
                    [
                        (
                            np.mean(P[name][key])
                            if key in P._parameters["data"]
                            else P[name][key]
                        )
                        for name in Pnames
                    ]
                )
                ax.scatter(X_particles / factor_x, Y_particles / factor, color=color)
            # except:
            #     pass
        keys = ["$" + k.replace("sigma", "\sigma") + "$" for k in keys]
        ax.set_ylabel(", ".join(keys) + f" ({unit})")

    # Collect legend
    if include_legend:
        lines = []
        labels = []
        for ax in ax_plot:
            a, b = ax.get_legend_handles_labels()
            lines += a
            labels += b
        ax_plot[0].legend(lines, labels, loc="best")

    # Layout
    if include_layout is not False:

        # Gives some space to the top plot
        # ax_layout.set_ylim(-1, 1.5)

        if xkey == "z":
            # ax_layout.set_axis_off()
            ax_layout.set_xlim(limits[0], limits[1])
        # else:
        #     ax_layout.set_xlabel('mean_z')
        #     limits = (0, I.stop)
        add_fieldmaps_to_axes(
            framework_object.framework,
            ax_layout,
            bounds=limits,
            include_labels=include_labels,
            fields=fields,
            magnets=magnets,
        )


def getattrsplit(self, attr):
    attrs = attr.split(".")
    for a in attrs:
        self = getattr(self, a)
    return self


def general_plot(
    framework_object,
    ykeys=[],
    ykeys2=[],
    xkey=[],
    limits=None,
    nice=True,
    include_layout=False,
    include_labels=True,
    include_legend=True,
    include_particles=False,
    fields=["cavity", "solenoid"],
    magnets=[
        "quadrupole",
        "dipole",
        "beam_position_monitor",
        "screen",
        "wall_current_monitor",
        "aperture",
    ],
    grid=False,
    **kwargs,
):

    if include_layout is not False:
        fig, all_axis = plt.subplots(2, gridspec_kw={"height_ratios": [4, 1]}, **kwargs)
        ax_layout = all_axis[-1]
        ax_plot = [all_axis[0]]
    else:
        fig, all_axis = plt.subplots(**kwargs)
        ax_plot = [all_axis]

    if grid:
        ax_plot[0].grid(b=True, which="major", color="#666666", linestyle="-")

    # collect axes
    if isinstance(ykeys, str):
        ykeys = [ykeys]

    if ykeys2:
        if isinstance(ykeys2, str):
            ykeys2 = [ykeys2]
        ax_plot.append(ax_plot[0].twinx())

    # Ensure we are using numpy arrays
    xdata = getattrsplit(framework_object, xkey)
    ydata = [getattrsplit(framework_object, y) for y in ykeys]
    ydata2 = [getattrsplit(framework_object, y) for y in ykeys2]

    # Split keys
    xkey = xkey.split(".")[-1]
    ykeys = [yk.split(".")[-1] for yk in ykeys]
    ykeys2 = [yk.split(".")[-1] for yk in ykeys2]

    # No need for a legend if there is only one plot
    if len(ydata) == 1 and not ydata2:
        include_legend = False

    X = xdata

    # Only get the data we need
    if limits:
        good = np.logical_and(X >= limits[0], X <= limits[1])
        idx = list(np.where(good == True)[0])
        if len(idx) > 0:
            if idx[0] > 0:
                good[idx[0] - 1] = True
            if (idx[-1] + 1) < len(good):
                good[idx[-1] + 1] = True
            X = X[good]
        if X.min() > limits[0]:
            limits[0] = X.min()
        if X.max() < limits[1]:
            limits[1] = X.max()
    else:
        limits = X.min(), X.max()
        good = slice(None, None, None)  # everything

    # X axis scaling
    units_x = xdata.units
    if nice:
        X, factor_x, prefix_x = nice_array(X)
        units_x = prefix_x + units_x
    else:
        factor_x = 1

    # set all but the layout
    for ax in ax_plot:
        ax.set_xlim(limits[0] / factor_x, limits[1] / factor_x)
        ax.set_xlabel(f"{xkey} ({units_x})")

    # Draw for Y1 and Y2

    linestyles = ["solid", "dashed"]

    ii = -1  # counter for colors
    for ix, (d, keys) in enumerate([[ydata, ykeys], [ydata2, ykeys2]]):
        if not keys:
            continue
        ax = ax_plot[ix]
        linestyle = linestyles[ix]

        # Check that units are compatible
        ulist = [dat.units for dat in d]
        if len(ulist) > 1:
            for u2 in ulist[1:]:
                assert ulist[0] == u2, f"Incompatible units: {ulist[0]} and {u2}"
        # String representation
        unit = str(ulist[0])

        # Data
        data = [key[good] for key in d]

        if nice:
            factor, prefix = nice_scale_prefix(np.ptp(data))
            unit = prefix + unit
        else:
            factor = 1

        # Make a line and point
        keys = ["$" + k.replace("sigma", "\sigma") + "$" for k in keys]
        for key, dat in zip(keys, data):
            #
            ii += 1
            color = "C" + str(ii)
            ax.plot(
                X,
                dat / factor,
                label=f"{key} ({unit})",
                color=color,
                linestyle=linestyle,
            )

        ax.set_ylabel(", ".join(keys) + f" ({unit})")

    # Collect legend
    if include_legend:
        lines = []
        labels = []
        for ax in ax_plot:
            a, b = ax.get_legend_handles_labels()
            lines += a
            labels += b
        ax_plot[0].legend(lines, labels, loc="best")

    # Layout
    if include_layout is not False:

        # Gives some space to the top plot
        # ax_layout.set_ylim(-1, 1.5)

        if xkey == "z":
            # ax_layout.set_axis_off()
            ax_layout.set_xlim(limits[0], limits[1])
        # else:
        #     ax_layout.set_xlabel('mean_z')
        #     limits = (0, I.stop)
        add_fieldmaps_to_axes(
            framework_object.framework,
            ax_layout,
            bounds=limits,
            include_labels=include_labels,
            fields=fields,
            magnets=magnets,
        )
