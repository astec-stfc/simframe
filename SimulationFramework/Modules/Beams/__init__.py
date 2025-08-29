"""
Simframe Beams Module

This module defines the base class and utilities for representing particle beams and groups of beams.

Each beam consists of particles (see :class:`~SimulationFramework.Modules.Beams.Particles.Particles`),
represented in 6-dimensional phase space (x, cpx, y, cpy, z, cpz).

Functions are provided to read/write the particle distribution from a range of simulation codes.

The `beamGroup` class is used for loading and analysing a group of beam distributions,
for example from a directory.

Classes:
    - :class:`~SimulationFramework.Modules.Beams.beam`: Generic container for a particle beam.

    - :class:`~SimulationFramework.Modules.Beams.beamGroup`: Container for a group of particle beams.

    - :class:`~SimulationFramework.Modules.Beams.particlesGroup`: Container for a group of particle distributions.
"""
import os
from pydantic import (
    BaseModel,
    ConfigDict,
)
from typing import Dict, Any, List
import numpy as np
from warnings import warn
import re
import copy
import glob
import h5py
from ..units import UnitValue
from .. import constants
from .Particles import Particles
from . import astra
from . import sdds
from . import gdf
from . import hdf5
from . import mad8

try:
    from . import plot

    use_matplotlib = True
except ImportError as e:
    print("Import error - plotting disabled. Missing package:", e)
    use_matplotlib = False

from .Particles.emittance import emittance as emittanceobject
from .Particles.twiss import twiss as twissobject
from .Particles.slice import slice as sliceobject
from .Particles.sigmas import sigmas as sigmasobject
from .Particles.centroids import centroids as centroidsobject
from .Particles.kde import kde as kdeobject

try:
    from .Particles.mve import MVE as MVEobject

    imported_mve = True
except ImportError:
    imported_mve = False


# I can't think of a clever way of doing this, so...
def get_properties(obj):
    props = [f for f in dir(obj) if type(getattr(obj, f)) is property and f != "__fields_set__"]
    if hasattr(obj, "model_fields"):
        props += list(obj.model_fields.keys())
    return props


parameters = {
    "data": get_properties(Particles),
    "emittance": get_properties(emittanceobject),
    "twiss": get_properties(twissobject),
    "slice": get_properties(sliceobject),
    "sigmas": get_properties(sigmasobject),
    "centroids": get_properties(centroidsobject),
    "kde": get_properties(kdeobject),
    "mve": get_properties(MVEobject) if imported_mve else [],
}


class particlesGroup(BaseModel):
    """
    Class for grouping together properties of multiple particle distributions, such as
    the :class:`~SimulationFramework.Modules.Beams.Particles.emittance.emittance` objects.
    """
    particles: List = None
    """List of :class:`~SimulationFramework.Modules.Beams.Particles.Particles` or its 
    sub-classes"""

    def __init__(self, particles=None, *args, **kwargs):
        super(particlesGroup, self).__init__(*args, **kwargs)
        self.particles = particles

    def __getitem__(self, key):
        if key == "particles":
            return getattr(self, key)
        else:
            data = [getattr(p, key) for p in self.particles]
            return UnitValue(data, units=data[0].units)


class statsGroup(object):
    """
    Class for grouping together statistical properties of multiple particle distributions.
    """

    def __init__(self, beam, function):
        self._beam = beam
        self._func = function

    def __getattr__(self, key):
        var = self._beam.__getitem__(key)
        return UnitValue([self._func(v) for v in var], units="m")
        # return np.sqrt(self._beam.covariance(var, var))


class beamGroup(BaseModel):
    """
    Class for grouping together multiple particle distributions. These distributions can be loaded in
    from a directory, for example, using the function
    :func:`~SimulationFramework.Modules.Beams.load_directory`.

    Properties such as the :class:`~SimulationFramework.Modules.Beams.Particles.emittance.emittance` objects
    for these distributions are stored as properties of the `beamGroup`.

    (see :class:`~SimulationFramework.Modules.Beams.particlesGroup`).
    """

    sddsindex: int = 0
    """Index for SDDS files"""

    beams: Dict = {}
    """Dictionary containing the :class:`~SimulationFramework.Modules.Beams.beam` objects,
    keyed by (file)name"""

    def __repr__(self):
        return repr(list(self.beams.keys()))

    def __len__(self):
        return len(list(self.beams.keys()))

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.beams.values())[key]
        elif isinstance(key, slice):
            return beamGroup(beams=list(self.beams.items())[key])
        elif hasattr(np, key):
            return statsGroup(self, getattr(np, key))
        for p in parameters:
            if key in parameters[p]:
                return getattr(self, key)
        else:
            return getattr(self, key)

    def __init__(self, filenames=[], beams=[], *args, **kwargs):
        super(beamGroup, self).__init__(*args, **kwargs)
        self.sddsindex = 0
        self.beams = dict()
        self._parameters = parameters
        for k, v in beams:
            self.beams[k] = v
        if isinstance(filenames, str):
            filenames = [filenames]
        for f in filenames:
            self.add(f)

    @property
    def data(self):
        return particlesGroup([b._beam for b in self.beams.values()])

    @property
    def sigmas(self):
        return particlesGroup([b._beam.sigmas for b in self.beams.values()])

    @property
    def centroids(self):
        return particlesGroup([b._beam.centroids for b in self.beams.values()])

    @property
    def twiss(self):
        return particlesGroup([b._beam.twiss for b in self.beams.values()])

    @property
    def slice(self):
        return particlesGroup([b._beam.slice for b in self.beams.values()])

    @property
    def emittance(self):
        return particlesGroup([b._beam.emittance for b in self.beams.values()])

    @property
    def kde(self):
        return particlesGroup([b._beam.kde for b in self.beams.values()])

    @property
    def mve(self):
        return particlesGroup([b._beam.mve for b in self.beams.values()])

    def sort(self, key="z", function="mean", *args, **kwargs):
        if isinstance(function, str) and hasattr(np, function):
            func = getattr(np, function)
        else:
            func = function
        self.beams = dict(
            sorted(
                self.beams.items(), key=lambda item: func(getattr(item[1], key)), *args, **kwargs
            )
        )
        return self

    def add(self, filename):
        if isinstance(filename, str):
            filename = [filename]
        for file in filename:
            if os.path.isdir(file):
                self.add_directory(file)
            elif os.path.isfile(file):
                file = file.replace("\\", "/")
                try:
                    self.beams[file] = beam(file)
                except Exception:
                    if file in self.beams:
                        del self.beams[file]

    def param(self, param):
        return [getattr(b._beam, param) for b in self.beams.values()]

    def getScreen(self, screen):
        for b in self.beams:
            if screen == os.path.splitext(os.path.basename(b))[0]:
                return self.beams[b]
        return None

    def getScreens(self):
        return {os.path.splitext(os.path.basename(b))[0]: b for b in self.beams.keys()}


class stats(object):

    def __init__(self, beam, function):
        self._beam = beam
        self._func = function

    def __getattr__(self, key):
        var = self._beam.__getitem__(key)
        return self._func(var)
        # return np.sqrt(self._beam.covariance(var, var))


class beam(BaseModel):
    """
    Class describing a particle distribution. The distribution is contained in the `beam` or `Particles` property
    of this class (see :class:`~SimulationFramework.Modules.Beams.Particles.Particles`).

    Additional results from analysis of the beam are contained in the following properties:

    - :attr:`~SimulationFramework.Modules.Beams.beam.sigmas` -- average beam properties,
    see :class:`~SimulationFramework.Modules.Beams.Particles.sigmas.sigmas`.

    - :attr:`~SimulationFramework.Modules.Beams.beam.centroids` -- beam centroids,
    see :class:`~SimulationFramework.Modules.Beams.Particles.centroids.centroids`.

    - :attr:`~SimulationFramework.Modules.Beams.beam.centroids` -- various emittance calculations,
    see :class:`~SimulationFramework.Modules.Beams.Particles.emittance.emittance`.

    - :attr:`~SimulationFramework.Modules.Beams.beam.kde` -- kernel density estimator,
    see :class:`~SimulationFramework.Modules.Beams.Particles.kde.kde`.

    - :attr:`~SimulationFramework.Modules.Beams.beam.mve` -- minimum volume ellipse,
    see :class:`~SimulationFramework.Modules.Beams.Particles.mve.MVE`.

    - :attr:`~SimulationFramework.Modules.Beams.beam.slices` -- calculations of slice properties,
    see :class:`~SimulationFramework.Modules.Beams.Particles.slice.slice`.

    - :attr:`~SimulationFramework.Modules.Beams.beam.twiss` -- Twiss parameters,
    see :class:`~SimulationFramework.Modules.Beams.Particles.twiss.twiss`.

    Functions are also provided for translating the particle distribution from and to HDF5 format
    (in-house developed or OpenPMD), ASTRA, GPT, OCELOT, or SDDS.
    """
    # particle_mass = UnitValue(constants.m_e, "kg")
    # E0 = UnitValue(particle_mass * constants.speed_of_light**2, "J")
    # E0_eV = UnitValue(E0 / constants.elementary_charge, "eV/c")
    q_over_c: UnitValue = UnitValue(constants.elementary_charge / constants.speed_of_light, "C/c")
    """Elementary charge divided by speed of light"""

    speed_of_light: UnitValue = UnitValue(constants.speed_of_light, "m/s")
    """Speed of light"""

    filename: str = None
    """Name of beam distribution file; if provided on instantiation, load the file into this object"""

    sddsindex: int = 0
    """Index for SDDS files"""

    code: str = None
    """Code from which the beam distribution was generated"""

    reference_particle: np.ndarray = None
    """Reference particle for ASTRA-type distributions"""

    longitudinal_reference: np.ndarray | str = None
    """Longitudinal reference position for ASTRA-type distributions"""

    starting_position: list | np.ndarray = [0, 0, 0]
    """Beam starting position [x,y,z]"""

    theta: float = 0
    """Horizontal angle of beam distribution"""

    offset: list | np.ndarray = [0, 0, 0]
    """Beam offset from nominal axis [x,y,z]"""

    particle_mass: np.ndarray = None
    """Particle mass in kg"""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    def __init__(self, filename=None, *args, **kwargs):
        super(beam, self).__init__(*args, **kwargs)
        self._beam = Particles()
        self._parameters = parameters
        # self.sigma = stats(self, lambda var:  np.sqrt(self._beam.covariance(var, var)))
        # self.mean = stats(self, np.mean)
        self.filename = filename
        # self.sddsindex = sddsindex
        self.code = None
        if self.filename is not None:
            self.read_beam_file(self.filename)

    def model_dump(self, *args, **kwargs) -> Dict:
        # Only include computed fields
        full_dump = super().model_dump(*args, **kwargs)
        full_dump.update({"Particles": self._beam.model_dump()})
        return full_dump

    @property
    def E0_eV(self) -> float:
        """
        Particle rest mass energy in eV

        Returns
        -------
        float
            Particle rest mass energy in eV; if already defined, just return the attribute;
            if not, calculate from the :attr:`~SimulationFramework.Modules.Beams.Particles` object;
            if not possible, assume electrons and calculate its rest mass energy
        """
        if hasattr(self, "particle_rest_energy_eV"):
            return self.particle_rest_energy_eV
        elif self._beam.particle_rest_energy is not None:
            return np.mean(self._beam.particle_rest_energy) / constants.elementary_charge
        else:
            particle_mass = UnitValue(constants.m_e, "kg")
            E0 = UnitValue(particle_mass * constants.speed_of_light ** 2, "J")
            E0_eV = UnitValue(E0 / constants.elementary_charge, "eV/c")
            return E0_eV

    @property
    def beam(self) -> Particles:
        """
        Property defining the particle distribution

        Returns
        -------
        :class:`~SimulationFramework.Modules.Beams.Particles.Particles`
            The particle distribution
        """
        return self._beam

    @property
    def Particles(self) -> Particles:
        """
        Property defining the particle distribution

        Returns
        -------
        :class:`~SimulationFramework.Modules.Beams.Particles.Particles`
            The particle distribution
        """
        return self._beam

    @property
    def data(self) -> Particles:
        """
        Property defining the particle distribution

        Returns
        -------
        :class:`~SimulationFramework.Modules.Beams.Particles.Particles`
            The particle distribution
        """
        return self._beam

    @property
    def sigmas(self) -> sigmasobject:
        """
        Property defining the beam sigmas

        Returns
        -------
        :class:`~SimulationFramework.Modules.Beams.Particles.sigmas.sigmas`
            Beam sigmas
        """
        return self._beam.sigmas

    @property
    def centroids(self) -> centroidsobject:
        """
        Property defining the beam centroids

        Returns
        -------
        :class:`~SimulationFramework.Modules.Beams.Particles.centroids.centroids`
            Beam centroids
        """
        return self._beam.centroids

    @property
    def twiss(self) -> twissobject:
        """
        Property defining the beam twiss properties

        Returns
        -------
        :class:`~SimulationFramework.Modules.Beams.Particles.twiss.twiss`
            Beam Twiss parameters
        """
        return self._beam.twiss

    @property
    def slice(self) -> sliceobject:
        """
        Property defining the beam slice properties

        Returns
        -------
        :class:`~SimulationFramework.Modules.Beams.Particles.slice.slice`
            Beam slice properties
        """
        return self._beam.slice

    @property
    def emittance(self) -> emittanceobject:
        """
        Property defining the beam emittances

        Returns
        -------
        :class:`~SimulationFramework.Modules.Beams.Particles.emittance.emittance`
            Beam emittance
        """
        return self._beam.emittance

    @property
    def kde(self) -> kdeobject:
        """
        Property defining the beam kernel density estimator

        Returns
        -------
        :class:`~SimulationFramework.Modules.Beams.Particles.kde.kde`
            KDE
        """
        return self._beam.kde

    @property
    def mve(self) -> Any:
        """
        Property defining the beam minimum volume ellipse

        Returns
        -------
        :class:`~SimulationFramework.Modules.Beams.Particles.`
            Beam sigmas
        """
        return self._beam.mve

    def rms(self, x, axis: int=None) -> float | np.ndarray   :
        """
        Calculate the RMS of a distribution

        Parameters
        ----------
        x: np.ndarray
            Array from which to calculate the RMS
        axis: int, optional
            Axis along which to calculate the RMS

        Returns
        -------
        float or np.ndarray
            RMS of the distribution
        """
        return np.sqrt(np.mean(x**2, axis=axis))

    def __len__(self):
        return len(self._beam.x)

    # def __getitem__(self, key):
    #     # print('beams key', key)
    #     for p in parameters:
    #         if key in parameters[p]:
    #             return getattr(self, key)
    #     if hasattr(np, key):
    #         return stats(self, getattr(np, key))
    #     if hasattr(self._beam, key):
    #         return getattr(self._beam, key)
    #     else:
    #         try:
    #             return getattr(self, key)
    #         except KeyError:
    #             raise AttributeError(key)

    def __setitem__(self, key, value):
        for p in parameters:
            if key in parameters[p]:
                return setattr(getattr(self, p), key, value)
        # if hasattr(np, key):
        #     return stats(self, getattr(np, key))
        if hasattr(self, "_beam") and hasattr(self._beam, key):
            return setattr(self._beam, key, value)
        else:
            try:
                return setattr(self, key, value)
            except KeyError:
                raise AttributeError(key)

    def __getattr__(self, key):
        for p in parameters:
            if key in parameters[p]:
                return getattr(getattr(self, p), key)
        # try:
        #     return getattr(self, key)
        # except AttributeError:
        #     return getattr(self._beam, key)

    def __repr__(self):
        return repr(
            {
                "filename": self.filename,
                "code": self.code,
            }
        )

    def set_particle_mass(self, mass: float=constants.m_e) -> None:
        """
        Set the mass of all particles in the distribution by updating
        :attr:`~SimulationFramework.Modules.Beams.beam.particle_mass`.

        Parameters
        ----------
        mass: float
            Particle mass in kg
        """
        self.particle_mass = np.full(len(self.x), mass)

    def normalise_to_ref_particle(self, array, index=0, subtractmean=False) -> np.ndarray:
        """
        Normalise a distribution to the first element in the array (i.e. the ASTRA reference particle)

        Parameters
        ----------
        array: np.ndarray
            The array to normalise
        index: int
            Not in use
        subtractmean: bool
            If true, subtract the reference particle from the array

        Returns
        -------
        np.ndarray
            The normalised array
        """
        array = copy.copy(array)
        array[1:] = array[0] + array[1:]
        if subtractmean:
            array = array - array[0]  # np.mean(array)
        return array

    def reset_dicts(self) -> None:
        """
        Clear out the :attr:`~SimulationFramework.Modules.Beams.Particles.Particles` object,
        removing the distribution from this object.
        """
        self._beam = Particles()

    def read_HDF5_beam_file(self, *args, **kwargs):
        """
        Load in an HDF5-type beam distribution file and update the
        :attr:`~SimulationFramework.Modules.Beams.beam.Particles` object.
        """
        hdf5.read_HDF5_beam_file(self, *args, **kwargs)

    def read_SDDS_beam_file(self, *args, **kwargs):
        """
        Load in an SDDS-type beam distribution file and update the
        :attr:`~SimulationFramework.Modules.Beams.beam.Particles` object.
        """
        sdds.read_SDDS_beam_file(self, *args, **kwargs)

    def read_gdf_beam_file(self, *args, **kwargs):
        """
        Load in a GDF-type beam distribution file and update the
        :attr:`~SimulationFramework.Modules.Beams.beam.Particles` object.
        """
        gdf.read_gdf_beam_file(self, *args, **kwargs)

    def read_astra_beam_file(self, *args, **kwargs):
        """
        Load in an ASTRA-type beam distribution file and update the
        :attr:`~SimulationFramework.Modules.Beams.beam.Particles` object.
        """
        astra.read_astra_beam_file(self, *args, **kwargs)

    def read_ocelot_beam_file(self, *args, **kwargs):
        """
        Load in an OCELOT-type beam distribution file and update the
        :attr:`~SimulationFramework.Modules.Beams.beam.Particles` object.
        """
        from . import ocelot

        ocelot.read_ocelot_beam_file(self, *args, **kwargs)

    def write_HDF5_beam_file(self, *args, **kwargs):
        """
        Write out an HDF5-type beam distribution file.
        """
        hdf5.write_HDF5_beam_file(self, *args, **kwargs)

    def write_SDDS_beam_file(self, *args, **kwargs):
        """
        Write out an SDDS-type beam distribution file.
        """
        sdds.write_SDDS_file(self, *args, **kwargs)

    def write_gdf_beam_file(self, *args, **kwargs):
        """
        Write out a GDF-type beam distribution file.
        """
        gdf.write_gdf_beam_file(self, *args, **kwargs)

    def write_astra_beam_file(self, *args, **kwargs):
        """
        Write out an ASTRA-type beam distribution file.
        """
        astra.write_astra_beam_file(self, *args, **kwargs)

    def write_ocelot_beam_file(self, *args, **kwargs):
        """
        Write out an OCELOT-type beam distribution file.
        """
        from . import ocelot

        return ocelot.write_ocelot_beam_file(self, *args, **kwargs)

    def write_mad8_beam_file(self, *args, **kwargs):
        """
        Write out a MAD8-type beam distribution file.
        """
        mad8.write_mad8_beam_file(self, *args, **kwargs)

    def read_beam_file(self, filename, run_extension="001"):
        """
        Load in a beam distribution file and update the
        :attr:`~SimulationFramework.Modules.Beams.beam.Particles` object.

        Based on the extension in `filename`, the appropriate function will be called.

        Parameters
        ----------
        filename: str
            The name of the file to be loaded
        run_extension: str
            Run extension for ASTRA-type beam distribution files.
        """
        pre, ext = os.path.splitext(os.path.basename(filename))
        if ext.lower()[:4] == ".hdf":
            hdf5.read_HDF5_beam_file(self, filename)
        elif ext.lower() == ".sdds":
            sdds.read_SDDS_beam_file(self, filename)
        elif ext.lower() == ".gdf":
            gdf.read_gdf_beam_file(self, filename)
        elif (ext.lower() == ".npz") and (".ocelot" in filename):
            from . import ocelot

            ocelot.read_ocelot_beam_file(self, filename)
        elif ext.lower() == ".astra":
            astra.read_astra_beam_file(self, filename)
        elif re.match(r".*.\d\d\d\d." + run_extension, filename):
            astra.read_astra_beam_file(self, filename)
        else:
            try:
                with open(filename, "r") as f:
                    firstline = f.readline()
                    if "SDDS" in firstline:
                        sdds.read_SDDS_beam_file(self, filename)
            except UnicodeDecodeError:
                if gdf.rgf.is_gdf_file(filename):
                    gdf.read_gdf_beam_file(self, filename)
                else:
                    warn("Could not load file")

    if use_matplotlib:

        def plot(self, **kwargs):
            return plot.plot(self, **kwargs)

        def slice_plot(self, *args, **kwargs):
            return plot.slice_plot(self, *args, **kwargs)

        def plotScreenImage(self, **kwargs):
            return plot.plotScreenImage(self, **kwargs)

    def resample(self, npart, **kwargs) -> beam:
        """
        Resample the beam using a kernel density estimator, updating the number of particles.
        See :class:`~SimulationFramework.Modules.Beams.Particles.kde.kde`.

        Parameters
        ----------
        npart: int
            Number of particles for the new distribution

        Returns
        -------
        :class:`~SimulationFramework.Modules.Beams.beam`
            The resampled beam.

        """
        postbeam = self.kde.resample(npart, **kwargs)
        newbeam = beam()
        newbeam.x = postbeam[0]
        newbeam.y = postbeam[1]
        newbeam.z = postbeam[2]
        newbeam.px = postbeam[3]
        newbeam.py = postbeam[4]
        newbeam.pz = postbeam[5]
        newbeam.t = newbeam.z / (-1 * newbeam.Bz * constants.speed_of_light)
        newbeam.total_charge = self.total_charge
        single_charge = newbeam.total_charge / (len(newbeam.x))
        newbeam.charge = np.full(len(newbeam.x), single_charge)
        newbeam.nmacro = np.full(len(newbeam.x), 1)
        newbeam.code = "KDE"
        newbeam.longitudinal_reference = "z"

        return newbeam

    def rotate_beamXZ(self, theta, preOffset=[0, 0, 0], postOffset=[0, 0, 0]):
        hdf5.rotate_beamXZ(self, theta, preOffset=preOffset, postOffset=postOffset)

    def unrotate_beamXZ(self):
        hdf5.unrotate_beamXZ(self)


def load_directory(directory=".", types={"SimFrame": ".hdf5"}, verbose=False) -> beamGroup:
    """
    Load in all beam distribution files from a directory and create a
    :class:`~SimulationFramework.Modules.Beams.beamGroup` object.

    Parameters
    ----------
    directory: str
        Directory from which to load the files
    types: Dict
        Beam distribution file types to load
    verbose: bool
        If true, print progress

    Returns
    -------
    :class:`~SimulationFramework.Modules.Beams.beamGroup`
        A new `beamGroup`.
    """
    bg = beamGroup()
    if verbose:
        print("Directory:", directory)
    for code, string in types.items():
        beam_files = glob.glob(directory + "/*" + string)
        if verbose:
            print(code, [os.path.basename(t) for t in beam_files])
        bg.add(beam_files)
        bg.sort()
    return bg


def load_file(filename, *args, **kwargs) -> beam:
    """
    Load in a beam distribution files and create a
    :class:`~SimulationFramework.Modules.Beams.beam` object.

    Parameters
    ----------
    filename: str
        Name of file to load

    Returns
    -------
    :class:`~SimulationFramework.Modules.Beams.beam`
        A new `beam`.
    """
    b = beam()
    b.read_beam_file(filename)
    return b


def save_HDF5_summary_file(directory=".", filename="./Beam_Summary.hdf5", files=None) -> None:
    if not files:
        beam_files = glob.glob(directory + "/*.hdf5")
        files = []
        for bf in beam_files:
            with h5py.File(bf, "r") as f:
                if "/beam/beam" in f:
                    files.append(bf)
    hdf5.write_HDF5_summary_file(filename, files)


def load_HDF5_summary_file(filename):
    dir = os.path.dirname(filename)
    bg = beamGroup()
    with h5py.File(filename, "r") as f:
        for screen in list(f.keys()):
            bg.add(os.path.join(dir, screen + ".hdf5"))
    bg.sort()
    return bg
