import os
import munch
from pydantic import BaseModel
import numpy as np
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
    return [f for f in dir(obj) if type(getattr(obj, f)) is property]


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


class particlesGroup(munch.Munch):

    def __init__(self, particles):
        self.particles = particles

    def __getitem__(self, key):
        if key == "particles":
            return super(particlesGroup, self).__getitem__(key)
        else:
            data = [getattr(p, key) for p in self.particles]
            return UnitValue(data, units=data[0].units)


class statsGroup(object):

    def __init__(self, beam, function):
        self._beam = beam
        self._func = function

    def __getattr__(self, key):
        var = self._beam.__getitem__(key)
        return UnitValue([self._func(v) for v in var], units="m")
        # return np.sqrt(self._beam.covariance(var, var))


class beamGroup(munch.Munch):

    def __repr__(self):
        return repr(list(self.beams.keys()))

    def __len__(self):
        return len(super(beamGroup, self).__getitem__("beams"))

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.beams.values())[key]
        elif isinstance(key, slice):
            return beamGroup(beams=list(self.beams.items())[key])
        elif hasattr(np, key):
            return statsGroup(self, getattr(np, key))
        for p in parameters:
            if key in parameters[p]:
                return getattr(super(beamGroup, self).__getattr__(p), key)
        else:
            return super(beamGroup, self).__getitem__(key)

    def __init__(self, filenames=[], beams=[]):
        self.sddsindex = 0
        self.beams = dict()
        self._parameters = parameters
        for k, v in beams:
            self.beams[k] = v
        if isinstance(filenames, (str)):
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
                self.beams.items(), key=lambda item: func(item[1][key]), *args, **kwargs
            )
        )
        return self

    def add(self, filename):
        if isinstance(filename, (str)):
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

    # particle_mass = UnitValue(constants.m_e, "kg")
    # E0 = UnitValue(particle_mass * constants.speed_of_light**2, "J")
    # E0_eV = UnitValue(E0 / constants.elementary_charge, "eV/c")
    q_over_c: UnitValue = UnitValue(constants.elementary_charge / constants.speed_of_light, "C/c")
    speed_of_light: UnitValue = UnitValue(constants.speed_of_light, "m/s")
    filename: str = None
    sddsindex: int = 0
    code: str = None
    reference_particle: np.ndarray = None
    longitudinal_reference: np.ndarray | str = None
    starting_position: list | np.ndarray = [0, 0, 0]
    theta: float = 0
    offset: list | np.ndarray = [0, 0, 0]
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

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

    def model_dump(self, *args, **kwargs):
        # Only include computed fields
        full_dump = super().model_dump(*args, **kwargs)
        full_dump.update({"Particles": self._beam.model_dump()})
        return full_dump

    @property
    def E0_eV(self):
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
    def beam(self):
        return self._beam

    @property
    def Particles(self):
        return self._beam

    @property
    def data(self):
        return self._beam

    @property
    def sigmas(self):
        return self._beam.sigmas

    @property
    def centroids(self):
        return self._beam.centroids

    @property
    def twiss(self):
        return self._beam.twiss

    @property
    def slice(self):
        return self._beam.slice

    @property
    def emittance(self):
        return self._beam.emittance

    @property
    def kde(self):
        return self._beam.kde

    @property
    def mve(self):
        return self._beam.mve

    def rms(self, x, axis=None):
        return np.sqrt(np.mean(x**2, axis=axis))

    def __len__(self):
        return len(self._beam.x)

    def __getitem__(self, key):
        # print('beams key', key)
        for p in parameters:
            if key in parameters[p]:
                return getattr(super().__getattr__(p), key)
        if hasattr(np, key):
            return stats(self, getattr(np, key))
        if hasattr(super().__getitem__("_beam"), key):
            return getattr(super().__getitem__("_beam"), key)
        else:
            try:
                return super(beam, self).__getitem__(key)
            except KeyError:
                raise AttributeError(key)

    def __setitem__(self, key, value):
        for p in parameters:
            if key in parameters[p]:
                return super().__getattr__(p).__setitem__(key, value)
        # if hasattr(np, key):
        #     return stats(self, getattr(np, key))
        if hasattr(self, "_beam") and hasattr(
            super(beam, self).__getitem__("_beam"), key
        ):
            return setattr(super(beam, self).__getitem__("_beam"), key, value)
        else:
            try:
                return super(beam, self).__setitem__(key, value)
            except KeyError:
                raise AttributeError(key)

    def __repr__(self):
        return repr(
            {
                "filename": self.filename,
                "code": self.code,
                "Particles": [
                    k
                    for k in self._beam.keys()
                    if isinstance(self._beam[k], np.ndarray) and self._beam[k].size > 0
                ],
            }
        )

    def set_particle_mass(self, mass=constants.m_e):
        self.particle_mass = np.full(len(self.x), mass)

    def normalise_to_ref_particle(self, array, index=0, subtractmean=False):
        array = copy.copy(array)
        array[1:] = array[0] + array[1:]
        if subtractmean:
            array = array - array[0]  # np.mean(array)
        return array

    def reset_dicts(self):
        self._beam = Particles()

    def read_HDF5_beam_file(self, *args, **kwargs):
        hdf5.read_HDF5_beam_file(self, *args, **kwargs)

    def read_SDDS_beam_file(self, *args, **kwargs):
        sdds.read_SDDS_beam_file(self, *args, **kwargs)

    def read_gdf_beam_file(self, *args, **kwargs):
        gdf.read_gdf_beam_file(self, *args, **kwargs)

    def read_astra_beam_file(self, *args, **kwargs):
        astra.read_astra_beam_file(self, *args, **kwargs)

    def read_ocelot_beam_file(self, *args, **kwargs):
        from . import ocelot

        ocelot.read_ocelot_beam_file(self, *args, **kwargs)

    def write_HDF5_beam_file(self, *args, **kwargs):
        hdf5.write_HDF5_beam_file(self, *args, **kwargs)

    def write_SDDS_beam_file(self, *args, **kwargs):
        sdds.write_SDDS_file(self, *args, **kwargs)

    def write_gdf_beam_file(self, *args, **kwargs):
        gdf.write_gdf_beam_file(self, *args, **kwargs)

    def write_astra_beam_file(self, *args, **kwargs):
        astra.write_astra_beam_file(self, *args, **kwargs)

    def write_ocelot_beam_file(self, *args, **kwargs):
        from . import ocelot

        return ocelot.write_ocelot_beam_file(self, *args, **kwargs)

    def write_mad8_beam_file(self, *args, **kwargs):
        mad8.write_mad8_beam_file(self, *args, **kwargs)

    def read_beam_file(self, filename, run_extension="001"):
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
                    return None

    if use_matplotlib:

        def plot(self, **kwargs):
            return plot.plot(self, **kwargs)

        def slice_plot(self, *args, **kwargs):
            return plot.slice_plot(self, *args, **kwargs)

        def plotScreenImage(self, **kwargs):
            return plot.plotScreenImage(self, **kwargs)

    def resample(self, npart, **kwargs):
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


def load_directory(directory=".", types={"SimFrame": ".hdf5"}, verbose=False):
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


def load_file(filename, *args, **kwargs):
    b = beam()
    b.read_beam_file(filename)
    return b


def save_HDF5_summary_file(directory=".", filename="./Beam_Summary.hdf5", files=None):
    if not files:
        beam_files = glob.glob(directory + "/*.hdf5")
        files = []
        for bf in beam_files:
            with h5py.File(bf, "a") as f:
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
