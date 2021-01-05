import os
import munch
from collections import OrderedDict
import numpy as np
import re
import copy
import glob
import h5py
from .. import constants
from .Particles import Particles
from . import astra
from . import sdds
from . import vsim
from . import gdf
from . import hdf5
from . import mad8
try:
    from . import plot
    use_matplotlib = True
except ImportError as e:
    print('Import error - plotting disabled. Missing package:', e)
    use_matplotlib = False

from .Particles.emittance import emittance as emittanceobject
from .Particles.twiss import twiss as twissobject
from .Particles.slice import slice as sliceobject
from .Particles.sigmas import sigmas as sigmasobject

# I can't think of a clever way of doing this, so...
def get_properties(obj): return [f for f in dir(obj) if type(getattr(obj, f)) is property]
parameters = {
'data': get_properties(Particles),
'emittance': get_properties(emittanceobject),
'twiss': get_properties(twissobject),
'slice': get_properties(sliceobject),
'sigmas': get_properties(sigmasobject),
}

class particlesGroup(munch.Munch):

    def __init__(self, particles):
        self.particles = particles

    def __getitem__(self, key):
        if key == 'particles':
            return super(particlesGroup, self).__getitem__(key)
        else:
            return [getattr(p, key) for p in self.particles]

class statsGroup(object):

    def __init__(self, beam, function):
        self._beam = beam
        self._func = function

    def __getattr__(self, key):
        var = self._beam.__getitem__(key)
        return np.array([self._func(v) for v in var])
        # return np.sqrt(self._beam.covariance(var, var))

class beamGroup(munch.Munch):

    def __repr__(self):
        return repr(list(self.beams.keys()))

    def __len__(self):
        return len(super(beamGroup, self).__getitem__('beams'))

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
        self.beams = OrderedDict()
        self._parameters = parameters
        for k,v in beams:
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
    def twiss(self):
        return particlesGroup([b._beam.twiss for b in self.beams.values()])
    @property
    def slice(self):
        return particlesGroup([b._beam.slice for b in self.beams.values()])
    @property
    def emittance(self):
        return particlesGroup([b._beam.emittance for b in self.beams.values()])

    def sort(self, key='z', function='mean', *args, **kwargs):
        if isinstance(function,str) and hasattr(np, function):
            func = getattr(np, function)
        else:
            func = function
        self.beams = OrderedDict(sorted(self.beams.items(), key=lambda item: func(item[1][key]), *args, **kwargs))
        return self

    def add(self, filename):
        if isinstance(filename, (str)):
            filename = [filename]
        for file in filename:
            if os.path.isdir(file):
                self.add_directory(file)
            elif os.path.isfile(file):
                file = file.replace('\\','/')
                try:
                    self.beams[file] = beam(file)
                except:
                    if file in self.beams:
                        del self.beams[file]

    def param(self, param):
        return [getattr(b._beam, param) for b in self.beams.values()]

    def getScreen(self, screen):
        for b in self.beams:
            if screen in b:
                return self.beams[b]
        return None

class stats(object):

    def __init__(self, beam, function):
        self._beam = beam
        self._func = function

    def __getattr__(self, key):
        var = self._beam.__getitem__(key)
        return self._func(var)
        # return np.sqrt(self._beam.covariance(var, var))

class beam(munch.Munch):

    particle_mass = constants.m_e
    E0 = particle_mass * constants.speed_of_light**2
    E0_eV = E0 / constants.elementary_charge
    q_over_c = (constants.elementary_charge / constants.speed_of_light)
    speed_of_light = constants.speed_of_light

    def __init__(self, filename=None, sddsindex=0):
        self._beam = Particles()
        self._parameters = parameters
        # self.sigma = stats(self, lambda var:  np.sqrt(self._beam.covariance(var, var)))
        # self.mean = stats(self, np.mean)
        self.sddsindex = sddsindex
        self.filename = ''
        self.code = None
        if filename is not None:
            self.read_beam_file(filename)

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
    def twiss(self):
        return self._beam.twiss
    @property
    def slice(self):
        return self._beam.slice
    @property
    def emittance(self):
        return self._beam.emittance

    def __len__(self):
        return len(self._beam.x)

    def __getitem__(self, key):
        for p in parameters:
            if key in parameters[p]:
                return getattr(super().__getattr__(p), key)
        if hasattr(np, key):
            return stats(self, getattr(np, key))
        if hasattr(super().__getitem__('_beam'),key):
            return getattr(super().__getitem__('_beam'),key)
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
        if hasattr(self, '_beam') and hasattr(super(beam, self).__getitem__('_beam'),key):
            return setattr(super(beam, self).__getitem__('_beam'),key,value)
        else:
            try:
                return super(beam, self).__setitem__(key, value)
            except KeyError:
                raise AttributeError(key)

    def __repr__(self):
        return repr({'filename': self.filename, 'code': self.code, 'Particles': [k for k in self._beam.keys() if isinstance(self._beam[k], np.ndarray) and self._beam[k].size > 0]})

    def set_particle_mass(self, mass=constants.m_e):
        self.particle_mass = mass

    def normalise_to_ref_particle(self, array, index=0,subtractmean=False):
        array = copy.copy(array)
        array[1:] = array[0] + array[1:]
        if subtractmean:
            array = array - array[0]#np.mean(array)
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

    def write_HDF5_beam_file(self, *args, **kwargs):
        hdf5.write_HDF5_beam_file(self, *args, **kwargs)
    def write_SDDS_beam_file(self, *args, **kwargs):
        sdds.write_SDDS_file(self, *args, **kwargs)
    def write_gdf_beam_file(self, *args, **kwargs):
        gdf.write_gdf_beam_file(self, *args, **kwargs)
    def write_astra_beam_file(self, *args, **kwargs):
        astra.write_astra_beam_file(self, *args, **kwargs)
    def write_mad8_beam_file(self, *args, **kwargs):
        mad8.write_mad8_beam_file(self, *args, **kwargs)

    def read_beam_file(self, filename, run_extension='001'):
        pre, ext = os.path.splitext(os.path.basename(filename))
        if ext.lower()[:4] == '.hdf':
            hdf5.read_HDF5_beam_file(self, filename)
        elif ext.lower() == '.sdds':
            sdds.read_SDDS_beam_file(self, filename)
        elif ext.lower() == '.gdf':
            gdf.read_gdf_beam_file(self, filename)
        elif ext.lower() == '.astra':
            astra.read_astra_beam_file(self, filename)
        elif re.match('.*.\d\d\d\d.'+run_extension, filename):
            astra.read_astra_beam_file(self, filename)
        else:
            try:
                with open(filename, 'r') as f:
                    firstline = f.readline()
                    if 'SDDS' in firstline:
                        sdds.read_SDDS_beam_file(self, filename)
            except UnicodeDecodeError:
                if gdf.rgf.is_gdf_file(filename):
                        gdf.read_gdf_beam_file(self, filename)
                else:
                    return None

    if use_matplotlib:
        def plot(self, **kwargs):
            plot.plot(self, **kwargs)

        def slice_plot(self, *args, **kwargs):
            plot.slice_plot(self, *args, **kwargs)

        def plotScreenImage(self, **kwargs):
            plot.plotScreenImage(self, **kwargs)

def load_directory(directory='.', types={'SimFrame':'.hdf5'}, verbose=False):
    bg = beamGroup()
    if verbose:
        print('Directory:',directory)
    for code, string in types.items():
        beam_files = glob.glob(directory+'/*'+string)
        if verbose:
            print(code, [os.path.basename(t) for t in beam_files])
        bg.add(beam_files)
        bg.sort()
    return bg

def load_file(filename, *args, **kwargs):
    b = beam()
    b.read_beam_file(filename)
    return b

def save_HDF5_summary_file(directory='.', filename='./Beam_Summary.hdf5', files=None):
    if not files:
        beam_files = glob.glob(directory+'/*.hdf5')
        files = []
        for bf in beam_files:
            with h5py.File(bf, "a") as f:
                if "/beam/beam" in f:
                    files.append(bf)
    hdf5.write_HDF5_summary_file(filename, files)
