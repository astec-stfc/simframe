import os
import sys
import munch
from collections import OrderedDict
import numpy as np
import re
import copy
import glob
import scipy.constants as constants
from .Particles import Particles
from . import astra
from . import sdds
from . import vsim
from . import gdf
from . import hdf5
from . import plot

# I can't think of a clever way of doing this, so...
parameters = {
'data': ['x', 'y',  'z', 't', 'px', 'cpx', 'py', 'cpy', 'pz', 'cpz', 'p', ],
'emittance': ['ex', 'ey', 'enx', 'eny',],
'twiss': ['alpha_x', 'beta_x', 'gamma_x', 'alpha_y', 'beta_y', 'gamma_y', 'eta_x', 'eta_xp', ],
'stats': ['sigma_x', 'sigma_y', 'sigma_z', 'sigma_t', ],
'slice': ['slice_ex', 'slice_ey', 'slice_enx', 'slice_eny', 'slice_current', 'slice_t', 'slice_normalized_horizontal_emittance', 'slice_normalized_vertical_emittance',
          'slice_relative_momentum_spread', 'slice_beta_x', 'slice_beta_y',],
}

class particlesGroup(munch.Munch):

    def __init__(self, particles):
        self.particles = particles

    def __getitem__(self, key):
        if key == 'particles':
            return super(particlesGroup, self).__getitem__(key)
        else:
            return [getattr(p, key) for p in self.particles]

class beamGroup(munch.Munch):

    def __repr__(self):
        return repr(list(self.beams.keys()))

    def __len__(self):
        return len(super(beamGroup, self).__getitem__('beams'))

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.beams.values())[key]
        for p in parameters:
            if key in parameters[p]:
                return getattr(super(beamGroup, self).__getattr__(p), key)
        else:
            return super(beamGroup, self).__getitem__(key)

    def __init__(self, filenames=[]):
        self.sddsindex = 0
        self.beams = OrderedDict()
        if isinstance(filenames, (str)):
            filenames = [filenames]
        for f in filenames:
            self.add(f)

    @property
    def data(self):
        return particlesGroup([b._beam for b in self.beams.values()])
    @property
    def stats(self):
        return particlesGroup([b._beam.stats for b in self.beams.values()])
    @property
    def twiss(self):
        return particlesGroup([b._beam.twiss for b in self.beams.values()])
    @property
    def slice(self):
        return particlesGroup([b._beam.slice for b in self.beams.values()])
    @property
    def emittance(self):
        return particlesGroup([b._beam.emittance for b in self.beams.values()])

    def sort(self, key='t', function='mean', *args, **kwargs):
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
                self.beams[file] = beam(file)

    def param(self, param):
        return [getattr(b._beam, param) for b in self.beams.values()]

class beam(munch.Munch):

    particle_mass = constants.m_e
    E0 = particle_mass * constants.speed_of_light**2
    E0_eV = E0 / constants.elementary_charge
    q_over_c = (constants.elementary_charge / constants.speed_of_light)
    speed_of_light = constants.speed_of_light

    def __init__(self, filename=None, sddsindex=0):
        self._beam = Particles()
        self.sddsindex = sddsindex
        self.filename = ''
        self.code = None
        if filename is not None:
            self.read_beam_file(filename)

    @property
    def Particles(self):
        return self._beam
    @property
    def data(self):
        return self._beam
    @property
    def stats(self):
        return self._beam.stats
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
                return getattr(super(beam, self).__getattr__(p), key)
        if hasattr(super(beam, self).__getitem__('_beam'),key):
            return getattr(super(beam, self).__getitem__('_beam'),key)
        else:
            return super(beam, self).__getitem__(key)

    def __repr__(self):
        return repr({'filename': self.filename, 'code': self.code, 'beam': [k for k in self._beam.keys() if isinstance(self._beam[k], np.ndarray) and self._beam[k].size > 0]})

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

    def plot(self, keys=None, **kwargs):
        if keys is not None:
            kwargs['key1'] = keys[0]
            if len(keys) > 1:
                kwargs['key2'] = keys[1]
            else:
                kwargs['key2'] = None
        plot.plot(self, **kwargs)

    def slice_plot(self, *args, **kwargs):
        plot.slice_plot(self, *args, **kwargs)

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
