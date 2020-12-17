import os
import math
import warnings
import numpy as np
try:
    from scipy import interpolate
    use_interpolate = True
except ImportError:
    use_interpolate = False
from .. import constants
import munch
import glob
from . import hdf5
from . import gpt
from . import astra
from . import elegant
try:
    from . import plot
    use_matplotlib = True
except ImportError:
    use_matplotlib = False

class twissData(np.ndarray):

    def __new__(cls=np.ndarray, input_array=[], units=''):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.units = units
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.units = getattr(obj, 'units', '')

class twiss(munch.Munch):

    properties = {
    'z': 'm',
    't': 's',
    'kinetic_energy': 'J',
    'gamma': '',
    'cp': 'eV/c',
    'cp_eV': 'eV/c',
    'p': 'kg*m/s',
    'enx': 'm-radians',
    'ex': 'm-radians',
    'eny': 'm-radians',
    'ey': 'm-radians',
    'enz': 'eV*s',
    'ez': 'eV*s',
    'beta_x': 'm',
    'gamma_x': '',
    'alpha_x': '',
    'beta_y': 'm',
    'gamma_y': '',
    'alpha_y': '',
    'beta_z': 'm',
    'gamma_z': '',
    'alpha_z': '',
    'sigma_x': 'm',
    'sigma_y': 'm',
    'sigma_z': 'm',
    'sigma_t': 's',
    'sigma_p': 'kg * m/s',
    'sigma_cp': 'eV/c',
    'sigma_cp_eV': 'eV/c',
    'mux': '2 pi',
    'muy': '2 pi',
    'eta_x': 'm',
    'eta_xp': 'mrad',
    'element_name': '',
    'ecnx': 'm-mrad',
    'ecny': 'm-mrad',
    'eta_x_beam': 'm',
    'eta_xp_beam': 'radians',
    'eta_y_beam': 'm',
    'eta_yp_beam': 'radians',
    'beta_x_beam': 'm',
    'beta_y_beam': 'm',
    'alpha_x_beam': '',
    'alpha_y_beam': '',
    }

    E0 = constants.m_e * constants.speed_of_light**2
    E0_eV = E0 / constants.elementary_charge
    q_over_c = (constants.e / constants.speed_of_light)

    def __init__(self):
        super(twiss, self).__init__()
        self.reset_dicts()
        self.sddsindex = 0
        self.codes = {
            'elegant': elegant.read_elegant_twiss_files,
            'gpt': gpt.read_gdf_twiss_files,
            'astra': astra.read_astra_twiss_files
        }
        self.code_signatures = [['elegant','.twi'], ['elegant','.flr'], ['elegant','.sig'], ['GPT', 'emit.gdf'], ['astra', 'Xemit.001']]

    # def __getitem__(self, key):
    #     if key in super(twiss, self).__getitem__('data') and super(twiss, self).__getitem__('data') is not None:
    #         return self.get(key)
    #     else:
    #         return super(twiss, self).__getitem__(key)

    def read_astra_twiss_files(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return astra.read_astra_twiss_files(self, *args, **kwargs)

    def read_elegant_twiss_files(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return elegant.read_elegant_twiss_files(self, *args, **kwargs)

    def read_gdf_twiss_files(self, *args, **kwargs):
        return self.read_GPT_twiss_files(*args, **kwargs)
        
    def read_GPT_twiss_files(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return gpt.read_gdf_twiss_files(self, *args, **kwargs)

    def save_HDF5_twiss_file(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return hdf5.write_HDF5_twiss_file(self, *args, **kwargs)

    def __repr__(self):
        return repr([k for k in self.properties if len(self[k]) > 0])

    def stat(self, key):
        if key in self.properties:
            return self[key]

    def units(self, key):
        if key in self.properties:
            return self.properties[key]

    def find_nearest(self, array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return idx-1
        else:
            return idx

    def reset_dicts(self):
        self.sddsindex = 0
        for k, v in self.properties.items():
            self[k] = twissData(units=v)
        self.elegantTwiss = {}

    def sort(self, key='z', reverse=False):
        index = self[key].argsort()
        for k in self.properties:
            if len(self[k]) > 0:
                if reverse:
                    self[k] = self[k][index[::-1]]
                else:
                    self[k] = self[k][index[::1]]

    def append(self, array, data):
        self[array] = twissData(np.concatenate([self[array], data]), units=self[array].units)

    def _which_code(self, name):
        if name.lower() in self.codes.keys():
            return self.codes[name.lower()]
        return None

    def _determine_code(self, filename):
        for k,v in self.code_signatures:
            l = -len(v)
            if v == filename[l:]:
                return self.codes[k]
        return None

    def interpolate(self, z=None, value='z', index='z'):
        if z is None:
            return np.interp(z, self[index], self[value])
        else:
            if z > max(self[index]):
                return 10**6
            else:
                return float(np.interp(z, self[index], self[value]))

    def extract_values(self, array, start, end):
        startidx = self.find_nearest(self['z'], start)
        endidx = self.find_nearest(self['z'], end) + 1
        return self[array][startidx:endidx]

    def get_parameter_at_z(self, param, z):
        if z in self['z']:
            idx = list(self['z']).index(z)
            return self[param][idx]
        else:
            return self.interpolate(z=z, value=param, index='z')

    if use_matplotlib:
        def plot(self, *args, **kwargs):
            plot.plot(self, *args, **kwargs)

    def covariance(self, u, up):
        u2 = u - np.mean(u)
        up2 = up - np.mean(up)
        return np.mean(u2*up2) - np.mean(u2)*np.mean(up2)

    # @property
    # def cp_eV(self):
    #     return self['cp']
    # @property
    # def cp_MeV(self):
    #     return self['cp'] / 1e6

def load_directory(directory='.', types={'elegant':'.twi', 'GPT': 'emit.gdf','ASTRA': 'Xemit.001'}, preglob='*', verbose=False, sortkey='z'):
    t = twiss()
    if verbose:
        print('Directory:',directory)
    for code, string in types.items():
        twiss_files = glob.glob(directory+'/'+preglob+string)
        if verbose:
            print(code, [os.path.basename(t) for t in twiss_files])
        if t._which_code(code) is not None and len(twiss_files) > 0:
            t._which_code(code)(t, twiss_files, reset=False)
    t.sort(key=sortkey)
    return t

def load_file(filename, *args, **kwargs):
    twissobject = twiss()
    code = twissobject._determine_code(filename)
    if code is not None:
        code(twissobject, filename, reset=False)
    return twissobject
