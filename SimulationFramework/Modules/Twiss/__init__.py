import os
import math
import warnings
from pydantic import BaseModel, ConfigDict, field_validator, ValidationInfo, Field
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
from . import ocelot

try:
    from . import plot

    use_matplotlib = True
except ImportError:
    use_matplotlib = False

from ..units import UnitValue


class twissParameter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    unit: str
    label: str = Field(default=None, validate_default=True)
    dtype: str = "f"

    @field_validator("label", mode="before")
    @classmethod
    def default_label(cls, v: str, info: ValidationInfo):
        if v is None:
            return info.data["name"]
        return v


class twiss(munch.Munch):

    properties = {
        "z": twissParameter(name="z", unit="m"),
        "t": twissParameter(name="t", unit="s"),
        "kinetic_energy": twissParameter(name="kinetic_energy", unit="eV"),
        "gamma": twissParameter(name="gamma", unit=""),
        "mean_gamma": twissParameter(name="mean_gamma", unit="", label=r"|gamma|"),
        "cp": twissParameter(name="cp", unit="eV/c"),
        "mean_cp": twissParameter(name="mean_cp", unit="eV/c", label=r"|cp|"),
        "cp_eV": twissParameter(name="cp_eV", unit="eV/c"),
        "p": twissParameter(name="p", unit="kg*m/s"),
        "enx": twissParameter(name="enx", unit="m-radians"),
        "ex": twissParameter(name="ex", unit="m-radians"),
        "eny": twissParameter(name="eny", unit="m-radians"),
        "ey": twissParameter(name="ey", unit="m-radians"),
        "enz": twissParameter(name="enz", unit="eV*s"),
        "ez": twissParameter(name="ez", unit="eV*s"),
        "beta_x": twissParameter(name="beta_x", unit="m"),
        "gamma_x": twissParameter(name="gamma_x", unit=""),
        "alpha_x": twissParameter(name="alpha_x", unit=""),
        "beta_y": twissParameter(name="beta_y", unit="m"),
        "gamma_y": twissParameter(name="gamma_y", unit=""),
        "alpha_y": twissParameter(name="alpha_y", unit=""),
        "beta_z": twissParameter(name="beta_z", unit="m"),
        "gamma_z": twissParameter(name="gamma_z", unit=""),
        "alpha_z": twissParameter(name="alpha_z", unit=""),
        "sigma_x": twissParameter(name="sigma_x", unit="m"),
        "sigma_xp": twissParameter(name="sigma_xp", unit="m"),        
        "sigma_y": twissParameter(name="sigma_y", unit="m"),
        "sigma_yp": twissParameter(name="sigma_yp", unit="m"),
        "sigma_z": twissParameter(name="sigma_z", unit="m"),
        "sigma_t": twissParameter(name="sigma_t", unit="s"),
        "sigma_p": twissParameter(name="sigma_p", unit="kg * m/s"),
        "sigma_cp": twissParameter(name="sigma_cp", unit="eV/c"),
        "sigma_cp_eV": twissParameter(name="sigma_cp_eV", unit="eV/c"),
        "mean_x": twissParameter(name="mean_x", unit="m"),
        "mean_y": twissParameter(name="mean_y", unit="m"),
        "mux": twissParameter(name="mux", unit="2 pi"),
        "muy": twissParameter(name="muy", unit="2 pi"),
        "eta_x": twissParameter(name="eta_x", unit="m"),
        "eta_xp": twissParameter(name="eta_xp", unit="mrad"),
        "element_name": twissParameter(name="element_name", unit="", dtype="U"),
        "ecnx": twissParameter(name="ecnx", unit="m-mrad"),
        "ecny": twissParameter(name="ecny", unit="m-mrad"),
        "eta_x_beam": twissParameter(name="eta_x_beam", unit="m"),
        "eta_xp_beam": twissParameter(name="eta_xp_beam", unit="radians"),
        "eta_y_beam": twissParameter(name="eta_y_beam", unit="m"),
        "eta_yp_beam": twissParameter(name="eta_yp_beam", unit="radians"),
        "beta_x_beam": twissParameter(name="beta_x_beam", unit="m"),
        "beta_y_beam": twissParameter(name="beta_y_beam", unit="m"),
        "alpha_x_beam": twissParameter(name="alpha_x_beam", unit=""),
        "alpha_y_beam": twissParameter(name="alpha_y_beam", unit=""),
    }

    def __init__(self, rest_mass=None):
        super(twiss, self).__init__()
        if rest_mass is not None:
            self.E0 = rest_mass * constants.speed_of_light**2
            self.E0_eV = self.E0 / constants.elementary_charge
        else:
            self.E0 = constants.m_e * constants.speed_of_light**2
            self.E0_eV = self.E0 / constants.elementary_charge
        self.q_over_c = constants.e / constants.speed_of_light

        self.reset_dicts()
        self.sddsindex = 0
        self.codes = {
            "elegant": elegant.read_elegant_twiss_files,
            "gpt": gpt.read_gdf_twiss_files,
            "astra": astra.read_astra_twiss_files,
            "ocelot": ocelot.read_ocelot_twiss_files,
        }
        self.code_signatures = [
            ["elegant", ".twi"],
            ["elegant", ".flr"],
            ["elegant", ".sig"],
            ["GPT", "emit.gdf"],
            ["astra", "Xemit.001"],
            ["ocelot", "_twiss.npz"],
        ]

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

    def read_ocelot_twiss_files(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ocelot.read_ocelot_twiss_files(self, *args, **kwargs)

    def save_HDF5_twiss_file(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return hdf5.write_HDF5_twiss_file(self, *args, **kwargs)

    def read_HDF5_twiss_file(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return hdf5.read_HDF5_twiss_file(self, *args, **kwargs)

    def __repr__(self):
        return repr(
            {
                k.name: self[k.name].val
                for k in self.properties.values()
                if len(self[k.name]) > 0
            }
        )

    def stat(self, key):
        if key in self.properties:
            return self[key]

    def units(self, key):
        if key in self.properties:
            return self.properties[key]

    def find_nearest(self, array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (
            idx == len(array)
            or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
        ):
            return idx - 1
        else:
            return idx

    def reset_dicts(self):
        self.sddsindex = 0
        for name, prop in self.properties.items():
            self[prop.name] = UnitValue([], units=prop.unit)
        self.elegantTwiss = {}

    def sort(self, key="z", reverse=False):
        index = self[key].argsort()
        for k in [p for p in self.properties]:
            if len(self[k]) > 0:
                if reverse:
                    self[k] = self[k][index[::-1]]
                else:
                    self[k] = self[k][index[::1]]

    def append(self, array, data):
        self[array] = UnitValue(
            np.concatenate([self[array], data]), units=self[array].units
        )

    def _which_code(self, name):
        if name.lower() in self.codes.keys():
            return self.codes[name.lower()]
        return None

    def _determine_code(self, filename):
        for k, v in self.code_signatures:
            l = -len(v)
            if v == filename[l:]:
                return self.codes[k]
        return None

    def interpolate(self, z=None, value="z", index="z"):
        if z is None:
            return np.interp(z, self[index], self[value])
        else:
            if z > max(self[index]):
                return 10**6
            else:
                return float(np.interp(z, self[index], self[value]))

    def extract_values(self, array, start, end):
        startidx = self.find_nearest(self["z"], start)
        endidx = self.find_nearest(self["z"], end) + 1
        return self[array][startidx:endidx]

    def find_nearest(self, array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (
            idx == len(array)
            or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
        ):
            return array[idx - 1]
        else:
            return array[idx]

    def get_parameter_at_z(self, param, z, tol=1e-3):
        if z in self["z"]:
            idx = list(self["z"]).index(z)
            return self[param][idx]
        else:
            nearest_z = self.find_nearest(self["z"], z)
            if abs(nearest_z - z) < tol:
                idx = list(self["z"]).index(nearest_z)
                return self[param][idx]
            else:
                # print('interpolate!', z, self['z'])
                return self.interpolate(z=z, value=param, index="z")

    def get_parameter_at_element(self, param, element_name):
        if element_name in self["element_name"]:
            idx = list(self["element_name"]).index(element_name)
            return self[param][idx]
        return None

    def get_twiss_at_element(self, element_name):
        if element_name in self["element_name"]:
            idx = list(self["element_name"]).index(element_name)
            twissdict = {}
            for param in self.properties.keys():
                try:
                    twissdict[param] = (self[param].val)[idx]
                except:
                    pass
            return twissdict
        return None

    if use_matplotlib:

        def plot(self, *args, **kwargs):
            return plot.plot(self, *args, **kwargs)

    def covariance(self, u, up):
        u2 = u - np.mean(u)
        up2 = up - np.mean(up)
        return np.mean(u2 * up2) - np.mean(u2) * np.mean(up2)

    def read_sdds_file(self, filename, ascii=False):
        sddsobject = munch.Munch()
        sddsobject.sddsindex = 0
        sddsobject.elegantTwiss = munch.Munch()
        elegant.read_sdds_file(sddsobject, filename, ascii)
        return sddsobject.elegantTwiss

    def load_directory(
        self,
        directory=".",
        types={
            "elegant": ".twi",
            "GPT": "emit.gdf",
            "ASTRA": "Xemit.001",
            "ocelot": "_twiss.npz",
        },
        preglob="*",
        verbose=False,
        sortkey="z",
    ):
        if verbose:
            print("Directory:", directory)
        for code, string in types.items():
            twiss_files = glob.glob(directory + "/" + preglob + string)
            if verbose:
                print(code, [os.path.basename(t) for t in twiss_files])
            if self._which_code(code) is not None and len(twiss_files) > 0:
                self._which_code(code)(self, twiss_files, reset=False)
        self.sort(key=sortkey)

    @classmethod
    def initialise_directory(cls, *args, **kwargs):
        t = cls()
        t.load_directory(*args, **kwargs)
        return t

    # @property
    # def cp_eV(self):
    #     return self['cp']
    # @property
    # def cp_MeV(self):
    #     return self['cp'] / 1e6


def load_directory(
    directory=".",
    types={
        "elegant": ".twi",
        "GPT": "emit.gdf",
        "ASTRA": "Xemit.001",
        "ocelot": "_twiss.npz",
    },
    preglob="*",
    verbose=False,
    sortkey="z",
):
    t = twiss()
    if verbose:
        print("Directory:", directory)
    for code, string in types.items():
        twiss_files = glob.glob(directory + "/" + preglob + string)
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
