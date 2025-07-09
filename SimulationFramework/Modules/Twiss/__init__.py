import os
import math
import warnings
from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    ValidationInfo,
    model_validator,
    Field,
)
import numpy as np
from typing import Dict, Literal, List

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

codes = {
    "elegant": elegant.read_elegant_twiss_files,
    "gpt": gpt.read_gdf_twiss_files,
    "astra": astra.read_astra_twiss_files,
    "ocelot": ocelot.read_ocelot_twiss_files,
}

code_signatures = [
    ["elegant", ".twi"],
    ["elegant", ".flr"],
    ["elegant", ".sig"],
    ["GPT", "emit.gdf"],
    ["astra", "Xemit.001"],
    ["ocelot", "_twiss.npz"],
]

class twissParameter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    unit: str
    val: List = []
    label: str = Field(default=None, validate_default=True)
    dtype: str = "f"

    @field_validator("label", mode="before")
    @classmethod
    def default_label(cls, v: str, info: ValidationInfo):
        if v is None:
            return info.data["name"]
        return v


class initialTwiss(BaseModel):
    alpha_x:    float
    beta_x:     float
    alpha_y:    float
    beta_y:     float
    ex:         float
    ey:         float
    enx:        float
    eny:        float
    eta_x:      float
    eta_xp:     float
    eta_y:      float
    eta_yp:     float


class twiss(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    z: twissParameter = twissParameter(name="z", unit="m")
    t: twissParameter = twissParameter(name="t", unit="s")
    kinetic_energy: twissParameter = twissParameter(name="kinetic_energy", unit="eV")
    gamma: twissParameter = twissParameter(name="gamma", unit="")
    mean_gamma: twissParameter = twissParameter(name="mean_gamma", unit="", label=r"|gamma|")
    cp: twissParameter = twissParameter(name="cp", unit="eV/c")
    mean_cp: twissParameter = twissParameter(name="mean_cp", unit="eV/c", label=r"|cp|")
    cp_eV: twissParameter = twissParameter(name="cp_eV", unit="eV/c")
    p: twissParameter = twissParameter(name="p", unit="kg*m/s")
    enx: twissParameter = twissParameter(name="enx", unit="m-radians")
    ex: twissParameter = twissParameter(name="ex", unit="m-radians")
    eny: twissParameter = twissParameter(name="eny", unit="m-radians")
    ey: twissParameter = twissParameter(name="ey", unit="m-radians")
    enz: twissParameter = twissParameter(name="enz", unit="eV*s")
    ez: twissParameter = twissParameter(name="ez", unit="eV*s")
    beta_x: twissParameter = twissParameter(name="beta_x", unit="m")
    gamma_x: twissParameter = twissParameter(name="gamma_x", unit="")
    alpha_x: twissParameter = twissParameter(name="alpha_x", unit="")
    beta_y: twissParameter = twissParameter(name="beta_y", unit="m")
    gamma_y: twissParameter = twissParameter(name="gamma_y", unit="")
    alpha_y: twissParameter = twissParameter(name="alpha_y", unit="")
    beta_z: twissParameter = twissParameter(name="beta_z", unit="m")
    gamma_z: twissParameter = twissParameter(name="gamma_z", unit="")
    alpha_z: twissParameter = twissParameter(name="alpha_z", unit="")
    sigma_x: twissParameter = twissParameter(name="sigma_x", unit="m")
    sigma_xp: twissParameter = twissParameter(name="sigma_xp", unit="m")
    sigma_y: twissParameter = twissParameter(name="sigma_y", unit="m")
    sigma_yp: twissParameter = twissParameter(name="sigma_yp", unit="m")
    sigma_z: twissParameter = twissParameter(name="sigma_z", unit="m")
    sigma_t: twissParameter = twissParameter(name="sigma_t", unit="s")
    sigma_p: twissParameter = twissParameter(name="sigma_p", unit="kg * m/s")
    sigma_cp: twissParameter = twissParameter(name="sigma_cp", unit="eV/c")
    sigma_cp_eV: twissParameter = twissParameter(name="sigma_cp_eV", unit="eV/c")
    mean_x: twissParameter = twissParameter(name="mean_x", unit="m")
    mean_y: twissParameter = twissParameter(name="mean_y", unit="m")
    mux: twissParameter = twissParameter(name="mux", unit="2 pi")
    muy: twissParameter = twissParameter(name="muy", unit="2 pi")
    eta_x: twissParameter = twissParameter(name="eta_x", unit="m")
    eta_xp: twissParameter = twissParameter(name="eta_xp", unit="mrad")
    eta_y: twissParameter = twissParameter(name='eta_y', unit='m')
    eta_yp: twissParameter = twissParameter(name='eta_yp', unit='mrad')
    element_name: twissParameter = twissParameter(name="element_name", unit="", dtype="U")
    lattice_name: twissParameter = twissParameter(name="lattice_name", unit="", dtype="U")
    ecnx: twissParameter = twissParameter(name="ecnx", unit="m-mrad")
    ecny: twissParameter = twissParameter(name="ecny", unit="m-mrad")
    eta_x_beam: twissParameter = twissParameter(name="eta_x_beam", unit="m")
    eta_xp_beam: twissParameter = twissParameter(name="eta_xp_beam", unit="radians")
    eta_y_beam: twissParameter = twissParameter(name="eta_y_beam", unit="m")
    eta_yp_beam: twissParameter = twissParameter(name="eta_yp_beam", unit="radians")
    beta_x_beam: twissParameter = twissParameter(name="beta_x_beam", unit="m")
    beta_y_beam: twissParameter = twissParameter(name="beta_y_beam", unit="m")
    alpha_x_beam: twissParameter = twissParameter(name="alpha_x_beam", unit="")
    alpha_y_beam: twissParameter = twissParameter(name="alpha_y_beam", unit="")
    rest_mass: float | None = None
    codes: Dict = codes
    code_signatures: Literal = code_signatures
    sddsindex: int = 0
    q_over_c: float = constants.e / constants.speed_of_light
    E0: float = constants.m_e * constants.speed_of_light**2
    E0_eV: float = E0 / constants.elementary_charge
    elegantTwiss: Dict = {}
    elegantData: Dict = {}

    def __init__(
            self,
            rest_mass=None,
    ):
        twiss.rest_mass = rest_mass
        super(
            twiss,
            self,
        ).__init__(
            rest_mass=rest_mass
        )

    @model_validator(mode="before")
    def validate_fields(cls, values):
        return values

    def set_E0(self, value):
        self.E0 = value * constants.speed_of_light**2
        self.E0_eV = self.E0 / constants.elementary_charge

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
                k: getattr(self, k)
                for k in self.model_fields_set
            }
        )

    def stat(self, key):
        return getattr(self, key)

    def units(self, key):
        return getattr(self, key)

    def find_nearest_idx(self, array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return idx-1
        else:
            return idx

    def find_nearest(self, array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return array[idx-1]
        else:
            return array[idx]

    def reset_dicts(self):
        self.sddsindex = 0
        for name in self.model_fields_set:
            if isinstance(getattr(self, name), twissParameter):
                setattr(self, name, twissParameter(name=name, unit=getattr(self, name).unit))
        self.elegantTwiss = {}

    def sort(self, key="z", reverse=False):
        flat = np.array(getattr(self, key).val)
        index = flat.argsort()
        for k in [p for p in self.model_fields]:
            if isinstance(getattr(self, k), twissParameter):
                if len(getattr(self, k).val) > 0:
                    try:
                        flat = np.array([x for xs in getattr(self, k).val for x in xs])
                    except Exception:
                        flat = getattr(self, k).val
                    if reverse:
                        getattr(self, k).val = flat[index[::-1]]
                    else:
                        getattr(self, k).val = flat[index[::1]]

    def append(self, array, data):
        newval = UnitValue(
            np.concatenate([getattr(self, array), data]), units=getattr(self, array).units
        )
        setattr(self, array, newval)

    def _which_code(self, name):
        if name.lower() in self.codes.keys():
            return self.codes[name.lower()]
        return None

    def _determine_code(self, filename):
        for k, v in self.code_signatures:
            cutl = -len(v)
            if v == filename[cutl:]:
                return self.codes[k]
        return None

    def interpolate(self, z=None, value="z", index="z"):
        if z is None:
            return np.interp(z, getattr(self, index), getattr(self, value))
        else:
            if z > max(self[index]):
                return 10**6
            else:
                return float(np.interp(z, self[index], self[value]))

    def extract_values(self, array, start, end):
        startidx = self.find_nearest_idx(getattr(self, 'z'), start)
        endidx = self.find_nearest_idx(getattr(self, 'z'), end) + 1
        return getattr(self, array)[startidx:endidx]

    def get_parameter_at_z(self, param, z, tol=1e-3):
        if z in self.z:
            idx = list(self.z).index(z)
            return getattr(self, param)[idx]
        else:
            nearest_z = self.find_nearest(self.z, z)
            if abs(nearest_z - z) < tol:
                idx = list(self.z).index(nearest_z)
                return getattr(self, param)[idx]
            else:
                # print('interpolate!', z, self['z'])
                return self.interpolate(z=z, value=param, index='z')

    def get_parameter_at_element(self, param, element_name):
        if element_name in self.element_name:
            idx = list(self.element_name).index(element_name)
            return self[param][idx]
        return None

    def get_twiss_dict(self, idx):
        twissdict = {}
        for param in self.model_fields:
            try:
                twissdict[param] = getattr(self, param).val[idx]
            except Exception:
                pass
        return twissdict

    def get_twiss_at_element(self, element_name, before=False):
        if element_name in self.element_name:
            idx = list(self.element_name).index(element_name)
            if before:
                idx = idx - 1
            return self.get_twiss_dict(idx)
        return None

    def get_twiss_at_z(self, z, tol=1e-3):
        if z in self.z:
            idx = list(self.z).index(z)
            return self.get_twiss_dict(idx)
        else:
            nearest_z = self.find_nearest(self.z, z)
            if abs(nearest_z - z) < tol:
                idx = list(self.z).index(nearest_z)
                return self.get_twiss_dict(idx)
            else:
                twissdict = {}
                for param in [k for k in self.model_fields if getattr(self, k).dtype == 'f']:
                    twissdict[param] = self.interpolate(z=z, value=param, index='z')
                return twissdict

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
        return self

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
