import h5py
from jedi.debug import speed

from ..units import UnitValue
from . import FieldParameter
from ..constants import speed_of_light
from warnings import warn

def read_HDF5_field_file(self, filename, local=False):
    self.reset_dicts()
    self.filename = filename
    with h5py.File(filename, "r") as h5file:
        try:
            if 'type' in list(h5file.attrs.keys()):
                self.field_type = h5file.attrs["type"]
            else:
                self.field_type = h5file.attrs["field_type"]
        except NameError as e:
            self.reset_dicts()
            print("ERROR! Field type is not defined")
        if "origin_code" in list(h5file.attrs.keys()):
            self.origin_code = h5file.attrs["origin_code"]
        if "norm" in list(h5file.attrs.keys()):
            if h5file.attrs["norm"] != 1.0:
                warn("Warning: Field is not normalised to 1.0.")
            self.norm = h5file.attrs["norm"]
        else:
            self.norm = 1.0
        length_set = False
        for key in h5file:
            if not length_set:
                self.length = len(h5file[key][()])
            setattr(self, key,
                    FieldParameter(name=key, value=UnitValue(h5file[key][()], units=h5file[key].attrs['units'])))
        if ("z" not in list(h5file.keys())) and ("t" in list(h5file.keys())):
            setattr(self, "z", self.t * speed_of_light)
        for param in ["Ex", "Ey", "Ez", "Er", "Bx", "By", "Bz", "Br"]:
            if getattr(self, param).value:
                if 0.999 < round(max(abs(getattr(self, param).value.val)), 3) < 1.001:
                    warn("Warning: Field is not normalised to 1.0.")
        self.read = True