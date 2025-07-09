import os
import h5py
from ..units import UnitValue
from . import FieldParameter
from ..constants import speed_of_light
from warnings import warn
from . import (
    allowed_cavities,
    tw_required_attrs,
)


def read_HDF5_field_file(self, filename, normalise=True):
    self.reset_dicts()
    with h5py.File(filename, "r") as h5file:
        for key, value in h5file.attrs.items():
            if key == "type":
                setattr(self, "field_type", value)
            else:
                setattr(self, key, value)
        if "origin_code" in list(h5file.attrs.keys()):
            self.origin_code = h5file.attrs["origin_code"]
        if "norm" in list(h5file.attrs.keys()):
            if h5file.attrs["norm"] != 1.0:
                warn("Warning: Field is not normalised to 1.0.")
            self.norm = h5file.attrs["norm"]
        else:
            self.norm = 1.0
        if "cavity_type" in list(h5file.attrs.keys()):
            cavtype = h5file.attrs["cavity_type"]
            if cavtype not in allowed_cavities:
                raise Exception(
                    f"cavity_type attributes of {filename} must be in {allowed_cavities}"
                )
            else:
                setattr(self, "cavity_type", cavtype)
            if cavtype == "TravellingWave":
                for param in tw_required_attrs:
                    if not param in (h5file.attrs.keys()):
                        raise Exception(
                            f"{param} must be an attribute of {filename} for a travelling wave linac"
                        )
                    else:
                        setattr(self, param, h5file.attrs[param])
        length_set = (
            False if self.field_type != "2DElectroDynamic" else h5file.attrs["length"]
        )
        for key in h5file:
            if not length_set:
                self.length = len(h5file[key][()])
            setattr(
                self,
                key,
                FieldParameter(
                    name=key,
                    value=UnitValue(h5file[key][()], units=h5file[key].attrs["units"]),
                ),
            )
        # if ("z" not in list(h5file.keys())) and ("t" in list(h5file.keys())):
        #     setattr(self, "z", self.t.value * speed_of_light)
        for param in ["Ex", "Ey", "Ez", "Er", "Bx", "By", "Bz", "Br", "G"]:
            if getattr(self, param).value is not None:
                lessthancond = 0.99 > round(max(abs(getattr(self, param).value.val)), 4)
                greaterthancond = (
                    round(max(abs(getattr(self, param).value.val)), 4) > 1.1
                )
                if lessthancond or greaterthancond:
                    warn(
                        f"Warning: Field {param} in {filename} is not normalised to 1.0."
                    )
        self.read = True
    return filename


def write_HDF5_field_file(self):
    self.filename = os.path.splitext(self.filename)[0] + ".hdf5"
    with h5py.File(self.filename, "w") as h5file:
        attrs = [
            "field_type",
            "frequency",
            "radius",
            "fourier",
            "cavity_type",
            "start_cell_z",
            "end_cell_z",
            "mode_numerator",
            "mode_denominator",
            "orientation",
        ]
        for att in attrs:
            if hasattr(self, att):
                if getattr(self, att) is not None:
                    h5file.attrs[att] = getattr(self, att)
        dsets = [
            "x",
            "y",
            "z",
            "r",
            "Ex",
            "Ey",
            "Ez",
            "Er",
            "Bx",
            "By",
            "Bz",
            "Br",
            "Wx",
            "Wy",
            "Wz",
            "Wr",
            "G",
        ]
        for dset in dsets:
            if isinstance(getattr(self, dset), FieldParameter):
                if getattr(self, dset).value is not None:
                    dataset = h5file.create_dataset(
                        dset, data=getattr(self, dset).value
                    )
                    dataset.attrs["units"] = getattr(self, dset).value.units
    return self.filename
