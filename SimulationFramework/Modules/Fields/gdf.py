from math import floor
import numpy as np
import easygdf
from warnings import warn
from . import FieldParameter
from ..units import UnitValue
from ..constants import speed_of_light


def write_gdf_field_file(self):
    gdf_file = self._output_filename(extension=".gdf")
    blocks = None
    zdata = self.z.value.val
    if self.field_type == "LongitudinalWake":
        wzdata = self.Wz.value.val
        blocks = [
            {"name": "z", "value": zdata},
            {"name": "Wz", "value": wzdata},
        ]
    elif self.field_type == "TransverseWake":
        wxdata = self.Wx.value.val
        wydata = self.Wy.value.val
        blocks = [
            {"name": "z", "value": zdata},
            {"name": "Wx", "value": wxdata},
            {"name": "Wy", "value": wydata},
        ]
    elif self.field_type == "3DWake":
        wxdata = self.Wx.value.val
        wydata = self.Wy.value.val
        wzdata = self.Wz.value.val
        blocks = [
            {"name": "z", "value": zdata},
            {"name": "Wx", "value": wxdata},
            {"name": "Wy", "value": wydata},
            {"name": "Wz", "value": wzdata},
        ]
    elif self.field_type == "1DMagnetoStatic":
        bzdata = self.Bz.value.val
        blocks = [
            {"name": "z", "value": zdata},
            {"name": "Bz", "value": bzdata},
        ]
    elif self.field_type == "1DElectroDynamic":
        ezdata = self.Ez.value.val
        fielddata = np.array([zdata, ezdata]).transpose()
        if self.cavity_type == "TravellingWave":
            startpos = list(zdata).index(self.start_cell_z)
            stoppos = list(zdata).index(self.end_cell_z)
            halfcell1 = 1.*fielddata[:startpos]
            halfcell2 = 1.*halfcell1[::-1]
            halfcell1[:, 1] /= max(halfcell1[:, 1])
            halfcell2[:, 0] = halfcell2[:, 0][::-1]
            halfcell2[:, 1] /= max(halfcell2[:, 1])
            halfcell1end = halfcell1[-1, 0]
            zstep = zdata[1] - zdata[0]
            lambdaRF = speed_of_light/self.frequency
            ncells = (self.n_cells - 1) * self.mode_numerator / self.mode_denominator
            nsteps = int(np.floor((ncells-1) * lambdaRF / zstep))
            middleRF = np.array([[(x*zstep + halfcell1end + zstep), np.cos((2 * np.pi / lambdaRF) * (x*zstep))] for x in range(0, nsteps+1)])
            halfcell2[:, 0] += middleRF[-1, 0] + zstep
            zdata, ezdata = np.concatenate([halfcell1, middleRF, halfcell2]).transpose()
        blocks = [
            {"name": "z", "value": zdata},
            {"name": "Ez", "value": ezdata},
        ]
    else:
        warn(f"Field type {self.field_type} not supported for GPT")
    if blocks is not None:
        easygdf.save(gdf_file, blocks)
    return gdf_file

def read_gdf_field_file(self, filename: str, field_type: str, cavity_type: str | None =  None, frequency: float | None = None):
    self.reset_dicts()
    setattr(self, "field_type", field_type)
    if "Electro" in field_type:
        if cavity_type is None:
            raise ValueError(f"cavity_type must be provided for {field_type}")
        else:
            setattr(self, "cavity_type", cavity_type)
        if frequency is None:
            raise ValueError(f"frequency must be provided for {field_type}")
        else:
            setattr(self, "frequency", frequency)
    fdat = easygdf.load(filename)["blocks"]
    try:
        zval = [k["value"] for k in fdat if k["name"].lower() == "z"][0]
    except:
        zval = [k["value"] * speed_of_light for k in fdat if k["name"].lower() == "t"][0]
    if field_type == "1DMagnetoStatic":
        bzval = [k["value"] for k in fdat if k["name"].lower().capitalize() == "Bz"][0]
        setattr(self, "z", FieldParameter(name="z", value=UnitValue(zval, units="m")))
        setattr(self, "Bz", FieldParameter(name="Bz", value=UnitValue(bzval/np.max(bzval), units="T")))
    elif field_type == "1DElectroDynamic":
        if cavity_type == "StandingWave":
            ezval = [k["value"] for k in fdat if k["name"].lower().capitalize() == "Ez"][0]
            setattr(self, "z", FieldParameter(name="z", value=UnitValue(zval, units="m")))
            setattr(self, "Ez", FieldParameter(name="Ez", value=UnitValue(ezval / np.max(ezval), units="V/m")))
        elif cavity_type == "TravellingWave":
            raise NotImplementedError(f"{cavity_type} not implemented for GDF files")
    elif field_type == "LongitudinalWake":
            wzval = [k["value"] for k in fdat if k["name"].lower().capitalize() == "Wz"][0]
            setattr(self, "z", FieldParameter(name="z", value=UnitValue(zval, units="m")))
            setattr(self, "Wz", FieldParameter(name="Wz", value=UnitValue(wzval, units="V/C")))
    elif field_type == "TransverseWake":
        wxval = [k["value"] for k in fdat if k["name"].lower().capitalize() == "Wx"][0]
        wyval = [k["value"] for k in fdat if k["name"].lower().capitalize() == "Wy"][0]
        setattr(self, "z", FieldParameter(name="z", value=UnitValue(zval, units="m")))
        setattr(self, "Wx", FieldParameter(name="Wx", value=UnitValue(wxval, units="V/C/m")))
        setattr(self, "Wy", FieldParameter(name="Wy", value=UnitValue(wyval, units="V/C/m")))
    elif field_type == "3DWake":
        wxval = [k["value"] for k in fdat if k["name"].lower().capitalize() == "Wx"][0]
        wyval = [k["value"] for k in fdat if k["name"].lower().capitalize() == "Wy"][0]
        wzval = [k["value"] for k in fdat if k["name"].lower().capitalize() == "Wz"][0]
        setattr(self, "z", FieldParameter(name="z", value=UnitValue(zval, units="m")))
        setattr(self, "Wx", FieldParameter(name="Wx", value=UnitValue(wxval, units="V/C/m")))
        setattr(self, "Wy", FieldParameter(name="Wy", value=UnitValue(wyval, units="V/C/m")))
        setattr(self, "Wz", FieldParameter(name="Wz", value=UnitValue(wzval, units="V/C")))
    else:
        raise NotImplementedError(f"{field_type} loading not implemented for ASTRA files")