from math import floor
import numpy as np
import easygdf
from warnings import warn


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
            startpos = zdata.index(self.start_cell_z)
            stoppos = zdata.index(self.end_cell_z)
            halfcell1 = fielddata[:startpos]
            halfcell1end = zdata[startpos]
            halfcell2 = fielddata[stoppos:]
            halfcell2start = zdata[stoppos]
            zstep = zdata[1] - zdata[0]
            lambdaRF = self.speed_of_light/self.frequency
            ncells = floor(self.length / lambdaRF - 1)
            nsteps = ncells * lambdaRF / zstep
            middleRF = np.array([[(x*zstep + halfcell1end), np.cos((2 * np.pi / lambdaRF) * (x*zstep))] for x in range(0, nsteps)])
            halfcell2[:, 0] += middleRF[-1, 0] - halfcell2start
            ezdata = np.concatenate([halfcell1, middleRF, halfcell2])
        blocks = [
            {"name": "z", "value": zdata},
            {"name": "Ez", "value": ezdata},
        ]
    else:
        warn(f"Field type {self.field_type} not supported for GPT")
    if blocks is not None:
        easygdf.save(gdf_file, blocks)
    return gdf_file
