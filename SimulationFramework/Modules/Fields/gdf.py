from math import floor
import numpy as np
import easygdf
from warnings import warn
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
