import numpy as np
from ..constants import speed_of_light
from warnings import warn
from ..SDDSFile import SDDSFile, SDDS_Types


def write_SDDS_field_file(self, sddsindex=0, ascii=False):
    """Save an SDDS file using the SDDS class."""
    sdds_filename = self._output_filename(extension=".sdds")
    sddsfile = SDDSFile(index=sddsindex, ascii=ascii)
    zdata = self.z_values
    tdata = self.t_values
    if self.field_type == "LongitudinalWake":
        wzdata = self.Wz.value.val
        cnames = ["z", "t", "Wz"]
        cunits = ["m", "s", "V/C"]
        ccolumns = [
            zdata,
            tdata,
            wzdata,
        ]
    elif self.field_type == "TransverseWake":
        wxdata = self.Wx.value.val
        wydata = self.Wy.value.val
        ccolumns = np.array(
            [
                zdata,
                tdata,
                wxdata,
                wydata,
            ]
        )
        cnames = ["z", "t", "Wx", "Wy"]
        cunits = ["m", "s", "V/C/m", "V/C/m"]
    elif self.field_type == "3DWake":
        wxdata = self.Wx.value.val
        wydata = self.Wy.value.val
        wzdata = self.Wz.value.val
        ccolumns = np.array(
            [
                zdata,
                tdata,
                wxdata,
                wydata,
                wzdata,
            ]
        )
        cnames = ["z", "t", "Wx", "Wy", "Wz"]
        cunits = ["m", "s", "V/C/m", "V/C/m", "V/C"]
    elif self.field_type == "1DElectroDynamic":
        ezdata = self.Ez.value.val
        cnames = ["z", "Ez"]
        cunits = ["m", "V"]
        ccolumns = [
            zdata,
            ezdata,
        ]
    else:
        warn(f"Field type {self.field_type} not supported for SDDS")
        return
    if ccolumns is not None:
        ctypes = [SDDS_Types.SDDS_DOUBLE for _ in ccolumns]
        csymbols = ["" for _ in ccolumns]
        sddsfile.add_columns(cnames, ccolumns, ctypes, cunits, csymbols)
        sddsfile.write_file(sdds_filename)
    return sdds_filename
