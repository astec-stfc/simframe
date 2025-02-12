import numpy as np
from ..constants import speed_of_light
from warnings import warn
from ..SDDSFile import SDDSFile, SDDS_Types


def write_SDDS_field_file(self, sddsindex=0, ascii=False):
    """Save an SDDS file using the SDDS class."""
    sdds_filename = self._output_filename(extension=".sdds")
    sddsfile = SDDSFile(index=sddsindex, ascii=ascii)
    data = None
    if self.field_type == "LongitudinalWake":
        zdata = self.z.value.val
        wzdata = self.Wz.value.val
        tdata = zdata / speed_of_light
        data = np.array([zdata, tdata, wzdata])
        cnames = ["z", "t", "W"]
        cunits = ["m", "s", "V/C"]
        ccolumns = [
            zdata,
            tdata,
            wzdata,
        ]
    elif self.field_type == "TransverseWake":
        zdata = self.z.value.val
        tdata = zdata / speed_of_light
        wxdata = self.Wx.value.val
        wydata = self.Wy.value.val
        data = np.array(
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
        zdata = self.z.value.val
        tdata = zdata / speed_of_light
        wxdata = self.Wx.value.val
        wydata = self.Wy.value.val
        wzdata = self.Wz.value.val
        data = np.array(
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
    else:
        warn(f"Field type {self.field_type} not supported for SDDS")
        return
    if data is not None:
        ctypes = [SDDS_Types.SDDS_DOUBLE for _ in len(data)]
        csymbols = ["" for _ in len(data)]
        sddsfile.add_columns(cnames, ccolumns, ctypes, cunits, csymbols)
        sddsfile.write_file(sdds_filename)
    return sdds_filename
