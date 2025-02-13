import numpy as np
from warnings import warn


def generate_astra_field_data(self):
    length = str(self.length)
    data = None
    zdata = self.z_values
    if self.field_type == "LongitudinalWake":
        wzdata = self.Wz.value.val
        data = np.concatenate([np.array([[length, ""]]), np.transpose([zdata, wzdata])])
    elif self.field_type == "TransverseWake":
        wxdata = self.Wx.value.val
        wydata = self.Wy.value.val
        data = np.concatenate(
            [np.array([[length, ""]]), np.transpose([zdata, wxdata, wydata])]
        )
    elif self.field_type == "3DWake":
        zpreamble = np.array(
            [
                [3, 0],
                [length, 0],
                [0, 0],
                [0, 0],
            ]
        )
        xpreamble = np.array(
            [
                [length, 0],
                [0, 0],
                [0, 13],
            ]
        )
        ypreamble = np.array(
            [
                [length, 0],
                [0, 0],
                [0, 24],
            ]
        )
        wxdata = self.Wx.value.val
        wydata = self.Wy.value.val
        wzdata = self.Wz.value.val
        zvals = np.concatenate([zpreamble, np.transpose([zdata, wzdata])])
        xvals = np.concatenate([xpreamble, np.transpose([zdata, wxdata])])
        yvals = np.concatenate([ypreamble, np.transpose([zdata, wydata])])
        data = np.concatenate([zvals, xvals, yvals])
    elif self.field_type == "1DMagnetoStatic":
        bzdata = self.Bz.value.val
        data = np.transpose([zdata, bzdata])
    elif self.field_type == "1DElectroDynamic":
        ezdata = self.Ez.value.val
        if self.cavity_type == "TravellingWave":
            spdata = ["" for _ in range(self.length)]
            preamble = np.array(
                [
                    [
                        self.start_cell_z,
                        self.end_cell_z,
                        self.mode_numerator,
                        self.mode_denominator,
                    ]
                ]
            )
            data = np.concatenate(
                [preamble, np.transpose([zdata, ezdata, spdata, spdata])]
            )
        else:
            data = np.transpose([zdata, ezdata])
    else:
        warn(f"Field type {self.field_type} not supported for ASTRA")
    return data


def write_astra_field_file(self):
    astra_file = self._output_filename(extension=".astra")
    data = generate_astra_field_data(self)
    if data is not None:
        with open(f"{astra_file}", "w") as f:
            for d in data:
                f.write(" ".join([str(x) for x in d]) + "\n")
    return astra_file
