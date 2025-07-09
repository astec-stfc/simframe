import numpy as np
import re
from warnings import warn
from . import FieldParameter
from ..units import UnitValue

d = ",!?/&-:;@'\n \t"


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
    elif self.field_type == "3DMagnetoStatic":
        xdata = self.x.value.val
        ydata = self.y.value.val
        bzdata = self.Bz.value.val
        data = np.array(
            [
                [z, Bz]
                for x, y, z, Bz in zip(xdata, ydata, zdata, bzdata)
                if x == 0.0 and y == 0.0
            ]
        )
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
    elif self.field_type == "1DQuadrupole":
        gdata = self.G.value.val
        data = np.transpose([zdata, gdata])
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


def read_astra_field_file(
    self,
    filename: str,
    field_type: str,
    cavity_type: str | None = None,
    frequency: float | None = None,
):
    self.reset_dicts()
    setattr(self, "field_type", field_type)
    try:
        if "Electro" in field_type:
            if cavity_type is None:
                raise ValueError(f"cavity_type must be provided for {field_type}")
            else:
                setattr(self, "cavity_type", cavity_type)
            if frequency is None:
                raise ValueError(f"frequency must be provided for {field_type}")
            else:
                setattr(self, "frequency", frequency)
    except Exception:
        raise ValueError(
            f"Fields read_astra_field_file error: {filename}, {field_type}, {cavity_type}, {frequency}"
        )
    if field_type == "1DMagnetoStatic":
        fdat = np.loadtxt(filename)
        setattr(
            self, "z", FieldParameter(name="z", value=UnitValue(fdat[::, 0], units="m"))
        )
        setattr(
            self,
            "Bz",
            FieldParameter(
                name="Bz", value=UnitValue(fdat[::, 1] / np.max(fdat[::, 1]), units="T")
            ),
        )
    elif field_type == "1DElectroDynamic":
        if cavity_type == "StandingWave":
            fdat = np.loadtxt(filename)
            setattr(
                self,
                "z",
                FieldParameter(name="z", value=UnitValue(fdat[::, 0], units="m")),
            )
            setattr(
                self,
                "Ez",
                FieldParameter(
                    name="Ez",
                    value=UnitValue(fdat[::, 1] / np.max(fdat[::, 1]), units="V/m"),
                ),
            )
        elif cavity_type == "TravellingWave":
            with open(filename) as f:
                rl = f.readlines()[0]
                twdat = re.split("[" + "\\".join(d) + "]", rl)
                setattr(self, "start_cell_z", float(twdat[0]))
                setattr(self, "end_cell_z", float(twdat[1]))
                setattr(self, "mode_numerator", float(twdat[2]))
                setattr(self, "mode_denominator", float(twdat[3]))
            fdat = np.loadtxt(filename, skiprows=1)
            setattr(
                self,
                "z",
                FieldParameter(name="z", value=UnitValue(fdat[::, 0], units="m")),
            )
            setattr(
                self,
                "Ez",
                FieldParameter(
                    name="Ez",
                    value=UnitValue(fdat[::, 1] / np.max(fdat[::, 1]), units="V/m"),
                ),
            )
    elif field_type == "LongitudinalWake":
        try:
            fdat = np.loadtxt(filename)
            numrows = int(fdat[1, 0])
            setattr(
                self,
                "z",
                FieldParameter(
                    name="z", value=UnitValue(fdat[4 : 4 + numrows][::, 0], units="m")
                ),
            )
            setattr(
                self,
                "Wz",
                FieldParameter(
                    name="Wz",
                    value=UnitValue(fdat[4 : 4 + numrows][::, 1]),
                    units="V/C",
                ),
            )
        except:
            fdat = np.loadtxt(filename, skiprows=1)
            setattr(
                self,
                "z",
                FieldParameter(name="z", value=UnitValue(fdat[::, 0], units="m")),
            )
            setattr(
                self,
                "Wz",
                FieldParameter(name="Wz", value=UnitValue(fdat[::, 1]), units="V/C"),
            )
    elif field_type == "3DWake":
        fdat = np.loadtxt(filename)
        numrows = int(fdat[1, 0]) - 1
        setattr(
            self,
            "z",
            FieldParameter(
                name="z", value=UnitValue(fdat[4 : 4 + numrows][::, 0], units="m")
            ),
        )
        setattr(
            self,
            "Wz",
            FieldParameter(
                name="Wz", value=UnitValue(fdat[4 : 4 + numrows][::, 1]), units="V/C"
            ),
        )
        setattr(
            self,
            "Wx",
            FieldParameter(
                name="Wx",
                value=UnitValue(fdat[numrows + 8 : (2 * numrows) + 8][::, 1]),
                units="V/C/m",
            ),
        )
        setattr(
            self,
            "Wy",
            FieldParameter(
                name="Wy",
                value=UnitValue(fdat[(2 * numrows) + 12 :][::, 1]),
                units="V/C/m",
            ),
        )
    elif field_type == "1DQuadrupole":
        fdat = np.loadtxt(filename)
        setattr(
            self, "z", FieldParameter(name="z", value=UnitValue(fdat[::, 0], units="m"))
        )
        setattr(
            self,
            "G",
            FieldParameter(
                name="G",
                value=UnitValue(fdat[::, 1] / np.max(fdat[::, 1]), units="T/m"),
            ),
        )
    else:
        raise NotImplementedError(
            f"{field_type} loading not implemented for ASTRA files"
        )
