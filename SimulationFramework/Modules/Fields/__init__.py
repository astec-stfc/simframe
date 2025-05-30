import os
import warnings

from ..units import UnitValue
from ..constants import speed_of_light
from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)
from typing import Literal

allowed_fields = [
    "1DElectroStatic",
    "1DMagnetoStatic",
    "1DElectroDynamic",
    "2DElectroStatic",
    "2DMagnetoStatic",
    "2DElectroDynamic",
    "3DElectroStatic",
    "3DMagnetoStatic",
    "3DElectroDynamic",
    "LongitudinalWake",
    "TransverseWake",
    "3DWake",
    "1DQuadrupole",
]

allowed_formats = [
    "astra",
    "sdds",
    "opal",
    "gdf",
]

allowed_cavities = [
    "StandingWave",
    "TravellingWave",
]

tw_required_attrs = [
    "start_cell_z",
    "end_cell_z",
    "mode_numerator",
    "mode_denominator",
]

fieldtype = Literal[
    "1DElectroStatic",
    "1DMagnetoStatic",
    "1DElectroDynamic",
    "2DElectroStatic",
    "2DMagnetoStatic",
    "2DElectroDynamic",
    "3DElectroStatic",
    "3DMagnetoStatic",
    "3DElectroDynamic",
    "LongitudinalWake",
    "TransverseWake",
    "3DWake",
    "1DQuadrupole",
]
cavitytype = Literal[
    "StandingWave",
    "TravellingWave",
]


class FieldParameter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    value: UnitValue | None = None


from . import astra  # noqa E402
from . import gdf  # noqa E402
from . import hdf5  # noqa E402
from . import sdds  # noqa E402
from . import opal  # noqa E402


class field(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    x: FieldParameter = None
    y: FieldParameter = None
    z: FieldParameter = None
    r: FieldParameter = None
    t: FieldParameter = None
    Ex: FieldParameter = None
    Ey: FieldParameter = None
    Ez: FieldParameter = None
    Er: FieldParameter = None
    Bx: FieldParameter = None
    By: FieldParameter = None
    Bz: FieldParameter = None
    Br: FieldParameter = None
    Wx: FieldParameter = None
    Wy: FieldParameter = None
    Wz: FieldParameter = None
    Wr: FieldParameter = None
    G: FieldParameter = None
    filename: str | None = None
    field_type: fieldtype | None = None
    origin_code: str | None = None
    norm: float = 1.0
    read: bool = False
    length: int | None = None
    frequency: float | None = None
    radius: float | None = (
        None  # MAGIC NUMBER FOR SOLENOID RADIUS, DEFAULTS TO 10cm in write_opal_field_file
    )
    fourier: int = 100
    cavity_type: cavitytype | None = None
    start_cell_z: float | None = None
    end_cell_z: float | None = None
    mode_numerator: float | None = None
    mode_denominator: float | None = None
    orientation: str | None = None
    n_cells: int | float | None = None
    # For TW linacs in ASTRA, the mode is 2π mode_numerator / mode_denominator

    def __init__(
        self,
        filename=None,
        field_type=None,
        frequency=None,
        cavity_type=None,
        *args,
        **kwargs,
    ):
        field.filename = filename
        field.field_type = field_type,
        field.frequency = frequency,
        field.cavity_type = cavity_type,
        super(
            field,
            self,
        ).__init__(
            filename=filename,
            field_type=field_type,
            frequency=frequency,
            cavity_type=cavity_type,
            *args,
            **kwargs,
        )
        if filename is not None:
            self.read_field_file(filename, field_type=field_type, frequency=frequency, cavity_type=cavity_type)

    @model_validator(mode="before")
    def validate_fields(cls, values):
        return values

    def model_dump(self):
        return self.filename

    def reset_dicts(self):
        self.origin_code = None
        self.field_type = None
        self.norm = 1.0
        setattr(self, "t", FieldParameter(name="t"))
        for par in [
            "x",
            "y",
            "z",
            "r",
        ]:
            setattr(self, par, FieldParameter(name=par))
            setattr(self, f"E{par}", FieldParameter(name=f"E{par}"))
            setattr(self, f"B{par}", FieldParameter(name=f"B{par}"))
            setattr(self, f"W{par}", FieldParameter(name=f"W{par}"))
        setattr(self, "G", FieldParameter(name="G"))
        self.read = False

    @property
    def z_values(self):
        if self.t.value is not None and self.z.value is None:
            return -1 * abs(self.t.value.val * speed_of_light)
        return self.z.value.val

    @property
    def t_values(self):
        if self.z is not None and self.t.value is None:
            return abs(self.z.value.val / speed_of_light)
        return self.t.value.val

    def read_field_file(self, filename: str, field_type: str | None = None, cavity_type: str | None = None, frequency: float | None = None):
        fext = os.path.splitext(os.path.basename(filename))[-1]
        if fext == ".hdf5":
            hdf5.read_HDF5_field_file(self, filename)
        else:
            if fext.lower() in [".astra", ".dat"]:
                # print('Field: read_field_file: astra', filename, fext.lower())
                astra.read_astra_field_file(self, filename, field_type=field_type, cavity_type=cavity_type, frequency=frequency)
            elif fext.lower() in [".sdds"]:
                # print('Field: read_field_file: SDDS', filename, fext.lower())
                sdds.read_SDDS_field_file(self, filename, field_type=field_type)
            elif fext.lower() in [".gdf"]:
                # print('Field: read_field_file: GPT', filename, fext.lower())
                gdf.read_gdf_field_file(self, filename, field_type=field_type, cavity_type=cavity_type, frequency=frequency)
            elif fext.lower() in [".opal"]:
                # print('Field: read_field_file: opal', filename, fext.lower())
                opal.read_opal_field_file(self, filename, field_type=field_type, cavity_type=cavity_type, frequency=frequency)
            self.read = True

    def _output_filename(self, extension: str = ".hdf5", location: str | None = None):
        if location is not None:
            _output_location = os.path.dirname(os.path.abspath(location))
        else:
            if not hasattr(self, '_output_location'):
                _output_location = os.path.dirname(os.path.dirname(self.filename))
            else:
                _output_location = self._output_location
        basefilename = os.path.basename(self.filename)
        pre, _ = os.path.splitext(basefilename)
        return os.path.relpath(os.path.join(_output_location, pre + extension))

    def get_field_data(self, code: str):
        if not self.read:
            print(
                "Field file not read in. Use read_field_file to load in an hdf5 field file."
            )
            return
        # try:
        if code.lower() in ["astra", "ocelot"]:
            return astra.generate_astra_field_data(self)
        return None
        # elif code.lower() in ["sdds", "elegant"]:
        #     return sdds.write_SDDS_field_file(self)
        # elif code.lower() in ["gdf", "gpt"]:
        #     return gdf.write_gdf_field_file(self)
        # elif code.lower() == "opal":
        #     return opal.write_opal_field_file(
        #         self,
        #         frequency=self.frequency,
        #         radius=self.radius,
        #         fourier=self.fourier,
        #         orientation=self.orientation,
        #     )

    def write_field_file(self, code: str, location: str | None = None):
        if not self.read:
            print(
                "Field file not read in. Use read_field_file to load in an hdf5 field file."
            )
            return
        # try:
        if location is not None:
            self._output_location = os.path.dirname(os.path.abspath(location))
        else:
            self._output_location = os.path.dirname(os.path.dirname(self.filename))
        if code.lower() in ["astra", "ocelot"]:
            return astra.write_astra_field_file(self)
        elif code.lower() in ["sdds", "elegant"]:
            return sdds.write_SDDS_field_file(self)
        elif code.lower() in ["gdf", "gpt"]:
            return gdf.write_gdf_field_file(self)
        elif code.lower() == "opal":
            return opal.write_opal_field_file(
                self,
                frequency=self.frequency,
                radius=self.radius,
                fourier=self.fourier,
                orientation=self.orientation,
            )
        elif code.lower() == "hdf5":
            return hdf5.write_HDF5_field_file(self)
        # except NotImplementedError:
        #     print("Supported formats are [astra, sdds, opal, gdf]")
