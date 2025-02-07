import os
from ..units import UnitValue
from warnings import warn
from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)
from typing import Optional, Literal

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
]

allowed_formats = [
    "astra",
    "sdds",
    "opal",
    "gdf",
]

fieldtype = Literal[*allowed_fields]

# I can't think of a clever way of doing this, so...
def get_properties(obj):
    return [f for f in dir(obj) if type(getattr(obj, f)) is property]

class FieldParameter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    value: Optional[UnitValue] = None

from . import astra
from . import gdf
from . import hdf5
from . import opal
from . import sdds

class field(BaseModel):
    x: Optional[FieldParameter(name="x")] = None
    y: Optional[FieldParameter(name="y")] = None
    z: Optional[FieldParameter(name="z")] = None
    r: Optional[FieldParameter(name="r")] = None
    t: Optional[FieldParameter(name="t")] = None
    Ex: Optional[FieldParameter(name="Ex")] = None
    Ey: Optional[FieldParameter(name="Ey")] = None
    Ez: Optional[FieldParameter(name="Ez")] = None
    Er: Optional[FieldParameter(name="Er")] = None
    Bx: Optional[FieldParameter(name="Bx")] = None
    By: Optional[FieldParameter(name="By")] = None
    Bz: Optional[FieldParameter(name="Bz")] = None
    Br: Optional[FieldParameter(name="Br")] = None
    Wx: Optional[FieldParameter(name="Ex")] = None
    Wy: Optional[FieldParameter(name="Ey")] = None
    Wz: Optional[FieldParameter(name="Ez")] = None
    Wr: Optional[FieldParameter(name="Er")] = None
    filename: Optional[str] = None
    field_type: Optional[fieldtype] = None
    origin_code: Optional[str] = None
    norm: float = 1.0
    read: bool = False
    length: Optional[int] = None
    frequency: Optional[float] = None
    radius: Optional[float] = None # MAGIC NUMBER FOR SOLENOID RADIUS, DEFAULTS TO 10cm in write_opal_field_file
    fourier: int = 100

    def __init__(
            self,
            filename = None,
            *args,
            **kwargs,
    ):
        field.filename = filename
        super(
            field,
            self,
        ).__init__(
            filename=filename,
            *args,
            **kwargs,
        )
        if filename is not None:
            self.read_field_file(filename)

    @model_validator(mode="before")
    def validate_fields(cls, values):
        return values

    def reset_dicts(self):
        self.origin_code = None
        self.field_type = None
        self.norm = 1.0
        setattr(self, "t", FieldParameter(name="t"))
        for par in ["x", "y", "z", "r", ]:
            setattr(self, par, FieldParameter(name=par))
            setattr(self, f"E{par}", FieldParameter(name=f"E{par}"))
            setattr(self, f"B{par}", FieldParameter(name=f"B{par}"))
            setattr(self, f"W{par}", FieldParameter(name=f"W{par}"))
        self.read = False

    def read_field_file(self, filename: str):
        fext = os.path.splitext(os.path.basename(filename))[-1]
        if fext == '.hdf5':
            hdf5.read_HDF5_field_file(self, filename)

    def write_field_file(self, code: str):
        if not self.read:
            print("Field file not read in. Use read_field_file to load in an hdf5 field file.")
            return
        try:
            if code.lower() == "astra":
                astra.write_astra_field_file(self)
            elif code.lower() == "sdds":
                sdds.write_SDDS_field_file(self)
            elif code.lower() == "opal":
                opal.write_opal_field_file(self, frequency=self.frequency, radius=self.radius, fourier=self.fourier)
        except NotImplementedError as e:
            print("Supported formats are [astra, sdds, opal, gdf]")


