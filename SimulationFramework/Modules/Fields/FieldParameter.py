from pydantic import BaseModel, ConfigDict
from ..units import UnitValue


class FieldParameter(BaseModel):
    """
    FieldParameter class to represent a field parameter with a name and an optional value.

    Attributes:
    -----------
    name (str):
        The name of the field parameter.
    value (UnitValue | None):
        The value of the field parameter, which can be a UnitValue or None.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    value: UnitValue | None = None
