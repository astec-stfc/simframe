"""
Simframe Objects Module

Various objects and functions to handle simulation lattices, commands, and elements.

Classes:
    - :class:`~SimulationFramework.Framework_objects.runSetup`: Defines simulation run settings, allowing\
    for single runs, element scans or jitter/error studies.

    - :class:`~SimulationFramework.Framework_objects.frameworkObject`: Base class for generic objects in SimFrame,\
    including lattice elements and simulation code commands.

    - :class:`~SimulationFramework.Framework_objects.frameworkElement`: Base class for generic\
     lattice elements in SimFrame, including lattice elements and simulation code commands.

    - :class:`~SimulationFramework.Framework_objects.csrdrift`: Drift element including CSR effects.

    - :class:`~SimulationFramework.Framework_objects.lscdrift`: Drift element including LSC effects.

    - :class:`~SimulationFramework.Framework_objects.edrift`: Basic drift element.

    - :class:`~SimulationFramework.Framework_objects.frameworkLattice`: Base class for simulation lattices,\
    consisting of a line of :class:`~SimulationFramework.Framework_objects.frameworkObject` s.

    - :class:`~SimulationFramework.Framework_objects.frameworkCounter`: Used for counting elements of the same\
    type in ASTRA and CSRTrack

    - :class:`~SimulationFramework.Framework_objects.frameworkGroup`: Used for grouping together\
    :class:`~SimulationFramework.Framework_objects.frameworkObject` s and controlling them all simultaneously.

    - :class:`~SimulationFramework.Framework_objects.element_group`: Subclass of\
    :class:`~SimulationFramework.Framework_objects.frameworkGroup` for grouping elements.\
    # TODO is this ever used?

    - :class:`~SimulationFramework.Framework_objects.r56_group`: Subclass of\
    :class:`~SimulationFramework.Framework_objects.frameworkGroup` for grouping elements with an R56.\
    # TODO is this ever used?

    - :class:`~SimulationFramework.Framework_objects.chicane`: Subclass of\
    :class:`~SimulationFramework.Framework_objects.frameworkGroup` for a 4-dipole bunch compressor chicane.

    - :class:`~SimulationFramework.Framework_objects.getGrids`: Used for determining the appropriate number\
    of space charge grids given a number of particles.
"""

import os
import subprocess
from warnings import warn
from copy import deepcopy
import yaml
from .Modules.merge_two_dicts import merge_two_dicts
from .Modules.MathParser import MathParser
from .FrameworkHelperFunctions import chunks, expand_substitution, checkValue, chop, dot
from .FrameworkHelperFunctions import _rotation_matrix
from .Modules.Fields import field
from .Codes import Executables as exes

try:
    import numpy as np
except ImportError:
    np = None
from pydantic import (
    BaseModel,
    field_validator,
    PositiveInt,
    SerializeAsAny,
    computed_field,
)
from typing import (
    Dict,
    List,
    Any,
)

if os.name == "nt":
    # from .Modules.symmlinks import has_symlink_privilege
    def has_symlink_privilege():
        return False

else:

    def has_symlink_privilege():
        return True


with open(
    os.path.dirname(os.path.abspath(__file__)) + "/Codes/type_conversion_rules.yaml",
    "r",
) as infile:
    type_conversion_rules = yaml.safe_load(infile)
    type_conversion_rules_Elegant = type_conversion_rules["elegant"]
    type_conversion_rules_Names = type_conversion_rules["name"]

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/Codes/Elegant/commands_Elegant.yaml",
    "r",
) as infile:
    commandkeywords = yaml.safe_load(infile)

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/Elements/elementkeywords.yaml", "r"
) as infile:
    elementkeywords = yaml.safe_load(infile)

with open(
    os.path.dirname(os.path.abspath(__file__))
    + "/Codes/Elegant/keyword_conversion_rules_elegant.yaml",
    "r",
) as infile:
    keyword_conversion_rules_elegant = yaml.safe_load(infile)

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/Codes/Elegant/elements_Elegant.yaml",
    "r",
) as infile:
    elements_Elegant = yaml.safe_load(infile)

with open(
    os.path.dirname(os.path.abspath(__file__))
    + "/Codes/Ocelot/keyword_conversion_rules_ocelot.yaml",
    "r",
) as infile:
    keyword_conversion_rules_ocelot = yaml.safe_load(infile)

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/Codes/Ocelot/elements_Ocelot.yaml",
    "r",
) as infile:
    elements_Ocelot = yaml.safe_load(infile)


class runSetup(object):
    """
    Class defining settings for simulations that include multiple runs
    such as error studies or parameter scans.
    """

    def __init__(self):
        # define the number of runs and the random number seed
        self.nruns = 1
        self.seed = 0

        # init errorElement and elementScan settings as None
        self.elementErrors = None
        self.elementScan = None

    def setNRuns(self, nruns: int | float) -> None:
        """
        Sets the number of simulation runs to a new value.

        Parameters
        -----------
        nruns : int or float
            The number of runs to set. If a float is passed, it will be converted to an integer.

        Raises
        ------
        TypeError
            If `nruns` is not an integer or float.
        """
        # enforce integer argument type
        if isinstance(nruns, (int, float)):
            self.nruns = int(nruns)
        else:
            raise TypeError(
                "Argument nruns passed to runSetup instance must be an integer"
            )

    def setSeedValue(self, seed: int | float) -> None:
        """
        Sets the random number seed to a new value for all lattice objects

        Parameters
        -----------
        seed : int or float
            The random number seed to set. If a float is passed, it will be converted to an integer.

        Raises
        ------
        TypeError
            If `seed` is not an integer or float.
        """
        # enforce integer argument type
        if isinstance(seed, (int, float)):
            self.seed = int(seed)
        else:
            raise TypeError("Argument seed passed to runSetup must be an integer")

    def loadElementErrors(self, file: str | dict) -> None:
        """
        Load error definitions from a file or dictionary and assign them to the elementErrors attribute.
        This method can handle both a YAML file and a dictionary containing error definitions.

        Parameters
        -----------
        file: str or dict
            - str: Path to a YAML file containing error definitions.
            - dict: A dictionary containing error definitions.
        """
        # load error definitions from markup file
        error_setup = None
        if isinstance(file, str) and (".yaml" in file):
            with open(file, "r") as inputfile:
                error_setup = dict(yaml.safe_load(inputfile))
        # define errors from dictionary
        elif isinstance(file, dict):
            error_setup = file
        else:
            warn("error_setup must be a str or dict")

        if error_setup is not None and "elements" in list(error_setup.keys()):
            # assign the element error definitions
            self.elementErrors = error_setup["elements"]
            self.elementScan = None

            # set the number of runs and random number seed, if available
            if "nruns" in error_setup:
                self.setNRuns(error_setup["nruns"])
            if "seed" in error_setup:
                self.setSeedValue(error_setup["seed"])

    def setElementScan(
        self,
        name: str,
        item: str,
        scanrange: list | tuple | np.ndarray,
        multiplicative: bool = False,
    ) -> None:
        """
        Define a parameter scan for a single parameter of a given machine element

        Parameters
        -----------
        name : str
            Name of the machine element to be scanned.
        item : str
            Name of the item (parameter) to be scanned within the machine element.
        scanrange : list or tuple or np.ndarray
            A list or tuple containing two floats, representing the minimum and maximum values of the scan range.
        multiplicative : bool, optional
            If True, the scan will be multiplicative; otherwise, it will be additive. Default is False.
        """
        if not (isinstance(name, str) and isinstance(item, str)):
            raise TypeError(
                "Machine element name and item (parameter) must be defined as strings"
            )

        if (
            isinstance(scanrange, (list, tuple, np.ndarray))
            and (len(scanrange) == 2)
            and all([isinstance(x, (float, int)) for x in scanrange])
        ):
            minval, maxval = scanrange
        else:
            raise TypeError("Scan range (min. and max.) must be defined as floats")

        if not isinstance(multiplicative, bool):
            raise ValueError(
                "Argument multiplicative passed to runSetup.setElementScan must be a boolean"
            )

        # if no type errors were raised, build an assign a dictionary
        self.elementScan = {
            "name": name,
            "item": item,
            "min": minval,
            "max": maxval,
            "multiplicative": multiplicative,
        }
        self.elementErrors = None


class frameworkObject(BaseModel):
    """
    Class defining a framework object, which is the base class for all elements
    in a simulation lattice. It provides methods to add properties, validate parameters,
    and handle various simulation-specific functionalities.
    """

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True
        validate_assignment = True

    objectname: str
    """Name of the object, used as a unique identifier in the simulation."""

    objecttype: str
    """Type of the object, which determines its behavior and properties in the simulation."""

    objectdefaults: Dict = {}
    """Default values for the object's properties, used when no specific value is provided."""

    allowedkeywords: List | Dict = {}
    """List of allowed keywords for the object, which defines what properties can be set."""

    def __init__(self, *args, **kwargs):
        super(frameworkObject, self).__init__(
            *args,
            **kwargs,
        )
        if "global_parameters" in kwargs:
            self.global_parameters = kwargs["global_parameters"]
        if self.objecttype in commandkeywords:
            self.allowedkeywords = commandkeywords[self.objecttype]
        elif self.objecttype in elementkeywords:
            self.allowedkeywords = merge_two_dicts(
                elementkeywords[self.objecttype]["keywords"],
                elementkeywords["common"]["keywords"],
            )
            if "framework_keywords" in elementkeywords[self.objecttype]:
                self.allowedkeywords = merge_two_dicts(
                    self.allowedkeywords,
                    elementkeywords[self.objecttype]["framework_keywords"],
                )
        else:
            warn(f"Unknown type = {self.objecttype}")
            raise NameError
        self.allowedkeywords = [x.lower() for x in self.allowedkeywords]
        for key, value in list(kwargs.items()):
            self.add_property(key, value)

    @field_validator("objectname", mode="before")
    @classmethod
    def validate_objectname(cls, value: str) -> str:
        """Validate the objectname to ensure it is a string."""
        if not isinstance(value, str):
            raise ValueError("objectname must be a string.")
        return value

    @field_validator("objecttype", mode="before")
    @classmethod
    def validate_objecttype(cls, value: str) -> str:
        """Validate the objecttype to ensure it is a string."""
        if not isinstance(value, str):
            raise ValueError("objecttype must be a string.")
        return value

    def __setattr__(self, name, value):
        # Let Pydantic set known fields normally
        if name in frameworkObject.model_fields:
            super().__setattr__(name, value)
        else:
            try:
                super().__setattr__(name, value)
            except Exception:
                # Store extras in __dict__ (allowed by Config.extra = 'allow')
                self.__dict__[name] = value

    def change_Parameter(self, key: str, value: Any) -> None:
        """
        Change a parameter of the object by setting an attribute.

        Parameters
        ----------
        key: str
            The name of the parameter to change.
        value: Any
            The new value to set for the parameter.
        """
        setattr(self, key, value)

    def add_property(self, key: str, value: Any) -> None:
        """
        Add a property to the object by setting an attribute if the key is allowed.

        Parameters
        ----------
        key: str
            The name of the property to add.
        value: Any
            The value to set for the property.
        """
        key = key.lower()
        if key in self.allowedkeywords:
            try:
                setattr(self, key, value)
            except Exception as e:
                warn(f"add_property error: ({self.objecttype} [{key}]: {e}")

    def add_properties(self, **keyvalues: dict) -> None:
        """
        Add multiple properties to the object by setting attributes for each key-value pair.

        Parameters
        ----------
        **keyvalues: dict
            A dictionary of key-value pairs where keys are property names
            and values are the corresponding values to set.
        """
        for key, value in keyvalues.items():
            key = key.lower()
            if key in self.allowedkeywords:
                try:
                    setattr(self, key, value)
                except Exception as e:
                    warn(f"add_properties error: ({self.objecttype} [{key}]: {e}")

    def add_default(self, key: str, value: Any) -> None:
        """
        Add a default value for a property of the object, updating `objectdefaults`.

        Parameters
        ----------
        key: str
            The name of the property to set a default value for.
        value: Any
            The name of the property to set a default value for and the value to set.
        """
        self.objectdefaults[key] = value

    @property
    def parameters(self) -> list:
        """
        Returns a list of all parameters (keys) of the object.

        Returns
        -------
        list
            A list of keys representing the parameters of the object.
        """
        return list(self.keys())

    @property
    def objectproperties(self):
        """
        Returns a dictionary of the object's properties, excluding disallowed keywords.

        Returns
        -------
        frameworkObject
            The object itself, allowing for method chaining.
        """
        return self

    # def __getitem__(self, key):
    #     lkey = key.lower()
    #     defaults = self.objectdefaults
    #     if lkey in defaults:
    #         try:
    #             return getattr(self, lkey)
    #         except Exception:
    #             return defaults[lkey]
    #     else:
    #         try:
    #             return getattr(self, lkey)
    #         except Exception:
    #             try:
    #                 return getattr(self, key)
    #             except Exception:
    #                 return None

    def __repr__(self):
        string = ""
        for k in self.model_fields_set:
            if k in self.allowedkeywords:
                string += f"{k} = {getattr(self, k)}" + "\n"
        return string


class frameworkElement(frameworkObject):
    """
    Class defining a framework element, which is a specific type of framework object
    that represents a physical component in a simulation lattice. It extends the frameworkObject
    class with additional properties and methods specific to elements, such as position, rotation,
    and field definitions.
    """

    length: float = 0.0
    """Length of the element in the simulation, typically in meters."""

    centre: List[float] = [0, 0, 0]
    """Centre of the element in the simulation [x,y,z]."""

    position_errors: List[float] = [0, 0, 0]
    """Position errors of the element in the simulation [x,y,z]."""

    rotation_errors: List[float] = [0, 0, 0]
    """Rotation errors of the element in the simulation [x,y,z]."""

    global_rotation: List[float] = [0, 0, 0]
    """Global rotation of the element in the simulation [x,y,z]."""

    rotation: SerializeAsAny[List[float] | float] = [0, 0, 0]
    """Local rotation of the element in the simulation [x,y,z]."""

    starting_rotation: SerializeAsAny[float | List[float]] = 0.0
    """Initial rotation of the element, used for specific simulation setups."""

    conversion_rules_elegant: Dict = {}
    """Conversion rules for keywords when exporting to Elegant format."""

    conversion_rules_ocelot: Dict = {}
    """Conversion rules for keywords when exporting to Ocelot format."""

    starting_offset: List[float] = [0, 0, 0]
    """Initial offset of the element, used for positioning in the simulation."""

    subelement: bool = False
    """Flag indicating whether the element is a sub-element of a larger structure."""

    field_definition: SerializeAsAny[field | str] = None
    """Field definition for the element, can be a field object or a string representing a file."""

    wakefield_definition: SerializeAsAny[field | str] = None
    """Wakefield definition for the element, can be a field object or a string representing a file."""

    def __init__(self, *args, **kwargs):
        super(frameworkElement, self).__init__(
            *args,
            **kwargs,
        )
        self.conversion_rules_elegant = keyword_conversion_rules_elegant["general"]
        self.conversion_rules_ocelot = keyword_conversion_rules_ocelot["general"]
        if self.objecttype in keyword_conversion_rules_elegant:
            self.conversion_rules_elegant = merge_two_dicts(
                keyword_conversion_rules_elegant[self.objecttype],
                keyword_conversion_rules_elegant["general"],
            )
        if self.objecttype in keyword_conversion_rules_ocelot:
            self.conversion_rules_ocelot = merge_two_dicts(
                keyword_conversion_rules_ocelot[self.objecttype],
                keyword_conversion_rules_ocelot["general"],
            )

    def __setattr__(self, name, value):
        # Let Pydantic set known fields normally
        if name in frameworkElement.model_fields:
            super().__setattr__(name, value)
        else:
            try:
                super().__setattr__(name, value)
            except Exception:
                # Store extras in __dict__ (allowed by Config.extra = 'allow')
                self.__dict__[name] = value

    def __mul__(self, other):
        return [self.objectproperties for x in range(other)]

    def __rmul__(self, other):
        return [self.objectproperties for x in range(other)]

    def __neg__(self):
        return self

    def __repr__(self):
        disallowed = [
            "allowedkeywords",
            "conversion_rules_elegant",
            "conversion_rules_ocelot",
            "objectdefaults",
            "global_parameters",
            "objectname",
            "subelement",
        ]
        return repr(
            {k: getattr(self, k) for k in self.model_fields_set if k not in disallowed}
        )

    @field_validator("length", mode="before")
    @classmethod
    def validate_length(cls, value: float) -> float:
        """Validate the length to ensure it is a non-negative float."""
        if not isinstance(value, (int, float)):
            raise ValueError("length must be a float or an int.")
        if value < 0:
            raise ValueError("length must be non-negative.")
        return float(value)

    @property
    def propertiesDict(self) -> dict:
        """
        Returns a dictionary of the object's properties, excluding disallowed keywords.

        Returns
        -------
        dict
            A dictionary containing the object's properties.
        """
        disallowed = [
            "allowedkeywords",
            "conversion_rules_elegant",
            "conversion_rules_ocelot",
            "objectdefaults",
            "global_parameters",
            "subelement",
        ]
        return {
            k: getattr(self, k) for k in self.model_fields_set if k not in disallowed
        }

    @property
    def k1(self) -> None:
        return None

    @property
    def k2(self) -> None:
        return None

    @property
    def k3(self) -> None:
        return None

    @property
    def x(self) -> float:
        """
        Returns the x-coordinate of the element's starting position.

        Returns
        -------
        float
            The x-coordinate of the element's starting position.

        """
        return self.position_start[0]

    @x.setter
    def x(self, x: float) -> None:
        """
        Sets the x-coordinate of the element's starting position.

        Parameters
        ----------
        float
            The x-coordinate of the element's starting position.

        """
        self.position_start[0] = x
        self.position_end[0] = x

    @property
    def y(self) -> float:
        """
        Returns the y-coordinate of the element's starting position.

        Returns
        -------
        float
            The y-coordinate of the element's starting position.

        """
        return self.position_start[1]

    @y.setter
    def y(self, y) -> None:
        """
        Sets the y-coordinate of the element's starting position.

        Parameters
        ----------
        float
            The y-coordinate of the element's starting position.

        """
        self.position_start[1] = y
        self.position_end[1] = y

    @property
    def z(self):
        """
        Returns the z-coordinate of the element's starting position.

        Returns
        -------
        float
            The z-coordinate of the element's starting position.

        """
        return self.position_start[2]

    @z.setter
    def z(self, z) -> None:
        """
        Sets the z-coordinate of the element's starting position.

        Parameters
        ----------
        float
            The z-coordinate of the element's starting position.

        """
        self.position_start[2] = z
        self.position_end[2] = z

    @property
    def dx(self) -> float:
        """
        Returns the x-offset of the element.

        Returns
        -------
        float
            The x-offset of the element.

        """
        return self.position_errors[0]

    @dx.setter
    def dx(self, x: float) -> None:
        """
        Sets the x-offset of the element.

        Parameters
        ----------
        float
            The x-coordinate of the element.

        """
        self.position_errors[0] = x

    @property
    def dy(self):
        """
        Returns the y-offset of the element.

        Returns
        -------
        float
            The y-offset of the element.

        """
        return self.position_errors[1]

    @dy.setter
    def dy(self, y: float) -> None:
        """
        Sets the y-offset of the element.

        Parameters
        ----------
        float
            The y-offset of the element.

        """
        self.position_errors[1] = y

    @property
    def dz(self):
        """
        Returns the z-offset of the element.

        Returns
        -------
        float
            The z-offset of the element.

        """
        return self.position_errors[2]

    @dz.setter
    def dz(self, z: float) -> None:
        """
        Sets the z-offset of the element.

        Parameters
        ----------
        float
            The z-offset of the element.

        """
        self.position_errors[2] = z

    @property
    def x_rot(self) -> float:
        """
        Returns the global x-rotation of the element.

        Returns
        -------
        float
            The global x-rotation of the element.

        """
        return self.global_rotation[1]

    @property
    def y_rot(self) -> float:
        """
        Returns the global y-rotation of the element.

        Returns
        -------
        float
            The global y-rotation of the element.

        """
        return self.global_rotation[2] + self.starting_rotation

    @property
    def z_rot(self) -> float:
        """
        Returns the global z-rotation of the element.

        Returns
        -------
        float
            The global z-rotation of the element.

        """
        return self.global_rotation[0]

    @property
    def dx_rot(self) -> float:
        """
        Returns the local x-rotation of the element.

        Returns
        -------
        float
            The local x-rotation of the element.

        """
        return self.rotation_errors[1]

    @dx_rot.setter
    def dx_rot(self, x: float) -> None:
        """
        Sets the x-rotation error of the element.

        Parameters
        ----------
        float
            The x-rotation error of the element.

        """
        self.rotation_errors[1] = x

    @property
    def dy_rot(self):
        """
        Returns the local y-rotation of the element.

        Returns
        -------
        float
            The local y-rotation of the element.

        """
        return self.rotation_errors[2]

    @dy_rot.setter
    def dy_rot(self, y: float) -> None:
        """
        Sets the y-rotation error of the element.

        Parameters
        ----------
        float
            The y-rotation error of the element.

        """
        self.rotation_errors[2] = y

    @property
    def dz_rot(self):
        """
        Returns the local z-rotation of the element.

        Returns
        -------
        float
            The local z-rotation of the element.

        """
        return self.rotation_errors[0]

    @dz_rot.setter
    def dz_rot(self, z: float) -> None:
        """
        Sets the z-rotation error of the element.

        Parameters
        ----------
        float
            The z-rotation error of the element.

        """
        self.rotation_errors[0] = z

    @property
    def tilt(self):
        """
        Returns the local z-rotation of the element.

        Returns
        -------
        float
            The local z-rotation of the element.

        """
        return self.dz_rot

    @property
    def PV(self) -> str:
        """
        Returns the PV root name of the element, which is used for identifying the element in a control system.

        Returns
        -------
        str
            The PV root name of the element, which is either the `PV_root` attribute or the `objectName`.

        """
        if hasattr(self, "PV_root"):
            return self.PV_root
        else:
            return self.objectname

    @property
    def get_field_amplitude(self) -> float:
        """
        Returns the field amplitude of the element, scaled by `field_scale` if it exists.

        Returns
        -------
        float
            The field amplitude of the element, which is either scaled by `field_scale`
            or directly taken from `field_amplitude`.

        """
        if hasattr(self, "field_scale") and isinstance(self.field_scale, (int, float)):
            return float(self.field_scale) * float(
                expand_substitution(self, self.field_amplitude)
            )
        else:
            return float(expand_substitution(self, self.field_amplitude))

    def get_field_reference_position(self) -> list:
        """
        Returns the position of the field reference point based on the `field_reference_position` attribute.

        Returns
        -------
        list
            The position of the field reference point, which can be 'start', 'middle', or 'end'.
            If `field_reference_position` is not set, it defaults to the start position.

        Raises
        ------
        ValueError
            If `field_reference_position` is set to an invalid value that is not 'start', 'middle', or 'end'.
        """
        if (
            hasattr(self, "field_reference_position")
            and self.field_reference_position is not None
        ):
            if self.field_reference_position.lower() == "start":
                return self.start
            elif self.field_reference_position.lower() == "middle":
                return self.middle
            elif self.field_reference_position.lower() == "end":
                return self.end
            else:
                raise ValueError(
                    "field_reference_position should be (start/middle/end) not",
                    self.field_reference_position,
                )
        else:
            return self.start

    @property
    def theta(self) -> float:
        """
        Returns the global rotation angle of the element in radians.

        Returns
        -------
        float
            The global rotation angle of the element, which is derived from `global_rotation`.
            If `global_rotation` is not set, it defaults to 0 radians.
        """
        if hasattr(self, "global_rotation") and self.global_rotation is not None:
            rotation = (
                self.global_rotation[2]
                if len(self.global_rotation) == 3
                else self.global_rotation
            )
        else:
            rotation = 0
        # if hasattr(self, 'starting_rotation') and self.starting_rotation is not None:
        #     rotation +=  self.starting_rotation
        return rotation

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Returns the rotation matrix for the element based on its global rotation angle :attr:`theta`.

        Returns
        -------
        np.ndarray
            The rotation matrix corresponding to the global rotation angle of the element.
        """
        return _rotation_matrix(self.theta)

    def rotated_position(
        self, pos: tuple = (0, 0, 0), offset: list = None, theta: float = None
    ) -> int | float | complex | list:
        """
        Returns the position of the element after applying a rotation and an offset.

        Parameters
        ----------
        pos: tuple, optional
            A tuple representing the position to be rotated. Default is (0, 0, 0).
        offset: list, optional
            A list representing the offset to be applied to the position. If not provided,
            it defaults to the element's starting offset or [0, 0, 0] if not set.
        theta: float, optional
            The rotation angle in radians to be applied to the position. If not provided,
            it defaults to the element's global rotation angle.

        Returns
        -------
        int | float | complex | list
            The rotated position of the element, adjusted for the specified offset and rotation angle.
            If `offset` is not provided, it uses the element's `starting_offset` or defaults to [0, 0, 0].
        """
        if offset is None:
            if not hasattr(self, "starting_offset") or self.starting_offset is None:
                offset = [0, 0, 0]
            else:
                offset = self.starting_offset
        if theta is None:
            return chop(
                np.dot(np.array(pos) - np.array(offset), self.rotation_matrix), 1e-6
            )
        else:
            return chop(
                np.dot(np.array(pos) - np.array(offset), _rotation_matrix(theta)), 1e-6
            )

    @property
    def start(self) -> list:
        """
        Returns the starting position of the element, which is calculated based on its center and length,
        see :attr:`position_start`.

        Returns
        -------
        list
            The starting position of the element, which is the position at the beginning of the element's length.

        """
        return self.position_start

    @property
    def position_start(self) -> list:
        """
        Returns the starting position of the element, which is calculated based on its center and length.

        Returns
        -------
        list
            The starting position of the element, which is the position at the beginning of the element's length.
        """
        middle = np.array(self.centre)
        start = middle - self.rotated_position(
            (0, 0, self.length / 2.0),
            offset=self.starting_offset,
            theta=self.y_rot,
        )
        return list(start)

    @property
    def middle(self) -> list:
        """
        Returns the middle position of the element, which is the center of the element's length.

        Returns
        -------
        list
            The middle position of the element, which is the center point of the element's length.
        """
        return self.centre

    @property
    def end(self) -> list:
        """
        Returns the end position of the element, which is calculated based on its starting position and length.

        Returns
        -------
        list
            The end position of the element, which is the position at the end of the element's length.
        """
        return self.position_end

    @property
    def position_end(self) -> list:
        """
        Returns the end position of the element, which is calculated based on its starting position and length.

        Returns
        -------
        list
            The end position of the element, which is the position at the end of the element's length.
        """
        start = np.array(self.position_start)
        end = start + self.rotated_position(
            (0, 0, self.length),
            offset=self.starting_offset,
            theta=self.y_rot,
        )
        return end

    def relative_position_from_centre(self, vec: tuple = (0, 0, 0)) -> list:
        """
        Returns the position relative to the centre of the element,
        taking into account the element's rotation and offset.

        Parameters
        ----------
        vec: tuple, optional
            A tuple representing the vector to be added to the centre position.

        Returns
        -------
        list
            The position relative to the centre of the element, adjusted for rotation and offset.
        """
        middle = np.array(self.centre)
        return list(
            middle
            + self.rotated_position(
                vec,
                offset=self.starting_offset,
                theta=self.y_rot,
            )
        )

    def relative_position_from_start(self, vec: tuple = (0, 0, 0)) -> list:
        """
        Returns the position relative to the start of the element,

        Parameters
        ----------
        vec: tuple, optional
            A tuple representing the vector to be added to the start position.

        Returns
        -------
        list
            The position relative to the start of the element, adjusted for rotation and offset.
        """
        start = np.array(self.position_start)
        return list(
            start
            + self.rotated_position(
                vec,
                offset=self.starting_offset,
                theta=self.y_rot,
            )
        )

    def update_field_definition(self) -> None:
        """
        Updates the field definitions to allow for the relative sub-directory location
        """
        if (
            hasattr(self, "field_definition")
            and self.field_definition is not None
            and isinstance(self.field_definition, str)
        ):
            field_kwargs = {
                "filename": expand_substitution(self, self.field_definition),
                "field_type": self.field_type,
            }
            if self.objecttype == "cavity":
                field_kwargs.update(
                    {
                        "frequency": self.frequency,
                        "cavity_type": self.Structure_Type,
                        "n_cells": self.n_cells,
                    }
                )
            self.field_definition = field(**field_kwargs)
        if (
            hasattr(self, "wakefield_definition")
            and self.wakefield_definition is not None
            and isinstance(self.wakefield_definition, str)
        ):
            self.wakefield_definition = field(
                filename=expand_substitution(self, self.wakefield_definition),
                field_type=self.field_type,
                frequency=self.frequency,
                cavity_type=self.Structure_Type,
                n_cells=self.n_cells,
            )

    def _write_ASTRA_dictionary(self, d: dict, n: int = 1) -> str:
        """
        Generates a string representation of the object's properties in the ASTRA format.

        Parameters
        ----------
        d: dict
            A dictionary containing the properties of the object to be formatted.
        n: int, optional
            An optional integer to specify the index for ASTRA objects. Default is 1.

        Returns
        -------
        str
            A formatted string representing the object's properties in ASTRA format.
        """
        output = ""
        for k, v in list(d.items()):
            if checkValue(self, v) is not None:
                if "type" in v and v["type"] == "list":
                    for i, l in enumerate(checkValue(self, v)):
                        if n is not None:
                            param_string = (
                                k
                                + "("
                                + str(i + 1)
                                + ","
                                + str(n)
                                + ") = "
                                + str(l)
                                + ", "
                            )
                        else:
                            param_string = k + " = " + str(l) + "\n"
                        if len((output + param_string).splitlines()[-1]) > 70:
                            output += "\n"
                        output += param_string
                elif "type" in v and v["type"] == "array":
                    if n is not None:
                        param_string = k + "(" + str(n) + ") = ("
                    else:
                        param_string = k + " = ("
                    for i, l in enumerate(checkValue(self, v)):
                        param_string += str(l) + ", "
                        if len((output + param_string).splitlines()[-1]) > 70:
                            output += "\n"
                    output += param_string[:-2] + "),\n"
                elif "type" in v and v["type"] == "not_zero":
                    if abs(checkValue(self, v)) > 0:
                        if n is not None:
                            param_string = (
                                k
                                + "("
                                + str(n)
                                + ") = "
                                + str(checkValue(self, v))
                                + ", "
                            )
                        else:
                            param_string = k + " = " + str(checkValue(self, v)) + ",\n"
                        if len((output + param_string).splitlines()[-1]) > 70:
                            output += "\n"
                        output += param_string
                else:
                    if n is not None:
                        param_string = (
                            k + "(" + str(n) + ") = " + str(checkValue(self, v)) + ", "
                        )
                    else:
                        param_string = k + " = " + str(checkValue(self, v)) + ",\n"
                    if len((output + param_string).splitlines()[-1]) > 70:
                        output += "\n"
                    output += param_string
        return output[:-2]

    def write_ASTRA(self, n, **kwargs) -> str:
        """
        Generates a string representation of the object's properties in the ASTRA format.

        Parameters
        ----------
        n: int
            An integer representing the index for ASTRA objects,
            typically used for multiple instances of the same element.
        **kwargs: dict
            Additional keyword arguments that can be used to pass extra parameters.

        Returns
        -------
        str
            A formatted string representing the object's properties in ASTRA format.
        """
        return self._write_ASTRA(n, **kwargs)

    def generate_field_file_name(self, param: field, code: str) -> str | None:
        """
        Generates a field file name based on the provided frameworkElement and tracking code.

        Parameters
        ----------
        param: field
            The :class:`SimulationFramework.Modules.Fields.field` object for which the field file is being generated.
        code: str
            The tracking code for which the field file is being generated (e.g., 'elegant', 'ocelot').

        Returns
        -------
        str | None
            The name of the field file if it exists, otherwise None.
        """
        if hasattr(param, "filename"):
            basename = (
                os.path.basename(param.filename).replace('"', "").replace("'", "")
            )
            efield_basename = os.path.abspath(
                self.global_parameters["master_subdir"].replace("\\", "/")
                + "/"
                + basename.replace("\\", "/")
            )
            return os.path.basename(
                param.write_field_file(code=code, location=efield_basename)
            )
        else:
            warn(
                f"param does not have a filename: {param}, it must be a `field` object"
            )
        return None

    def _write_Elegant(self) -> str:
        """
        Generates a string representation of the object's properties in the Elegant format.

        Returns
        -------
        str
            A formatted string representing the object's properties in Elegant format.
        """
        wholestring = ""
        etype = self._convertType_Elegant(self.objecttype)
        string = self.objectname + ": " + etype
        setattr(self, "k1", self.k1 if self.k1 is not None else 0)
        setattr(self, "k2", self.k2 if self.k2 is not None else 0)
        setattr(self, "k3", self.k3 if self.k3 is not None else 0)
        for key, value in self.objectproperties:
            if (
                not key == "name"
                and not key == "type"
                and not key == "commandtype"
                and self._convertKeyword_Elegant(key) in elements_Elegant[etype]
            ):
                value = (
                    getattr(self, key)
                    if hasattr(self, key) and getattr(self, key) is not None
                    else value
                )
                key = self._convertKeyword_Elegant(key)
                value = 1 if value is True else value
                value = 0 if value is False else value
                tmpstring = ", " + key + " = " + str(value)
                if len(string + tmpstring) > 76:
                    wholestring += string + ",&\n"
                    string = ""
                    string += tmpstring[2::]
                else:
                    string += tmpstring
        wholestring += string + ";\n"
        return wholestring

    def write_Elegant(self) -> str:
        """
        Generates a string representation of the object's properties in the Elegant format,
        see :func:`_write_Elegant`.

        Returns
        -------
        str
            A formatted string representing the object's properties in Elegant format.
        """
        if not self.subelement:
            return self._write_Elegant()

    def _convertType_Elegant(self, etype: str) -> str:
        """
        Converts the element type to the corresponding Elegant type using predefined rules.

        Parameters
        ----------
        etype: str
            The type of the element to be converted.

        Returns
        -------
        str
            The converted type of the element, or the original type if no conversion rule exists.
        """
        return (
            type_conversion_rules_Elegant[etype]
            if etype in type_conversion_rules_Elegant
            else etype
        )

    def _convertKeyword_Elegant(self, keyword: str) -> str:
        """
        Converts a keyword to its corresponding Elegant keyword using predefined rules.

        Parameters
        ----------
        keyword: str:
            The keyword to be converted.

        Returns
        -------
        str
            The converted keyword for Elegant, or the original keyword if no conversion rule exists.

        """
        return (
            self.conversion_rules_elegant[keyword]
            if keyword in self.conversion_rules_elegant
            else keyword
        )

    def _write_Ocelot(self) -> object:
        """
        Generates an Ocelot object based on the element's properties and type.

        Returns
        -------
        object
            An Ocelot object representing the element, initialized with its properties.
        """
        from ocelot.cpbd.elements import Marker, Aperture
        from .Codes.Ocelot import ocelot_conversion

        type_conversion_rules_Ocelot = ocelot_conversion.ocelot_conversion_rules
        obj = type_conversion_rules_Ocelot[self.objecttype](eid=self.objectname)
        setattr(self, "k1", self.k1 if self.k1 is not None else 0)
        setattr(self, "k2", self.k2 if self.k2 is not None else 0)
        setattr(self, "k3", self.k3 if self.k3 is not None else 0)
        for key, value in self.objectproperties:
            if (key not in ["name", "type", "commandtype"]) and (
                not type(obj) in [Aperture, Marker]
            ):
                value = (
                    getattr(self, key)
                    if hasattr(self, key) and getattr(self, key) is not None
                    else value
                )
                setattr(obj, self._convertKeyword_Ocelot(key), value)
        return obj

    def write_Ocelot(self) -> object:
        """
        Generates an Ocelot object based on the element's properties and type,
        see :func:`_write_Ocelot`.

        Returns
        -------
        object
            An Ocelot object representing the element, initialized with its properties.
        """
        if not self.subelement:
            return self._write_Ocelot()

    def _convertType_Ocelot(self, etype: str) -> object:
        """
        Converts the element type to the corresponding Ocelot type using predefined rules.

        Parameters
        ----------
        etype: str
            The type of the element to be converted.

        Returns
        -------
        object
            The Ocelot element, or the original type if no conversion rule exists.
        """
        from .Codes.Ocelot import ocelot_conversion

        type_conversion_rules_Ocelot = ocelot_conversion.ocelot_conversion_rules
        return (
            type_conversion_rules_Ocelot[etype]
            if etype in type_conversion_rules_Ocelot
            else etype
        )

    def _convertKeyword_Ocelot(self, keyword: str) -> str:
        """
        Converts a keyword to its corresponding Ocelot keyword using predefined rules.

        Parameters
        ----------
        keyword: str
            The keyword to be converted.

        Returns
        -------
        str
            The converted keyword for Ocelot, or the original keyword if no conversion rule exists.
        """
        return (
            self.conversion_rules_ocelot[keyword]
            if keyword in self.conversion_rules_ocelot
            else keyword
        )

    def _write_CSRTrack(self, n: int = 0, **kwargs: dict) -> str:
        pass

    def write_CSRTrack(self, n: int = 0, **kwargs: dict) -> str:
        return self._write_CSRTrack(n, **kwargs)

    def write_GPT(self, Brho: float, ccs: str = "wcs", *args, **kwargs) -> str:
        return self._write_GPT(Brho, ccs, *args, **kwargs)

    def _write_GPT(self, Brho: float, ccs: str = "wcs", *args, **kwargs) -> str:
        return ""

    def gpt_coordinates(self, position: list, rotation: float) -> str:
        """
        Get the GPT coordinates for a given position and rotation

        Parameters
        ----------
        position: list
            The lattice position.
        rotation: float
            The element rotation

        Returns
        -------
        str
            A GPT-formatted position string.
        """
        x, y, z = chop(position, 1e-6)
        psi, phi, theta = rotation
        output = ""
        for c in [-x, y, z]:
            output += str(c) + ", "
        output += "cos(" + str(theta) + "), 0, -sin(" + str(theta) + "), 0, 1 ,0"
        return output

    def gpt_ccs(self, ccs: str) -> str:
        """
        Get the GPT coordinate system for the element.

        Parameters
        ----------
        ccs: str
            The GPT coordinate system.

        Returns
        -------
        str
            The GPT coordinate system

        """
        return ccs

    def array_names_string(self) -> str:
        """
        Get the array names for a given element (i.e. the parameters in the field file)

        Returns
        -------
        str
            A formatted string containing the array names for the element.
        """
        array_names = (
            self.default_array_names if self.array_names is None else self.array_names
        )
        return ", ".join(['"' + name + '"' for name in array_names])


class csrdrift(frameworkElement):
    """
    Class defining a drift including CSR effects.
    """

    lsc_interpolate: int = 1
    """Flag to allow for interpolation of computed longitudinal space charge wake.
    See `Elegant manual`_"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(csrdrift, self).__init__(
            *args,
            **kwargs,
        )

    def _write_Elegant(self) -> str:
        """
        Writes the csrdrift element string for ELEGANT.

        Returns
        -------
        str
            String representation of the element for ELEGANT
        """
        wholestring = ""
        etype = self._convertType_Elegant(self.objecttype)
        string = self.objectname + ": " + etype
        for key, value in self.objectproperties:
            if (
                not key == "name"
                and not key == "type"
                and not key == "commandtype"
                and self._convertKeyword_Elegant(key) in elements_Elegant[etype]
            ):

                value = (
                    getattr(self, key)
                    if hasattr(self, key) and getattr(self, key) is not None
                    else value
                )
                key = self._convertKeyword_Elegant(key)
                value = 1 if value is True else value
                value = 0 if value is False else value
                tmpstring = ", " + key + " = " + str(value)
                if len(string + tmpstring) > 76:
                    wholestring += string + ",&\n"
                    string = ""
                    string += tmpstring[2::]
                else:
                    string += tmpstring
        wholestring += string + ";\n"
        return wholestring


class lscdrift(csrdrift):
    """
    Class defining a drift including LSC effects.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(lscdrift, self).__init__(
            *args,
            **kwargs,
        )


class edrift(csrdrift):
    """
    Class defining a drift.
    """

    def __init__(self, *args, **kwargs):
        super(edrift, self).__init__(
            *args,
            **kwargs,
        )


class frameworkLattice(BaseModel):
    """
    Class defining a framework lattice object, which contains all elements and groups
    of elements in a simulation lattice. It also contains methods to manipulate and
    retrieve information about the elements and groups, as well as methods to run
    simulations and process results.

    See :ref:`creating-the-lattice-elements`
    """

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True
        validate_assignment = True

    name: str
    """Name of the lattice, used as a prefix for output files and commands."""

    file_block: Dict
    """File block containing input and output settings for the lattice."""

    elementObjects: Dict
    """Dictionary of element objects, where keys are element names and values are element instances."""

    groupObjects: Dict
    """Dictionary of group objects, where keys are group names and values are group instances."""

    runSettings: runSetup
    """Run settings for the lattice, including number of runs and random seed."""

    settings: Dict
    """Settings for the lattice, including global and group-specific settings."""

    executables: exes.Executables
    """Executable commands for running simulations, defined in the Executables class.
    See :class:`SimulationFramework.Framework.Codes.Executables.Executables` for more details."""

    global_parameters: Dict
    """Global parameters for the lattice, including master subdirectory and other configuration settings."""

    allow_negative_drifts: bool = False
    """If True, allows negative drifts in the lattice."""

    _csr_enable: bool = True
    """Flag to enable CSR drifts in the lattice."""

    csrDrifts: bool = True
    """Flag to enable CSR drifts in the lattice."""

    lscDrifts: bool = True
    """Flag to enable LSC drifts in the lattice."""

    lsc_bins: int = 20
    """Number of bins for LSC drifts."""

    lsc_high_frequency_cutoff_start: float = -1
    """Spatial frequency at which smoothing filter begins. If not positive, no frequency filter smoothing is done. 
    See `Elegant manual`_
    
    .. _Elegant manual: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu168.html#x179-18000010.58"""

    lsc_high_frequency_cutoff_end: float = -1
    """Spatial frequency at which smoothing filter is 0. See `Elegant manual`_"""

    lsc_low_frequency_cutoff_start: float = -1
    """Highest spatial frequency at which low-frequency cutoff filter is zero. See `Elegant manual`_"""

    lsc_low_frequency_cutoff_end: float = -1
    """Lowest spatial frequency at which low-frequency cutoff filter is 1. See `Elegant manual`_"""

    sample_interval: int = 1
    """Sample interval for downsampling particles, in units of 2**(3*sample_interval)"""

    globalSettings: Dict = {"charge": None}
    """Global settings for the lattice, including charge and other parameters."""

    groupSettings: Dict = {}
    """Group settings for the lattice, including group-specific parameters."""

    allElements: List = []
    """List of all element names in the lattice."""

    initial_twiss: Dict = {}
    """Initial Twiss parameters for the lattice, used for tracking and analysis."""

    def __init__(
        self,
        name,
        file_block,
        elementObjects,
        groupObjects,
        runSettings,
        settings,
        executables,
        global_parameters,
        *args,
        **kwargs,
    ):
        super(frameworkLattice, self).__init__(
            name=name,
            file_block=file_block,
            elementObjects=elementObjects,
            groupObjects=groupObjects,
            runSettings=runSettings,
            settings=settings,
            executables=executables,
            global_parameters=global_parameters,
            *args,
            **kwargs,
        )
        for key, value in list(self.elementObjects.items()):
            setattr(self, key, value)
        self.allElements = list(self.elementObjects.keys())
        self.objectname = self.name

        # define settings for simulations with multiple runs
        self.updateRunSettings(runSettings)

    @field_validator("file_block", mode="before")
    @classmethod
    def validate_file_block(cls, value: Dict) -> Dict:
        """
        Validate the file_block dictionary to ensure it has the required structure.
        This method checks if the file_block is a dictionary and contains the necessary keys.

        Raises
        ------
        ValueError
            If the file_block is not a dictionary or does not contain the required keys.
        """
        if not isinstance(value, dict):
            raise ValueError("file_block must be a dictionary.")
        if "groups" in value:
            if value["groups"] is not None:
                cls.groupSettings = value["groups"]
        if "input" in value:
            if "sample_interval" in value["input"]:
                cls.sample_interval = value["input"]["sample_interval"]
        return value

    @field_validator("settings", mode="before")
    @classmethod
    def validate_settings(cls, value: Dict) -> Dict:
        """
        Validate the settings dictionary to ensure it has the required structure.
        This method checks if the settings is a dictionary and contains the necessary keys.

        Raises
        ------
        ValueError
            If the settings is not a dictionary or does not contain the required keys.

        """
        if not isinstance(value, dict):
            raise ValueError("settings must be a dictionary.")
        if "global" in value:
            if value["global"] is not None:
                cls.globalSettings = value["global"]
        return value

    def __setattr__(self, name, value):
        # Let Pydantic set known fields normally
        if name in frameworkLattice.model_fields:
            super().__setattr__(name, value)
        else:
            try:
                super().__setattr__(name, value)
            except Exception:
                # Store extras in __dict__ (allowed by Config.extra = 'allow')
                self.__dict__[name] = value

    def insert_element(self, index: int, element) -> None:
        """
        Insert an element at a specific index in the elements dictionary.

        Parameters
        ----------
        index: int
            The index at which to insert the element.
        element: :class:`SimulationFramework.Framework_objects.frameworkElement`
            The element to insert into the elements dictionary.

        """
        for i, _ in enumerate(range(len(self.elements))):
            k, v = self.elements.popitem(False)
            self.elements[element.objectname if i == index else k] = element

    @property
    def csr_enable(self) -> bool:
        """
        Property to get or set the CSR enable flag.
        """
        return self._csr_enable

    @csr_enable.setter
    def csr_enable(self, csr) -> None:
        print(1)
        self.csrDrifts = csr
        self._csr_enable = csr

    def get_prefix(self) -> str:
        """
        Get the prefix from the input file block.

        Returns
        -------
        str
            The prefix string used in the input file block.
        """
        if "input" not in self.file_block:
            self.file_block["input"] = {}
        if "prefix" not in self.file_block["input"]:
            self.file_block["input"]["prefix"] = ""
        return self.file_block["input"]["prefix"]

    def set_prefix(self, prefix: str) -> None:
        """
        Set the prefix for the input file block.

        Parameters
        ----------
        prefix: str
            The prefix string used in the input file block.
        """
        if not hasattr(self, "file_block") or self.file_block is None:
            self.file_block = {}
        if "input" not in self.file_block or self.file_block["input"] is None:
            self.file_block["input"] = {}
        self.file_block["input"]["prefix"] = prefix

    @computed_field
    @property
    def prefix(self) -> str:
        return self.get_prefix()

    @prefix.setter
    def prefix(self, prefix: str) -> None:
        self.set_prefix(prefix)

    def update_groups(self) -> None:
        """
        Update the group objects in the lattice with their settings.
        """
        for g in list(self.groupSettings.keys()):
            if g in self.groupObjects:
                setattr(self, g, self.groupObjects[g])
                if self.groupSettings[g] is not None:
                    self.groupObjects[g].update(**self.groupSettings[g])

    def getElement(self, element: str, param: str = None) -> dict | frameworkElement:
        """
        Get an element or group object by its name and optionally a specific parameter.
        This method checks if the element exists in the allElements dictionary or in the groupObjects dictionary.
        If the element exists, it returns the element object or the specified parameter of the element.

        Parameters
        ----------
        element: str
        param: str, optional
            The parameter to retrieve from the element object. If None, returns the entire element object.

        Returns
        -------
        dict | :class:`SimulationFramework.Framework_objects.frameworkElement`
            The element object or the specified parameter of the element.
        """
        if element in self.allElements:
            if param is not None:
                return getattr(self.elementObjects[element], param.lower())
            else:
                return self.allElements[element]
        elif element in list(self.groupObjects.keys()):
            if param is not None:
                return getattr(self.groupObjects[element], param.lower())
            else:
                return self.groupObjects[element]
        else:
            warn(f"WARNING: Element {element} does not exist")
            return {}

    def getElementType(
        self,
        typ: list | tuple | str,
        param: list | tuple | str = None,
    ) -> list | tuple | zip:
        """
        Get all elements of a specific type or types from the lattice.

        Parameters
        ----------
        typ: list, tuple, or str
            The type or types of elements to retrieve.
            If a list or tuple is provided, it retrieves elements of all specified types.
        param: list, tuple, or str, optional
            The specific parameter to retrieve from each element.

        Returns
        -------
        list | tuple | zip
            A list or tuple of elements of the specified type(s), or a zip object if multiple parameters are specified.
            If `param` is provided, it returns the specified parameter for each element.
        """
        if isinstance(typ, (list, tuple)):
            return [self.getElementType(t, param=param) for t in typ]
        if isinstance(param, (list, tuple)):
            return zip(*[self.getElementType(typ, param=p) for p in param])
        return [
            self.elements[element] if param is None else self.elements[element][param]
            for element in list(self.elements.keys())
            if self.elements[element].objecttype.lower() == typ.lower()
        ]

    def setElementType(
        self, typ: list | tuple | str, setting: str, values: list | tuple | Any
    ) -> None:
        """
        Set a specific setting for all elements of a specific type or types in the lattice.

        Parameters
        ----------
        typ: list, tuple, or str
            The type or types of elements to set the setting for.
        setting: str
            The setting to be updated for the elements. This can be a single setting or a list of settings.
        values: list, tuple, or Any
            The values to set for the specified setting.

        Raises
        ------
        ValueError
            If the number of elements of the specified type does not match the number of values provided.
        """
        elems = self.getElementType(typ)
        if len(elems) == len(values):
            for e, v in zip(elems, values):
                e[setting] = v
        else:
            raise ValueError

    @property
    def quadrupoles(self) -> list:
        """
        Property to get all quadrupole elements in the lattice.

        Returns
        -------
        list
            A list of quadrupole elements in the lattice.
        """
        return self.getElementType("quadrupole")

    @property
    def cavities(self) -> list:
        """
        Property to get all cavity elements in the lattice.

        Returns
        -------
        list
            A list of cavity elements in the lattice.
        """
        return self.getElementType("cavity")

    @property
    def solenoids(self) -> list:
        """
        Property to get all solenoid elements in the lattice.

        Returns
        -------
        list
            A list of solenoid elements in the lattice.
        """
        return self.getElementType("solenoid")

    @property
    def dipoles(self) -> list:
        """
        Property to get all dipole elements in the lattice.

        Returns
        -------
        list
            A list of dipole elements in the lattice.
        """
        return self.getElementType("dipole")

    @property
    def kickers(self) -> list:
        """
        Property to get all kicker elements in the lattice.

        Returns
        -------
        list
            A list of kicker elements in the lattice.
        """
        return self.getElementType("kicker")

    @property
    def dipoles_and_kickers(self) -> list:
        """
        Property to get all dipole and kicker elements in the lattice.

        Returns
        -------
        list
            A list of dipole and kicker elements in the lattice.
        """
        return sorted(
            self.getElementType("dipole") + self.getElementType("kicker"),
            key=lambda x: x.position_end[2],
        )

    @property
    def wakefields(self) -> list:
        """
        Property to get all wakefield elements in the lattice.

        Returns
        -------
        list
            A list of wakefield elements in the lattice.
        """
        return self.getElementType("wakefield")

    @property
    def wakefields_and_cavity_wakefields(self) -> list:
        """
        Property to get all wakefield and cavity wakefield elements in the lattice.

        Returns
        -------
        list
            A list of wakefield and cavity wakefield elements in the lattice.
        """
        cavities = [
            cav
            for cav in self.getElementType("cavity")
            if (
                isinstance(cav.longitudinal_wakefield, field)
                or cav.longitudinal_wakefield != ""
            )
            or (
                isinstance(cav.transverse_wakefield, field)
                or cav.transverse_wakefield != ""
            )
            or (
                isinstance(cav.wakefield_definition, field)
                or cav.wakefield_definition != ""
            )
        ]
        wakes = self.getElementType("wakefield")
        return cavities + wakes

    @property
    def screens(self) -> list:
        """
        Property to get all screen elements in the lattice.

        Returns
        -------
        list
            A list of screen elements in the lattice.
        """
        return self.getElementType("screen")

    @property
    def screens_and_bpms(self) -> list:
        """
        Property to get all screen and BPM elements in the lattice.

        Returns
        -------
        list
            A list of screen and BPM elements in the lattice.
        """
        return sorted(
            self.getElementType("screen")
            + self.getElementType("beam_position_monitor"),
            key=lambda x: x.position_start[2],
        )

    @property
    def screens_and_markers_and_bpms(self) -> list:
        """
        Property to get all screen and BPM and marker elements in the lattice.

        Returns
        -------
        list
            A list of screen and BPM and marker elements in the lattice.
        """
        return sorted(
            self.getElementType("screen")
            + self.getElementType("marker")
            + self.getElementType("beam_position_monitor"),
            key=lambda x: x.position_start[2],
        )

    @property
    def apertures(self) -> list:
        """
        Property to get all aperture and collimator elements in the lattice.

        Returns
        -------
        list
            A list of aperture and collimator elements in the lattice.
        """
        return sorted(
            self.getElementType("aperture") + self.getElementType("collimator"),
            key=lambda x: x.position_start[2],
        )

    @property
    def lines(self) -> list:
        """
        Property to get all lines in the lattice.

        Returns
        -------
        list
            A list of lines in the lattice.
        """
        return list(self.lineObjects.keys())

    @property
    def start(self) -> frameworkElement:
        """
        Property to get the starting element of the lattice.
        This method checks if the file block contains a "start_element" key or a "zstart" key.
        If "start_element" is present, it returns the corresponding element.
        If "zstart" is present, it iterates through the elementObjects to find the element
        with the matching start position. If no match is found, it returns the first element in the elementObjects.


        Returns
        -------
        frameworkElement
            The starting element of the lattice.
        """
        if "start_element" in self.file_block["output"]:
            return self.file_block["output"]["start_element"]
        elif "zstart" in self.file_block["output"]:
            for e in list(self.elementObjects.keys()):
                if (
                    self.elementObjects[e].position_start[2]
                    == self.file_block["output"]["zstart"]
                ):
                    return e
                return self.elementObjects[0]
        else:
            return self.elementObjects[0]

    @property
    def startObject(self) -> frameworkElement:
        """
        Property to get the starting element of the lattice.
        See :func:`start` for more details.


        Returns
        -------
        frameworkElement
            The starting element of the lattice.
        """
        return self.elementObjects[self.start]

    @property
    def end(self) -> frameworkElement:
        """
        Property to get the ending element of the lattice.
        This method checks if the file block contains an "end_element" key or a "zstop" key.
        If "end_element" is present, it returns the corresponding element.
        If "zstop" is present, it iterates through the elementObjects to find the element
        with the matching end position. If no match is found, it returns the last element in the elementObjects.


        Returns
        -------
        frameworkElement
            The final element of the lattice.
        """
        if "end_element" in self.file_block["output"]:
            return self.file_block["output"]["end_element"]
        elif "zstop" in self.file_block["output"]:
            endelems = []
            for e in list(self.elementObjects.keys()):
                if (
                    self.elementObjects[e]["position_end"]
                    == self.file_block["output"]["zstop"]
                ):
                    endelems.append(e)
                elif (
                    self.elementObjects[e]["position_end"]
                    > self.file_block["output"]["zstop"]
                    and len(endelems) == 0
                ):
                    endelems.append(e)
            return endelems[-1]
        else:
            return self.elementObjects[0]

    @property
    def endObject(self) -> frameworkElement:
        """
        Property to get the final element of the lattice.
        See :func:`end` for more details.


        Returns
        -------
        frameworkElement
            The final element of the lattice.
        """
        return self.elementObjects[self.end]

    @property
    def elements(self) -> dict:
        """
        Property to get a dictionary of elements in the lattice.

        Returns
        -------
        dict
            A dictionary where keys are element names and values are the corresponding element objects.
        """
        index_start = self.allElements.index(self.start)
        index_end = self.allElements.index(self.end)
        f = dict(
            [
                [e, self.elementObjects[e]]
                for e in self.allElements[index_start : index_end + 1]
            ]
        )
        return f

    def write(self):
        pass

    def run(self) -> None:
        """
        Run the code with input 'filename'
        This method constructs the command to run the simulation using the specified executable
        and the name of the lattice. It redirects the output to a log file in the master subdirectory.

        Raises
        ------
        FileNotFoundError
            If the executable for the specified code is not found in the executables dictionary.
        """
        command = self.executables[self.code] + [self.name]
        with open(
            os.path.relpath(
                self.global_parameters["master_subdir"] + "/" + self.name + ".log",
                ".",
            ),
            "w",
        ) as f:
            subprocess.call(
                command, stdout=f, cwd=self.global_parameters["master_subdir"]
            )

    def getInitialTwiss(self) -> dict:
        """
        Get the initial Twiss parameters from the file block
        This method checks if the file block contains an "input" key with a "twiss" subkey.
        If the "twiss" subkey exists and contains values, it retrieves the alpha, beta, and normalized emittance
        parameters for both horizontal and vertical planes.

        Returns
        -------
        dict
            A dictionary containing the initial Twiss parameters for horizontal and vertical planes.
            If the parameters are not found, it returns False for each parameter.
        """
        if (
            "input" in self.file_block
            and "twiss" in self.file_block["input"]
            and self.file_block["input"]["twiss"]
        ):
            alpha_x = (
                self.file_block["input"]["twiss"]["alpha_x"]
                if "alpha_x" in self.file_block["input"]["twiss"]
                else False
            )
            alpha_y = (
                self.file_block["input"]["twiss"]["alpha_y"]
                if "alpha_y" in self.file_block["input"]["twiss"]
                else False
            )
            beta_x = (
                self.file_block["input"]["twiss"]["beta_x"]
                if "beta_x" in self.file_block["input"]["twiss"]
                else False
            )
            beta_y = (
                self.file_block["input"]["twiss"]["beta_y"]
                if "beta_y" in self.file_block["input"]["twiss"]
                else False
            )
            nemit_x = (
                self.file_block["input"]["twiss"]["nemit_x"]
                if "nemit_x" in self.file_block["input"]["twiss"]
                else False
            )
            nemit_y = (
                self.file_block["input"]["twiss"]["nemit_y"]
                if "nemit_y" in self.file_block["input"]["twiss"]
                else False
            )
            return {
                "horizontal": {
                    "alpha": alpha_x,
                    "beta": beta_x,
                    "nEmit": nemit_x,
                },
                "vertical": {
                    "alpha": alpha_y,
                    "beta": beta_y,
                    "nEmit": nemit_y,
                },
            }
        else:
            return {
                "horizontal": {
                    "alpha": False,
                    "beta": False,
                    "nEmit": False,
                },
                "vertical": {
                    "alpha": False,
                    "beta": False,
                    "nEmit": False,
                },
            }

    def preProcess(self) -> None:
        """
        Pre-process the lattice before running the simulation.
        This method initializes the initial Twiss parameters by calling the `getInitialTwiss` method.

        Returns
        -------
        None
        """
        self.initial_twiss = self.getInitialTwiss()

    def postProcess(self):
        pass

    def __repr__(self):
        return self.elements

    def __str__(self):
        str = self.name + " = ("
        for e in self.elements:
            if len((str + e).splitlines()[-1]) > 60:
                str += "&\n"
            str += e + ", "
        return str + ")"

    def createDrifts(
        self, drift_elements: tuple = ("screen", "beam_position_monitor")
    ) -> dict:
        """
        Insert drifts into a sequence of 'elements'.
        This method creates drifts for elements that are not subelements and have a length greater than zero.
        It calculates the start and end positions of each element and creates drift elements accordingly.

        Parameters
        ----------
        drift_elements: tuple, optional
            A tuple of element types for which drifts should be created.
            Default is ("screen", "beam_position_monitor").

        Returns
        -------
        dict
            A dictionary containing the new drift elements created for the lattice.
            The keys are the names of the new drift elements, and the values are the corresponding drift objects.
        """
        positions = []
        originalelements = dict()
        elementno = 0
        newelements = dict()
        for name in list(self.elements.keys()):
            if not self.elements[name].subelement:
                originalelements[name] = self.elements[name]
                pos = np.array(self.elementObjects[name].position_start)
                # If element is a cavity, we need to offset the cavity by the coupling cell length
                # to make it consistent with ASTRA
                if originalelements[name].objecttype == "cavity" and hasattr(
                    originalelements[name], "coupling_cell_length"
                ):
                    pos += originalelements[name].coupling_cell_length
                    # print('Adding coupling_cell_length of ', originalelements[name].coupling_cell_length,'to the start position')
                positions.append(pos)
                positions.append(self.elementObjects[name].position_end)
        positions = positions[1:]
        positions.append(positions[-1])
        driftdata = list(
            zip(iter(list(originalelements.items())), list(chunks(positions, 2)))
        )

        lscbins = self.lsc_bins if self.lscDrifts is True else 0
        csr = 1 if self.csrDrifts is True else 0
        lsc = 1 if self.lscDrifts is True else 0
        drifttype = lscdrift if self.csrDrifts or self.lscDrifts else edrift
        if drifttype == lscdrift:
            objtype = "lscdrift"
        else:
            objtype = "edrift"

        for e, d in driftdata:
            if (e[1].objecttype in drift_elements) and round(e[1].length / 2, 6) > 0:
                name = e[0] + "-drift-01"
                driftparams = {
                    "length": round(e[1].length / 2, 6),
                    "csr_enable": csr,
                    "lsc_enable": lsc,
                    "use_stupakov": 1,
                    "csrdz": 0.01,
                    "lsc_bins": lscbins,
                    "lsc_high_frequency_cutoff_start": self.lsc_high_frequency_cutoff_start,
                    "lsc_high_frequency_cutoff_end": self.lsc_high_frequency_cutoff_end,
                    "lsc_low_frequency_cutoff_start": self.lsc_low_frequency_cutoff_start,
                    "lsc_low_frequency_cutoff_end": self.lsc_low_frequency_cutoff_end,
                }
                newdrift = drifttype(
                    objectname=name,
                    objecttype=objtype,
                    global_parameters=self.global_parameters,
                    **driftparams,
                )
                newelements[name] = newdrift
                new_bpm_screen = deepcopy(e[1])
                new_bpm_screen.length = 0
                newelements[e[0]] = new_bpm_screen
                name = e[0] + "-drift-02"
                newdrift = drifttype(
                    objectname=name,
                    objecttype=objtype,
                    global_parameters=self.global_parameters,
                    **driftparams,
                )
                newelements[name] = newdrift
            else:
                # print('NOT Drift Element', e[1]["objecttype"], round(e[1]["length"] / 2, 6))
                newelements[e[0]] = e[1]
            if e[1].objecttype == "dipole":
                drifttype = (
                    csrdrift
                    if self.csrDrifts
                    else lscdrift if self.lscDrifts else edrift
                )
            if len(d) > 1:
                x1, y1, z1 = d[0]
                x2, y2, z2 = d[1]
                try:
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
                    vector = dot((d[1] - d[0]), [0, 0, -1])
                except Exception as exc:
                    warn(f"Element with error = {e[0]}")
                    warn(d)
                    raise exc
                if self.allow_negative_drifts or (
                    round(length, 6) > 0 and vector < 1e-6
                ):
                    elementno += 1
                    name = self.objectname + "_DRIFT_" + str(elementno).zfill(2)
                    middle = [(a + b) / 2.0 for a, b in zip(d[0], d[1])]
                    newdrift = drifttype(
                        objectname=name,
                        objecttype=objtype,
                        global_parameters=self.global_parameters,
                        **{
                            "length": round(length, 6),
                            "position_start": list(d[0]),
                            "position_end": list(d[1]),
                            "centre": middle,
                            "csr_enable": csr,
                            "lsc_enable": lsc,
                            "use_stupakov": 1,
                            "csrdz": 0.01,
                            "lsc_bins": lscbins,
                            "lsc_high_frequency_cutoff_start": self.lsc_high_frequency_cutoff_start,
                            "lsc_high_frequency_cutoff_end": self.lsc_high_frequency_cutoff_end,
                            "lsc_low_frequency_cutoff_start": self.lsc_low_frequency_cutoff_start,
                            "lsc_low_frequency_cutoff_end": self.lsc_low_frequency_cutoff_end,
                        },
                    )
                    newelements[name] = newdrift
                elif length < 0 or vector > 1e-6:
                    raise Exception(
                        "Lattice has negative drifts!",
                        self.allow_negative_drifts,
                        e[0],
                        e[1],
                        length,
                    )
        return newelements

    def getSValues(
        self, as_dict: bool = False, at_entrance: bool = False
    ) -> list | dict:
        """
        Get the S values for the elements in the lattice.
        This method calculates the cumulative length of the elements in the lattice,
        starting from the entrance or the first element, depending on the `at_entrance` parameter.
        It returns a list or dict of S values, which represent the positions of the elements along the lattice.

        Parameters
        ----------
        as_dict: bool, optional
            If True, returns a dictionary with element names as keys and their S values as values.
        at_entrance: bool, optional
            If True, calculates S values starting from the entrance of the lattice.
            If False, calculates S values starting from the first element.

        Returns
        -------
        list | dict
            A list or dictionary of S values for the elements in the lattice.
            If `as_dict` is True, returns a dictionary with element names as keys and their S values as values.
            If `as_dict` is False, returns a list of S values.
        """
        elems = self.createDrifts()
        s = [0]
        for e in list(elems.values()):
            s.append(s[-1] + e.length)
        s = s[:-1] if at_entrance else s[1:]
        if as_dict:
            return dict(zip([e.objectname for e in elems.values()], s))
        return list(s)

    def getZValues(self, drifts: bool = True, as_dict: bool = False) -> list | dict:
        """
        Get the Z values for the elements in the lattice.
        This method calculates the cumulative length of the elements in the lattice,
        starting from the entrance or the first element, depending on the `at_entrance` parameter.
        It returns a list or dict of S values, which represent the positions of the elements along the lattice.

        Parameters
        ----------
        drifts: bool, optional
            If True, includes drift elements in the calculation.
            If False, only considers the main elements in the lattice.
        as_dict: bool, optional
            If True, returns a dictionary with element names as keys and their Z values as values.

        Returns
        -------
        list | dict
            A list or dictionary of Z values for the elements in the lattice.
            If `as_dict` is True, returns a dictionary with element names as keys and their Z values as values.
            If `as_dict` is False, returns a list of Z values.
        """
        if drifts:
            elems = self.createDrifts()
        else:
            elems = self.elements
        if as_dict:
            return {e.objectname: [e.start[2], e.end[2]] for e in elems.values()}
        return [[e.start[2], e.end[2]] for e in elems.values()]

    def getNames(self, drifts: bool = True) -> list:
        """
        Get the names of the elements in the lattice.

        Parameters
        ----------
        drifts: bool, optional
            If True, includes drift elements in the list of names.

        Returns
        -------
        list
            A list of names of the elements in the lattice.
            If `drifts` is True, includes drift elements; otherwise, only includes main elements.
        """
        if drifts:
            elems = self.createDrifts()
        else:
            elems = self.elements
        return [e.objectname for e in list(elems.values())]

    def getElems(self, drifts: bool = True, as_dict: bool = False) -> list | dict:
        """
        Get the elements in the lattice.

        Parameters
        ----------
        drifts: bool, optional
            If True, includes drift elements in the list of elements.
        as_dict: bool, optional
            If True, returns a dictionary with element names as keys and their corresponding element objects as values.

        Returns
        -------
        list | dict
            A list or dictionary of elements in the lattice.
        """
        if drifts:
            elems = self.createDrifts()
        else:
            elems = self.elements
        if as_dict:
            return {e.objectname: e for e in list(elems.values())}
        return [e for e in list(elems.values())]

    def getSNames(self) -> list:
        """
        Get the names and S values of the elements in the lattice.

        Returns
        -------
        list
            A list of tuples, where each tuple contains the name of an element and its corresponding S value.
        """
        s = self.getSValues()
        names = self.getNames()
        return list(zip(names, s))

    def getSNamesElems(self) -> tuple:
        """
        Get the names, elements, and S values of the elements in the lattice.

        Returns
        -------
        tuple
            A tuple containing three elements:
            - A list of names of the elements.
            - A list of element objects.
            - A list of S values corresponding to the elements.
        """
        s = self.getSValues()
        names = self.getNames()
        elems = self.getElems()
        return names, elems, s

    def getZNamesElems(self) -> tuple:
        """
        Get the names, elements, and Z values of the elements in the lattice.

        Returns
        -------
        tuple
            A tuple containing three elements:
            - A list of names of the elements.
            - A list of element objects.
            - A list of Z values corresponding to the elements.
        """
        z = self.getZValues()
        names = self.getNames()
        elems = self.getElems()
        return names, elems, z

    def findS(self, elem) -> list:
        """
        Find the S values for a specific element in the lattice.

        Parameters
        ----------
        elem: str
            The name of the element to find in the lattice.


        Returns
        -------
        list
            A list of tuples, where each tuple contains the name of the element and its corresponding S value.
            If the element does not exist in the lattice, returns an empty list.
        """
        if elem in self.allElements:
            sNames = self.getSNames()
            return [a for a in sNames if a[0] == elem]
        return []

    def updateRunSettings(self, runSettings: runSetup) -> None:
        """
        Update the run settings for the lattice.

        Parameters
        ----------
        runSettings: runSetup
            An instance of runSetup containing the new run settings.

        Raises
        ------
        TypeError
            If the `runSettings` argument is not an instance of `runSetup`.

        """
        if isinstance(runSettings, runSetup):
            self.runSettings = runSettings
        else:
            raise TypeError(
                "runSettings argument passed to frameworkLattice.updateRunSettings is not a runSetup instance"
            )


class frameworkCommand(frameworkObject):
    """
    Class defining a framework command, which is used to generate commands used in setup files
    for various simulation codes.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(frameworkCommand, self).__init__(
            *args,
            **kwargs,
        )
        if self.objecttype not in commandkeywords:
            raise NameError("Command '%s' does not exist" % self.objecttype)

    def write_Elegant(self) -> str:
        """
        Writes the command string for ELEGANT.

        Returns
        -------
        str
            String representation of the command for ELEGANT
        """
        string = "&" + self.objecttype + "\n"
        for key in commandkeywords[self.objecttype]:
            if (
                key.lower() in self.objectproperties.allowedkeywords
                and not key == "objectname"
                and not key == "objecttype"
                and hasattr(self, key)
            ):
                string += "\t" + key + " = " + str(getattr(self, key.lower())) + "\n"
        string += "&end\n"
        return string

    def write_MAD8(self) -> str:
        """
        Writes the command string for MAD8.
        # TODO deprecated?

        Returns
        -------
        str
            String representation of the command for MAD8
        """
        string = self.objecttype
        # print(self.objecttype, self.objectproperties)
        for key in commandkeywords[self.objecttype]:
            if (
                key.lower() in self.objectproperties
                and not key == "name"
                and not key == "type"
                and not self.objectproperties[key.lower()] is None
            ):
                e = "," + key + "=" + str(self.objectproperties[key.lower()])
                if len((string + e).splitlines()[-1]) > 79:
                    string += ",&\n"
                string += e
        string += ";\n"
        return string


class frameworkGroup(object):
    """
    Class defining a framework group, which is used to group together elements to perform coordinated
    actions on them.
    """

    def __init__(self, name, elementObjects, type, elements, **kwargs):
        super(frameworkGroup, self).__init__()
        self.objectname = name
        self.type = type
        self.elements = elements
        self.allElementObjects = elementObjects.elementObjects
        self.allGroupObjects = elementObjects.groupObjects

    def update(self, **kwargs):
        pass

    def get_Parameter(self, p: str) -> Any:
        """
        Get a specific parameter associated with the group, i.e. bunch compressor angle

        Parameters
        ----------
        p: str
            A parameter associated with the group

        Returns
        -------
        Any
            The parameter, if defined.
        """
        try:
            isinstance(type(getattr(self, p)), p)
            return getattr(self, p)
        except Exception:
            if self.elements[0] in self.allGroupObjects:
                return getattr(self.allGroupObjects[self.elements[0]], p)
            return getattr(self.allElementObjects[self.elements[0]], p)

    def change_Parameter(self, p: Any, v: Any) -> None:
        """
        Set a parameter on all elements in the group.

        Parameters
        ----------
        p: str
            The parameter to be set
        v: Any
            The value to be set.
        """
        try:
            getattr(self, p)
            setattr(self, p, v)
            if p == "angle":
                self.set_angle(v)
            # print ('Changing group ', self.objectname, ' ', p, ' = ', v, '  result = ', self.get_Parameter(p))
        except Exception:
            for e in self.elements:
                setattr(self.allElementObjects[e], p, v)
                # print ('Changing group elements ', self.objectname, ' ', p, ' = ', v, '  result = ', self.allElementObjects[self.elements[0]].objectname, self.get_Parameter(p))

    # def __getattr__(self, p):
    #     return self.get_Parameter(p)

    def __repr__(self):
        return str([self.allElementObjects[e].objectname for e in self.elements])

    def __str__(self):
        return str([self.allElementObjects[e].objectname for e in self.elements])

    def __getitem__(self, key):
        return self.get_Parameter(key)

    def __setitem__(self, key, value):
        return self.change_Parameter(key, value)


class element_group(frameworkGroup):
    """
    Class defining a group of elements, which is used to group together elements to perform coordinated
    actions on them.
    """

    def __init__(self, name, elementObjects, type, elements, **kwargs):
        super().__init__(name, elementObjects, type, elements, **kwargs)

    def __str__(self):
        return str([self.allElementObjects[e] for e in self.elements])


class r56_group(frameworkGroup):
    """
    Class defining a group of elements with a total R56.
    """

    def __init__(self, name, elementObjects, type, elements, ratios, keys, **kwargs):
        super().__init__(name, elementObjects, type, elements, **kwargs)
        self.ratios = ratios
        self.keys = keys
        self._r56 = None

    def __str__(self):
        return str({e: k for e, k in zip(self.elements, self.keys)})

    def get_Parameter(self, p: str) -> Any:
        """
        Get a parameter associated with the group.

        Parameters
        ----------
        p: str
            The parameter to be retrieved.

        Returns
        -------
        Any
            The parameter.
        """
        if str(p) == "r56":
            return self.r56
        else:
            return super().get_Parameter(p)

    @property
    def r56(self) -> float:
        """
        Get the R56 of the group of elements

        Returns
        -------
        float
            The R56 pararmeter
        """
        return self._r56

    @r56.setter
    def r56(self, r56: float) -> None:
        """
        Set the R56 of the group of elements

        Parameters
        ----------
        r56: float
            The R56 to be set
        """
        # print('Changing r56!', self._r56)
        self._r56 = r56
        data = {"r56": self._r56}
        parser = MathParser(data)
        values = [parser.parse(e) for e in self.ratios]
        # print('\t', list(zip(self.elements, self.keys, values)))
        for e, k, v in zip(self.elements, self.keys, values):
            self.updateElements(e, k, v)

    def updateElements(self, element: str | list | tuple, key: str, value: Any) -> None:
        """
        Update one or more elements in the group.

        Parameters
        ----------
        element: str, list or tuple
            The element(s) to be updated
        key: str
            The parameter in the element or group of elements to be changed
        value: Any
            The value to which the parameter should be set
        """
        # print('R56 : updateElements', element, key, value)
        if isinstance(element, (list, tuple)):
            [self.updateElements(e, key, value) for e in self.elements]
        else:
            if element in self.allElementObjects:
                # print('R56 : updateElements : element', element, key, value)
                self.allElementObjects[element].change_Parameter(key, value)
            if element in self.allGroupObjects:
                # print('R56 : updateElements : group', element, key, value)
                self.allGroupObjects[element].change_Parameter(key, value)


class chicane(frameworkGroup):
    """
    Class defining a 4-dipole chicane.
    """

    def __init__(self, name, elementObjects, type, elements, **kwargs):
        super(chicane, self).__init__(name, elementObjects, type, elements, **kwargs)
        self.ratios = (1, -1, -1, 1)

    def update(self, **kwargs) -> None:
        """
        Update the bending angle and/or dipole width and/or dipole gap of all magnets in the chicane.

        Parameters
        ----------
        **kwargs: Dict
            Dictionary containing parameters to be updated -- must be in ["dipoleangle", "width", "gap"]
        """
        if "dipoleangle" in kwargs:
            self.set_angle(kwargs["dipoleangle"])
        if "width" in kwargs:
            self.change_Parameter("width", kwargs["width"])
        if "gap" in kwargs:
            self.change_Parameter("gap", kwargs["gap"])
        return None

    @property
    def angle(self) -> float:
        """
        Bending angle of the chicane

        Returns
        -------
        float
            The bending angle
        """
        obj = [self.allElementObjects[e] for e in self.elements]
        return float(obj[0].angle)

    @angle.setter
    def angle(self, theta: float) -> None:
        """
        Set the bending angle of the chicane; see :func:`~SimulationFramework.Framework_objects.chicane.set_angle`.

        Parameters
        -----------
        theta: float
            Chicane bending angle
        """
        self.set_angle(theta)

    # def set_angle2(self, a):
    #     indices = list(sorted([list(self.allElementObjects).index(e) for e in self.elements]))
    #     dipole_objs = [self.allElementObjects[e] for e in self.elements]
    #     obj = [self.allElementObjects[list(self.allElementObjects)[e]] for e in range(indices[0],indices[-1]+1)]
    #     starting_angle = obj[0].theta
    #     dipole_number = 0
    #     for i in range(len(obj)):
    #         start = obj[i].position_start
    #         x1 = np.transpose([start])
    #         obj[i].global_rotation[2] = starting_angle
    #         if obj[i] in dipole_objs:
    #             start_angle = obj[i].angle
    #             obj[i].angle = a*self.ratios[dipole_number]
    #             if abs(obj[i].angle) > 0:
    #                 scale = (np.tan(obj[i].angle/2.0) / obj[i].angle) / (np.tan(start_angle/2.0) / start_angle)
    #             else:
    #                 scale = 1
    #             obj[i].length = obj[i].length / scale
    #             dipole_number += 1
    #             elem_angle = obj[i].angle
    #         else:
    #             elem_angle = obj[i].angle if obj[i].angle is not None else 0
    #         if not obj[i] in dipole_objs:
    #             obj[i].centre = list(obj[i].middle)
    #         xstart, ystart, zstart = obj[i].position_end
    #         if i < len(obj)-1:
    #             xend, yend, zend = obj[i+1].position_start
    #             angle = starting_angle + elem_angle
    #             # print('angle = ', angle, starting_angle, obj[i+1].objectname)
    #             length = float((zend - zstart))
    #             endx = chop(float(xstart - np.tan(angle)*(length/2.0)))
    #             obj[i+1].centre[0] =  endx
    #             obj[i+1].global_rotation[2] =  angle
    #             starting_angle += elem_angle

    def set_angle(self, a: float) -> None:
        """
        Set the chicane bending angle, including updating the inter-dipole drift lengths.

        Parameters
        ----------
        a: float
            The angle to be set
        """
        indices = list(
            sorted([list(self.allElementObjects).index(e) for e in self.elements])
        )
        dipole_objs = [self.allElementObjects[e] for e in self.elements]
        obj = [
            self.allElementObjects[list(self.allElementObjects)[e]]
            for e in range(indices[0], indices[-1] + 1)
        ]
        dipole_number = 0
        ref_pos = None
        ref_angle = None
        for i in range(len(obj)):
            if dipole_number > 0:
                # print('before',obj[i])
                adj = obj[i].centre[2] - ref_pos[2]
                # print('  adj', adj)
                # print('  ref_angle', ref_angle)
                obj[i].centre = [
                    ref_pos[0] + np.tan(-1.0 * ref_angle) * adj,
                    0,
                    obj[i].centre[2],
                ]
                obj[i].global_rotation[2] = ref_angle
                # print('after',obj[i])
            if obj[i] in dipole_objs:
                # print('DIPOLE before',obj[i])
                ref_pos = obj[i].middle
                obj[i].angle = a * self.ratios[dipole_number]
                ref_angle = obj[i].global_rotation[2] + obj[i].angle
                dipole_number += 1
                # print('DIPOLE after',obj[i])
        # print('\n\n\n')

    def __str__(self):
        return str(
            [
                [
                    self.allElementObjects[e].objectname,
                    self.allElementObjects[e].angle,
                    self.allElementObjects[e].global_rotation[2],
                    self.allElementObjects[e].position_start,
                    self.allElementObjects[e].position_end,
                ]
                for e in self.elements
            ]
        )


class s_chicane(chicane):
    """
    Class defining an s-type chicane; in this case the bending ratios for
    :func:`~SimulationFramework.Framework_objects.chicane.set_angle` are different.
    """

    def __init__(self, name, elementObjects, type, elements, **kwargs):
        super(s_chicane, self).__init__(name, elementObjects, type, elements, **kwargs)
        self.ratios = (-1, 2, -2, 1)


class frameworkCounter(dict):
    """
    Class defining a counter object, used for numbering elements of the same type in ASTRA and CSRTrack
    """

    def __init__(self, sub={}):
        super(frameworkCounter, self).__init__()
        self.sub = sub

    def counter(self, typ: str) -> int:
        """
        Increment count of elements of a given type in the lattice.

        Parameters
        ----------
        typ: str
            Element type

        Returns
        -------
        int
            The updated number of elements of a given type defined so far
        """
        typ = self.sub[typ] if typ in self.sub else typ
        if typ not in self:
            return 1
        return self[typ] + 1

    def value(self, typ: str) -> int:
        """
        Number of elements of a given type in the lattice.

        Parameters
        ----------
        typ: str
            Element type

        Returns
        -------
        int
            The number of elements of a given type defined so far
        """
        typ = self.sub[typ] if typ in self.sub else typ
        if typ not in self:
            return 1
        return self[typ]

    def add(self, typ: str, n: PositiveInt = 1) -> int:
        """
        Add to count of elements of a given type in the lattice.

        Parameters
        ----------
        typ: str
            Element type
        n: PositiveInt, optional
            Add more than one element at a time

        Returns
        -------
        int
            The number of elements of a given type defined so far
        """
        typ = self.sub[typ] if typ in self.sub else typ
        if typ not in self:
            self[typ] = n
        else:
            self[typ] += n
        return self[typ]

    def subtract(self, typ: str) -> int:
        """
        Reduce count of elements of a given type in the lattice.

        Parameters
        ----------
        typ: str
            Element type

        Returns
        -------
        int
            The updated number of elements of a given type defined so far
        """
        typ = self.sub[typ] if typ in self.sub else typ
        if typ not in self:
            self[typ] = 0
        else:
            self[typ] = self[typ] - 1 if self[typ] > 0 else 0
        return self[typ]


class getGrids(object):
    """
    Class defining the appropriate number of space charge bins given the number of particles,
    defined as the closest power of 8 to the cube root of the number of particles.
    """

    def __init__(self):
        self.powersof8 = np.asarray([2**j for j in range(1, 20)])

    def getGridSizes(self, x: PositiveInt) -> int:
        """
        Calculate the 3D space charge grid size given the number of particles, minimum of 4

        Parameters
        ----------
        x: PositiveInt
            Number of particles

        Returns
        -------
        int
            The number of space charge grids
        """
        self.x = abs(x)
        self.cuberoot = int(round(self.x ** (1.0 / 3)))
        return max([4, self.find_nearest(self.powersof8, self.cuberoot)])

    def find_nearest(self, array: np.ndarray | list, value: int) -> int:
        """
        Get the nearest value in an array to the value provided; in this case the array should be a list of
        powers of 8.

        Parameters
        ----------
        array: np.ndarray or list
            Array of values to be checked
        value: Value to be found in the array

        Returns
        -------
        int
            The closest value in `array` to `value`
        """
        self.array = array
        self.value = value
        self.idx = (np.abs(self.array - self.value)).argmin()
        return self.array[self.idx]
