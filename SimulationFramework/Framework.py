"""
Simframe Framework Module

The main class for handling the tracking of a particle distribution through a lattice.

Settings files can be loaded in, consisting of one or more :ref:`MasterLattice` YAML files. This creates
:class:`~SimulationFramework.Framework_objects.frameworkLattice` objects, each of which contains
:class:`~SimulationFramework.Framework_objects.frameworkElement` objects.

These objects can be modified directly through the :class:`~SimulationFramework.Framework.Framework` class.

Based on the tracking code(s) provided to the framework, the particle distribution is tracked through the lattice
sequentially, and output beam distributions are generated and converted to the standard SimFrame HDF5 format.

Summary files containing Twiss parameters, and a summary of the beam files, are generated after tracking.

Classes:
    - :class:`~SimulationFramework.Framework.Framework`: Top-level class for loading and modifying lattice\
    settings and tracking through them

    - :class:`~SimulationFramework.Framework.frameworkDirectory`: Class to load a tracking run from a directory \
    and reading the Beam and Twiss files and making them available.
"""

import os
import sys
import yaml
import inspect
from typing import Any, Dict
from pprint import pprint
import numpy as np
from copy import deepcopy
from .Modules.merge_two_dicts import merge_two_dicts
from .Modules import Beams as rbf
from .Modules import Twiss as rtf
from .Modules import constants
from .Codes import Executables as exes
from .Codes.Generators.Generators import (
    ASTRAGenerator,
    GPTGenerator,
    generator_keywords,
    frameworkGenerator,
)
from .Framework_objects import runSetup, frameworkElement
from . import Framework_lattices as frameworkLattices
from . import Framework_elements as frameworkElements
from .Framework_Settings import FrameworkSettings
from .FrameworkHelperFunctions import (
    _rotation_matrix,
    clean_directory,
    convert_numpy_types,
)
from pydantic import (
    BaseModel,
    field_validator,
)
from warnings import warn

try:
    import MasterLattice  # type: ignore

    if MasterLattice.__file__ is not None:
        MasterLatticeLocation = os.path.dirname(MasterLattice.__file__) + "/"
    else:
        MasterLatticeLocation = None
except ImportError:
    MasterLatticeLocation = None
try:
    import SimCodes  # type: ignore

    SimCodesLocation = os.path.dirname(SimCodes.__file__) + "/"
except ImportError:
    SimCodesLocation = None
try:
    import SimulationFramework.Modules.plotting.plotting as groupplot

    use_matplotlib = True
except ImportError as e:
    print("Import error - plotting disabled. Missing package:", e)
    use_matplotlib = False
from tqdm import tqdm
from munch import Munch, unmunchify

_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG


def dict_representer(dumper, data):
    return dumper.represent_dict(iter(list(data.items())))


def dict_constructor(loader, node):
    return dict(loader.construct_pairs(node))


yaml.add_representer(dict, dict_representer)
yaml.add_constructor(_mapping_tag, dict_constructor)

latticeClasses = [
    obj
    for name, obj in inspect.getmembers(
        sys.modules["SimulationFramework.Framework_lattices"]
    )
    if inspect.isclass(obj)
]

disallowed = [
    "allowedkeywords",
    "conversion_rules_elegant",
    "conversion_rules_ocelot",
    "objectdefaults",
    "global_parameters",
    "objectname",
    "subelement",
    "beam",
]

disallowed_changes = [
    "allowedkeywords",
    "conversion_rules_elegant",
    "conversion_rules_ocelot",
    "objectdefaults",
    "global_parameters",
    "beam",
    "field_definition",
    "wakefield_definition",
]


class Framework(BaseModel):
    """
    The main class for handling the tracking of a particle distribution through a lattice.

    Settings files can be loaded in, consisting of one or more :ref:`MasterLattice` YAML files. This creates
    :class:`~SimulationFramework.Framework_objects.frameworkLattice` objects, each of which contains
    :class:`~SimulationFramework.Framework_objects.frameworkElement` objects.

    These objects can be modified directly through the :class:`~SimulationFramework.Framework.Framework` class.

    Based on the tracking code(s) provided to the framework, the particle distribution is tracked through the lattice
    sequentially, and output beam distributions are generated and converted to the standard SimFrame HDF5 format.

    Summary files containing Twiss parameters, and a summary of the beam files, are generated after tracking.
    """

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True
        validate_assignment = True

    directory: str = "test"
    """The directory into which simulation files will be placed"""

    master_lattice: str | None = None
    """Location of the master lattice files. If the package is installed, 
    this will be configured automatically"""

    simcodes: str | None = None
    """Location of the simulation codes directory. If the package is installed, 
    this will be configured autonatically"""

    overwrite: bool | None = None
    """Flag to indicate whether existing files are to be overwritten
    #TODO deprecated?"""

    runname: str = "CLARA_240"
    """Name of the run for this setup
    #TODO deprecated?"""

    clean: bool = False
    """Flag to indicate whether all files in the existing directory are to be removed"""

    verbose: bool = True
    """Flag to indicate whether status updates should be printed during tracking"""

    sddsindex: int = 0
    """Index for SDDS files"""

    delete_output_files: bool = False
    """Flag to indicate whether output files are to be deleted after tracking"""

    global_parameters: Dict = {}
    """Dictionary containing global parameters accessible to all classes"""

    elementObjects: Dict = {}
    """Dictionary containing all :class:`~SimulationFramework.Framework_objects.frameworkElement` objects"""

    latticeObjects: Dict = {}
    """Dictionary containing all :class:`~SimulationFramework.Framework_objects.frameworkLattice` objects"""

    commandObjects: Dict = {}
    """Dictionary containing all :class:`~SimulationFramework.Framework_objects.frameworkCommand` objects"""

    groupObjects: Dict = {}
    """Dictionary containing all :class:`~SimulationFramework.Framework_objects.frameworkGroup` objects"""

    fileSettings: Dict = {}
    """Dictionary containing all file settings"""

    globalSettings: Dict = {}
    """Dictionary containing all global settings"""

    generatorSettings: Dict = {}
    """Dictionary containing all generator settings"""

    original_elementObjects: Dict = {}
    """Dictionary containing all :class:`~SimulationFramework.Framework_objects.frameworkElement` objects
    before changes are made"""

    progress: int | float = 0
    """Current progress of tracking"""

    tracking: bool = False
    """Flag to indicate whether the Framework is tracking"""

    basedirectory: str = ""
    """Current working directory"""

    filedirectory: str = ""
    """Directory for files"""

    subdirectory: str | None = None
    """Directory into which files are to be placed"""

    generator: frameworkGenerator | None = None
    """The :class:`~SimulationFramework.Codes.Generators.Generators.frameworkGenerator` object"""

    settings: FrameworkSettings | None = None
    """Settings for the lattice"""

    settingsFilename: str | None = None
    """Filename containing lattice settings"""


    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(Framework, self).__init__(
            *args,
            **kwargs,
        )
        gptlicense = os.environ["GPTLICENSE"] if "GPTLICENSE" in os.environ else ""
        astra_use_wsl = os.environ["WSL_ASTRA"] if "WSL_ASTRA" in os.environ else 1
        self.global_parameters = {
            "beam": rbf.beam(sddsindex=self.sddsindex),
            "GPTLICENSE": gptlicense,
            "delete_tracking_files": self.delete_output_files,
            "astra_use_wsl": astra_use_wsl,
        }
        self.setSubDirectory(self.directory)
        self.setMasterLatticeLocation(self.master_lattice)
        self.setSimCodesLocation(self.simcodes)

        self.executables = exes.Executables(self.global_parameters)
        self.defineASTRACommand = self.executables.define_astra_command
        self.defineElegantCommand = self.executables.define_elegant_command
        self.defineASTRAGeneratorCommand = (
            self.executables.define_ASTRAgenerator_command
        )
        self.defineCSRTrackCommand = self.executables.define_csrtrack_command
        self.define_gpt_command = self.executables.define_gpt_command

        # object encoding settings for simulations with multiple runs
        self.runSetup = runSetup()

    @field_validator("basedirectory", mode="before")
    @classmethod
    def validate_base_directory(cls, value: str) -> str:
        if len(value) > 0 and os.path.isdir(value):
            return value
        return os.getcwd()

    @field_validator("filedirectory", mode="before")
    @classmethod
    def validate_file_directory(cls, value: str) -> str:
        if len(value) > 0 and os.path.isdir(value):
            return value
        return os.path.dirname(os.path.abspath(__file__))

    def __repr__(self) -> repr:
        return repr(
            {
                "master_lattice_location": self.global_parameters[
                    "master_lattice_location"
                ],
                "subdirectory": self.subdirectory,
                "settingsFilename": self.settingsFilename,
            }
        )

    def clear(self) -> None:
        """
        Clear out :attr:`~elementObjects`, :attr:`~latticeObjects`, :attr:`~commandObjects`, :attr:`~groupObjects`
        """
        self.elementObjects = dict()
        self.latticeObjects = dict()
        self.commandObjects = dict()
        self.groupObjects = dict()

    def change_subdirectory(self, *args, **kwargs) -> None:
        """
        Change the subdirectory and `master_subdir` in :attr:`~global_parameters` to which lattice and
        beam files will be written.
        """
        self.setSubDirectory(*args, **kwargs)

    def setSubDirectory(self, direc: str) -> None:
        """
        Change the subdirectory and `master_subdir` in :attr:`~global_parameters` to which lattice and
        beam files will be written.

        If :attr:`~clean`, then all existing files in the directory are removed.

        Parameters
        ----------
        direc: str
            Directory to which files will be written. If directory does not exist, create it.
        """
        self.subdirectory = os.path.abspath(direc)
        self.global_parameters["master_subdir"] = self.subdirectory
        if not os.path.exists(self.subdirectory):
            os.makedirs(self.subdirectory, exist_ok=True)
        else:
            if self.clean is True:
                clean_directory(self.subdirectory)
        if self.overwrite is None:
            self.overwrite = True

    def setMasterLatticeLocation(self, master_lattice: str | None = None) -> None:
        """
        Set the location of the :ref:`MasterLattice` package.

        This then also sets the `master_lattice_location` in :attr:`~global_parameters`.

        Parameters
        ----------
        master_lattice: str
            The full path to the MasterLattice folder
        """
        global MasterLatticeLocation
        if master_lattice is None:
            if MasterLatticeLocation is not None:
                self.global_parameters["master_lattice_location"] = (
                    MasterLatticeLocation.replace("\\", "/")
                )
                if self.verbose:
                    print(
                        "Found MasterLattice Package =",
                        self.global_parameters["master_lattice_location"],
                    )
            elif os.path.isdir(
                os.path.abspath(
                    os.path.dirname(os.path.abspath(__file__))
                    + "/../../MasterLattice/MasterLattice"
                )
                + "/"
            ):
                self.global_parameters["master_lattice_location"] = (
                    os.path.abspath(
                        os.path.dirname(os.path.abspath(__file__))
                        + "/../../MasterLattice/MasterLattice"
                    )
                    + "/"
                ).replace("\\", "/")
                if self.verbose:
                    print(
                        "Found MasterLattice Directory 2-up =",
                        self.global_parameters["master_lattice_location"],
                    )
            elif os.path.isdir(
                os.path.abspath(
                    os.path.dirname(os.path.abspath(__file__))
                    + "/../MasterLattice/MasterLattice"
                )
                + "/"
            ):
                self.global_parameters["master_lattice_location"] = (
                    os.path.abspath(
                        os.path.dirname(os.path.abspath(__file__))
                        + "/../MasterLattice/MasterLattice"
                    )
                    + "/"
                ).replace("\\", "/")
                if self.verbose:
                    print(
                        "Found MasterLattice Directory 1-up =",
                        self.global_parameters["master_lattice_location"],
                    )
            elif os.path.isdir(
                os.path.abspath(
                    os.path.dirname(os.path.abspath(__file__)) + "/../MasterLattice"
                )
                + "/"
            ):
                self.global_parameters["master_lattice_location"] = (
                    os.path.abspath(
                        os.path.dirname(os.path.abspath(__file__)) + "/../MasterLattice"
                    )
                    + "/"
                ).replace("\\", "/")
                if self.verbose:
                    print(
                        "Found MasterLattice Directory 1-up =",
                        self.global_parameters["master_lattice_location"],
                    )
            else:
                if self.verbose:
                    print(
                        "Master Lattice not available - specify using master_lattice=<location>"
                    )
                    self.global_parameters["master_lattice_location"] = "."
        else:
            self.global_parameters["master_lattice_location"] = os.path.join(
                os.path.abspath(master_lattice), "./"
            )
        MasterLatticeLocation = self.global_parameters["master_lattice_location"]

    def setSimCodesLocation(self, simcodes: str | None = None) -> None:
        """
        Set the location of the :ref:`SimCodes` package.

        This then also sets the `simcodes_location` in :attr:`~global_parameters`.

        Parameters
        ----------
        simcodes: str
            The full path to the SimCodes folder
        """
        global SimCodesLocation
        if simcodes is None:
            if SimCodesLocation is not None:
                self.global_parameters["simcodes_location"] = SimCodesLocation.replace(
                    "\\", "/"
                )
                if self.verbose:
                    print(
                        "Found SimCodes Package =",
                        self.global_parameters["simcodes_location"],
                    )
            elif os.path.isdir(
                os.path.abspath(
                    os.path.dirname(os.path.abspath(__file__))
                    + "/../../SimCodes/SimCodes"
                )
                + "/"
            ):
                self.global_parameters["simcodes_location"] = (
                    os.path.abspath(
                        os.path.dirname(os.path.abspath(__file__))
                        + "/../../SimCodes/SimCodes"
                    )
                    + "/"
                ).replace("\\", "/")
                if self.verbose:
                    print(
                        "Found SimCodes Directory 2-up =",
                        self.global_parameters["simcodes_location"],
                    )
            elif os.path.isdir(
                os.path.abspath(
                    os.path.dirname(os.path.abspath(__file__)) + "/../SimCodes/SimCodes"
                )
                + "/"
            ):
                self.global_parameters["simcodes_location"] = (
                    os.path.abspath(
                        os.path.dirname(os.path.abspath(__file__))
                        + "/../SimCodes/SimCodes"
                    )
                    + "/"
                ).replace("\\", "/")
                if self.verbose:
                    print(
                        "Found SimCodes Directory 1-up =",
                        self.global_parameters["simcodes_location"],
                    )
            elif os.path.isdir(
                os.path.abspath(
                    os.path.dirname(os.path.abspath(__file__)) + "/../SimCodes"
                )
                + "/"
            ):
                self.global_parameters["simcodes_location"] = (
                    os.path.abspath(
                        os.path.dirname(os.path.abspath(__file__)) + "/../SimCodes"
                    )
                    + "/"
                ).replace("\\", "/")
                if self.verbose:
                    print(
                        "Found SimCodes Directory 1-up =",
                        self.global_parameters["simcodes_location"],
                    )
            else:
                if self.verbose:
                    print("SimCodes not available - specify using simcodes=<location>")
                self.global_parameters["simcodes_location"] = None
        else:
            self.global_parameters["simcodes_location"] = os.path.join(
                os.path.abspath(simcodes), "./"
            )
        SimCodesLocation = self.global_parameters["simcodes_location"]

    def load_Elements_File(self, inp: str | list | tuple | dict) -> None:
        """
        Load a YAML file or list of YAML files with element definitions.
        The `elements` entry in this file(s) are then parsed and read into :attr:`~elementObjects`
        in order to build up the :class:`~SimulationFramework.Framework_objects.frameworkLattice` object.

        Parameters
        ----------
        inp: str or list or tuple or dict
            Input file or dict containing the lattice `elements`
        """
        if isinstance(inp, (list, tuple)):
            filename = inp
        else:
            filename = [inp]
        for f in filename:
            if os.path.isfile(f):
                with open(f, "r") as stream:
                    elements = yaml.safe_load(stream)["elements"]
            elif os.path.isfile(self.subdirectory + "/" + f):
                with open(self.subdirectory + "/" + f, "r") as stream:
                    elements = yaml.safe_load(stream)["elements"]
            else:
                with open(
                    self.global_parameters["master_lattice_location"] + f, "r"
                ) as stream:
                    elements = yaml.safe_load(stream)["elements"]
            for name, elem in list(elements.items()):
                self.read_Element(name, elem)

    def loadSettings(
            self,
            filename: str | None = None,
            settings: FrameworkSettings | None = None,
    ) -> None:
        """
        Load Lattice Settings from file or dictionary. These settings contain the lattice lines and
        their respective settings, YAML files and global parameters.

        Parameters
        ----------
        filename: str or None
            Name of .def file containing lattice definitions
        settings: FrameworkSettings or None
            Settings for the lattice
        """
        if isinstance(filename, str):
            self.settingsFilename = filename
            self.settings = FrameworkSettings()
            if os.path.isfile(filename):
                self.settings.loadSettings(filename)
            else:
                self.settings.loadSettings(
                    self.global_parameters["master_lattice_location"] + filename
                )
        elif isinstance(settings, FrameworkSettings):
            self.settingsFilename = settings.settingsFilename
            self.settings = settings

        self.globalSettings = self.settings["global"]
        if "generator" in self.settings and len(self.settings["generator"]) > 0:
            self.generatorSettings = self.settings["generator"]
            self.add_Generator(**self.generatorSettings)
        self.fileSettings = self.settings["files"] if "files" in self.settings else {}
        elements = self.settings["elements"]
        groups = (
            self.settings["groups"]
            if "groups" in self.settings and self.settings["groups"] is not None
            else {}
        )
        changes = (
            self.settings["changes"]
            if "changes" in self.settings and self.settings["changes"] is not None
            else {}
        )

        for name, elem in list(elements.items()):
            self.read_Element(name, elem)

        for name, elem in list(groups.items()):
            if "type" in elem:
                group = getattr(frameworkElements, elem["type"])(
                    name, self, global_parameters=self.global_parameters, **elem
                )
                self.groupObjects[name] = group

        for name, lattice in list(self.fileSettings.items()):
            self.read_Lattice(name, lattice)

        self.apply_changes(changes)

        self.original_elementObjects = {}
        for e in self.elementObjects:
            self.original_elementObjects[e] = deepcopy(self.elementObjects[e])
        self.original_elementObjects["generator"] = deepcopy(self.generator)

    def save_settings(
        self,
        filename: str | None = None,
        directory: str = ".",
        elements: dict | None = None,
    ) -> None:
        """
        Save Lattice Settings to a file.

        Parameters
        ----------
        filename: str or None
            Filename to which the settings will be saved; defaults to `settings.def`
        directory: str
            Directory to which the settings will be saved
        elements: dict or None
            Dictionary of :class:`~SimulationFramework.Framework_objects.frameworkElement` objects to save
        """
        if filename is None:
            pre, ext = os.path.splitext(os.path.basename(self.settingsFilename))
        else:
            pre, ext = os.path.splitext(os.path.basename(filename))
        if filename is None:
            filename = "settings.def"
        settings = self.settings.copy()
        if elements is not None:
            settings["elements"] = elements
        settings = convert_numpy_types(settings)
        with open(directory + "/" + filename, "w") as yaml_file:
            yaml.default_flow_style = True
            yaml.safe_dump(settings, yaml_file, sort_keys=False)

    def read_Lattice(self, name: str, lattice: dict) -> None:
        """
        Create an instance of a <code>Lattice class;
        see :class:`~SimulationFramework.Framework_objects.frameworkLattice` and its child classes.
        This instance is then added to the :attr:`~latticeObjects` dictionary.

        Parameters
        ----------
        name: str
            The name of the lattice line
        lattice: dict
            Dictionary containing settings for the lattice line
        """
        code = lattice["code"] if "code" in lattice else "astra"
        self.latticeObjects[name] = getattr(
            frameworkLattices, code.lower() + "Lattice"
        )(
            name,
            lattice,
            self.elementObjects,
            self.groupObjects,
            self.runSetup,
            self.settings,
            self.executables,
            self.global_parameters,
        )

    def detect_changes(
            self,
            elementtype: str | None = None,
            elements: list | None = None,
    ) -> dict:
        """
        Detect lattice changes from the original loaded lattice and return a dictionary of changes.

        Parameters
        ----------
        elementtype: str or None
            Element type to check; check all if None
        elements: list or None
            Elements to check; check all if None

        Returns
        -------
        dict
            Dictionary containing changes in the lattice, with element names and changed parameters
        """
        changedict = {}
        if elementtype is not None:
            changeelements = self.getElementType(elementtype, "objectname")
        elif elements is not None:
            changeelements = elements
        else:
            changeelements = ["generator"] + list(self.elementObjects.keys())
        if (
            len(changeelements) > 0
            and isinstance(changeelements[0], (list, tuple, dict))
            and len(changeelements[0]) > 1
        ):
            for ek in changeelements:
                new = None
                e, k = ek[:2]
                if e in self.elementObjects:
                    new = self.elementObjects[e]
                elif e in self.groupObjects:
                    new = self.groupObjects[e]
                if new is not None:
                    if e not in changedict:
                        changedict[e] = {}
                    changedict[e][k] = convert_numpy_types(getattr(new, k[0]))
        else:
            for e in changeelements:
                element = None
                if e in self.elementObjects:
                    element = self.elementObjects[e]
                elif e == "generator":
                    element = self.generator
                cond = False
                orig = self.original_elementObjects[e]
                new = element
                if isinstance(element, frameworkElement):
                    kval = [k for k in new.model_fields_set if k not in disallowed_changes]
                    new_model_fields = {k: getattr(new, k) for k in new.model_fields_set if
                                        k not in disallowed_changes}
                    orig_model_fields = {k: getattr(orig, k) for k in orig.model_fields_set if
                                         k not in disallowed_changes}
                    for k in kval:
                        if not k in list(orig_model_fields.keys()):
                            cond = True
                        elif new_model_fields[k] != orig_model_fields[k]:
                            cond = True
                if cond:
                    orig = self.original_elementObjects[e]
                    new = element
                    # try:
                    changedict[e] = {
                        k[0]: convert_numpy_types(getattr(new, k[0]))
                        for k in new
                        if k in orig and not getattr(new, k[0]) == getattr(orig, k[0]) and k[0] not in disallowed_changes
                    }
                    changedict[e].update(
                        {
                            k[0]: convert_numpy_types(getattr(new, k[0]))
                            for k in new
                            if k not in orig and k[0] not in disallowed_changes
                        }
                    )
                    if changedict[e] == {}:
                        del changedict[e]
                    # except Exception:
                    #     print("##### ERROR IN CHANGE ELEMS: ")  # , e, new)
                    #     pass
        return changedict

    def save_changes_file(
        self,
        filename: str | None = None,
        typ: str | None = None,
        elements: dict | None = None,
        dictionary: bool = False,
    ) -> dict | None:
        """
        Save a file, or returns a dictionary, of detected changes in the lattice from the loaded version.

        Parameters
        ----------
        filename: str or None
            Name of file containing changes; defaults to `changes.yaml`
        typ: str or None
            Element types to check; if `None`, check all
        elements: dict or None
            Dictionary containing elements and parameters to check; if `None`, check all
        dictionary: bool
            Flag to return changes as dictionary; if False, save a YAML file

        Returns
        -------
        dict or None
            If `dictionary`, return a dict; otherwise, save a file and return `None`
        """
        if filename is None:
            pre, ext = os.path.splitext(os.path.basename(self.settingsFilename))
            filename = pre + "_changes.yaml"
        changedict = self.detect_changes(elementtype=typ, elements=elements)
        if dictionary:
            return changedict
        else:
            with open(filename, "w") as yaml_file:
                yaml.default_flow_style = True
                yaml.dump(changedict, yaml_file)

    def save_lattice(
        self,
        lattice: str | None = None,
        filename: str | None = None,
        directory: str = ".",
        dictionary: bool = False,
    ) -> dict | None:
        """
        Save lattice to a file, or return a dictionary containing the lattice elements

        Parameters
        ----------
        lattice: str or None
            Name of lattice file; if `None`, sets to the name of the lattice
        filename: str or None
            Name of the file to be saved; if `None`, sets to the name of the lattice + '_lattice.yaml'
        directory: str
            Directory to which the file will be saved
        dictionary: bool
            Flag to save lattice as dictionary; if False, save a YAML file

        Returns
        -------
        dict or None
            If `dictionary`, return a dict; otherwise, save a file and return `None`
        """
        if filename is None:
            pre, ext = os.path.splitext(os.path.basename(self.settingsFilename))
        else:
            pre, ext = os.path.splitext(os.path.basename(filename))
        dic = dict({"elements": dict()})
        latticedict = dic["elements"]
        if lattice is None:
            elements = list(self.elementObjects.keys())
            filename = pre + ".yaml"
        else:
            if self.latticeObjects[lattice].elements is None:
                return
            elements = list(self.latticeObjects[lattice].elements.keys())
            filename = pre + "_" + lattice + "_lattice.yaml"
        for e in elements:
            new = self.elementObjects[e]
            # try:
            if (
                "subelement" in new and not new["subelement"]
            ) or "subelement" not in new:
                # latticedict.update({e: {}})
                # for k in new:
                #     if k not in disallowed:
                #         latticedict[e][k[0].replace("object", "")] = convert_numpy_types(getattr(new, k[0]))
                latticedict[e] = {
                    k[0].replace("object", ""): convert_numpy_types(getattr(new, k[0]))
                    for k in new
                    if k[0] not in disallowed and getattr(new, k[0]) is not None
                }
                if "sub_elements" in new:
                    latticedict[e].update({"sub_elements": {}})
                    for subelem in new["sub_elements"]:
                        newsub = self.elementObjects[subelem]
                        latticedict[e]["sub_elements"][subelem] = {
                            k[0].replace("object", ""): convert_numpy_types(
                                getattr(newsub, k[0])
                            )
                            for k in newsub
                            if k not in disallowed
                        }
            # except Exception:
            #     print("##### ERROR IN CHANGE ELEMS: ", e, new)
            #     pass
        if dictionary:
            return latticedict
        else:
            with open(directory + "/" + filename, "w") as yaml_file:
                yaml.default_flow_style = True
                yaml.dump(dic, yaml_file)

    def load_changes_file(
            self,
            filename: str | tuple | list | None = None,
            apply: bool = True,
            verbose: bool = False,
    ) -> dict | list:
        """
        Loads a saved changes file and applies the settings to the current lattice.
        Returns a list of changes.
        See :func:`~apply_changes`.

        Parameters
        ----------
        filename: str or list or tuple or None
            Changes filename to save; if `None`, base it on the settings filename
        apply: bool
            Flag to apply the changes
        verbose: bool
            Flag to print the changes applied

        Returns
        -------
        dict or list
            If `filename` is a `list` or `tuple`, call this function again
            If `filename` is `None` or a `str`, return the dictionary of changes.
        """
        if isinstance(filename, (tuple, list)):
            return [self.load_changes_file(c) for c in filename]
        else:
            if filename is None:
                pre, ext = os.path.splitext(os.path.basename(self.settingsFilename))
                filename = pre + "_changes.yaml"
            with open(filename, "r") as infile:
                changes = dict(yaml.safe_load(infile))
            if apply:
                self.apply_changes(changes, verbose=verbose)
            return changes

    def apply_changes(self, changes: dict, verbose: bool = False) -> None:
        """
        Applies a dictionary of changes to the current lattice.

        Parameters
        ----------
        changes: dict
            Dictionary of changes to elements, keyed by element name and containing parameters and values
            to change
        verbose: bool
            Flag to indicate which elements are being modified
        """
        for e, d in list(changes.items()):
            # print 'found change element = ', e
            if e in self.elementObjects:
                # print 'change element exists!'
                for k, v in list(d.items()):
                    self.modifyElement(e, k, v)
                    if verbose:
                        print("modifying ", e, "[", k, "]", " = ", v)
            if e in self.groupObjects:
                # print ('change group exists!')
                for k, v in list(d.items()):
                    self.groupObjects[e].change_Parameter(k, v)
                    if verbose:
                        print("modifying ", e, "[", k, "]", " = ", v)

    def check_lattice(self, decimals: int = 4) -> bool:
        """
        Checks that there are no positioning errors in the lattice.

        Parameters
        ----------
        decimals: int
            Number of decimals to check errors

        Returns
        -------
        bool
            True if errors are detected
        """
        noerror = True
        for elem in self.elementObjects.values():
            start = elem.position_start
            end = elem.end
            length = elem.length
            theta = elem.global_rotation[2]
            if elem.objecttype == "dipole" and abs(float(elem.angle)) > 0:
                angle = float(elem.angle)
                rho = length / angle
                clength = np.array([rho * (np.cos(angle) - 1), 0, rho * np.sin(angle)])
            else:
                clength = np.array([0, 0, length])
            cend = start + np.dot(clength, _rotation_matrix(theta))
            if not np.round(cend - end, decimals=decimals).any() == 0:
                noerror = False
                print("check_lattice error:", elem.objectname, cend, end, cend - end)
        return noerror

    def check_lattice_drifts(self, decimals: int = 4) -> bool:
        """
        Checks that there are no positioning errors in the lattice.

        Parameters
        ----------
        decimals: int
            Number of decimals to check errors

        Returns
        -------
        bool
            True if errors are detected
        """
        noerror = True
        for elem in self.elementObjects.values():
            start = elem.position_start
            end = elem.position_end
            length = elem.length
            theta = elem.global_rotation[2]
            if elem.objecttype == "dipole" and abs(float(elem.angle)) > 0:
                angle = float(elem.angle)
                rho = length / angle
                clength = np.array([rho * (np.cos(angle) - 1), 0, rho * np.sin(angle)])
            else:
                clength = np.array([0, 0, length])
            cend = start + np.dot(clength, _rotation_matrix(theta))
            if not np.round(cend - end, decimals=decimals).any() == 0:
                noerror = False
                print(
                    "check_lattice_drifts error:",
                    elem.objectname,
                    cend,
                    end,
                    cend - end,
                )
        return noerror

    def change_Lattice_Code(
            self,
            latticename: str,
            code: str,
            exclude: str | list | tuple | None = None,
    ) -> None:
        """
        Changes the tracking code for a given lattice.

        Parameters
        ----------
        latticename: str
            Name of the lattice line defined in the :attr:`~latticeObjects`
        code: str
            Simulation code to use for `latticename`; can be `All`
        exclude:
            Exclude certain lines from this function
        """
        if latticename == "All":
            [self.change_Lattice_Code(lo, code, exclude) for lo in self.latticeObjects]
        elif isinstance(latticename, (tuple, list)):
            [self.change_Lattice_Code(ln, code, exclude) for ln in latticename]
        else:
            if not latticename == "generator" and not (
                latticename == exclude
                or (isinstance(exclude, (list, tuple)) and latticename in exclude)
            ):
                # print('Changing lattice ', name, ' to ', code.lower())
                currentLattice = self.latticeObjects[latticename]
                self.latticeObjects[latticename] = getattr(
                    frameworkLattices, code.lower() + "Lattice"
                )(
                    currentLattice.objectname,
                    currentLattice.file_block,
                    self.elementObjects,
                    self.groupObjects,
                    self.runSetup,
                    self.settings,
                    self.executables,
                    self.global_parameters,
                )

    def read_Element(
            self,
            elementname: str,
            element: dict,
            subelement: bool = False,
            parent: str = None,
    ) -> None:
        """
        Reads an element definition and creates the element and any sub-elements

        Parameters
        ----------
        elementname: str
            Name of element
        element: dict
            Dictionary containins the elements
        subelement: bool
            Flag to indicate whether this element is a sub-element of another,
            i.e. a solenoid around an RF cavity
        parent: str
            Name of the parent element if this element is a subelement
        """
        if elementname == "filename":
            self.load_Elements_File(element)
        else:
            if subelement:
                if "subelement" in element:
                    del element["subelement"]
                if "parent" in element:
                    del element["parent"]
                self.add_Element(elementname, subelement=True, parent=parent, **element)
            else:
                self.add_Element(elementname, **element)
            if "sub_elements" in element:
                for name, elem in list(element["sub_elements"].items()):
                    self.read_Element(name, elem, subelement=True, parent=elementname)

    def add_Element(
            self,
            name: str | None = None,
            typ: str | None = None,
            **kwargs,
    ) -> frameworkElement | None:
        """
        Instantiates and adds the element definition to :attr:`~elementObjects`

        Parameters
        ----------
        name: str
            Name of the element to add
        typ: str
            Type of element; see :ref:`framework-elements`

        Returns
        -------
        :class:`~SimulationFramework.Framework_objects.frameworkElement`
            The newly created element

        Raises
        ------
        NameError
            If the element does not have a `name`
        Exception
            In case of errors when generating the element
        """
        if name is None:
            if "name" not in kwargs:
                raise NameError("Element does not have a name")
            else:
                name = kwargs["name"]
        # try:
        if typ is None:
            typ = kwargs["type"]
        try:
            element = getattr(frameworkElements, typ)(
                objectname=name,
                objecttype=typ,
                global_parameters=self.global_parameters,
                **kwargs,
            )
            element.update_field_definition()
            self.elementObjects[name] = element
            return element
        except Exception as ex:
            print("add_Element error:", ex)
            print("add_Element error:", typ, name, kwargs)
        return
        # except Exception as e:
        #     raise NameError('Element \'%s\' does not exist' % type)

    def replace_Element(
            self, name: str | None = None,
            typ: str | None = None,
            **kwargs
    ) -> frameworkElement:
        """
        Replaces an element type with a new type and updates the definitions

        Parameters
        ----------
        name: str
            Name of the element to replace
        typ: str
            Type of element; see :ref:`framework-elements`

        Returns
        -------
        :class:`~SimulationFramework.Framework_objects.frameworkElement`
            The replaced element

        Raises
        ------
        NameError
            If the element does not have a `name`
        """
        if name is None:
            if "name" not in kwargs:
                raise NameError("Element does not have a name")
            else:
                name = kwargs["name"]
        original_element = self.getElement(name)
        original_properties = {
            a: original_element[a]
            for a in original_element.objectproperties
            if a != "objectname" and a != "objecttype"
        }
        new_properties = merge_two_dicts(kwargs, original_properties)
        element = getattr(frameworkElements, typ)(name, typ, **new_properties)
        # print element
        self.elementObjects[name] = element
        return element
        # except Exception as e:
        #     raise NameError('Element \'%s\' does not exist' % type)

    def getElement(
            self,
            element: str,
            param: str | None = None,
    ) -> dict | Any | frameworkElement:
        """
        Returns the element object or a parameter of that element

        Parameters
        ----------
        element: str
            Name of element to get
        param: str or None
            Parameter to retrieve; if `None`, return the entire element

        Returns
        -------
        dict or Any or :class:`~SimulationFramework.Framework_objects.frameworkElement
            Get the `param` associated with `element`, or the entire element, or an empty dictionary if
            the element does not exist in the entire lattice
        """
        if self.__getitem__(element) is not None:
            if param is not None:
                param = param.lower()
                return getattr(self.__getitem__(element), param)
            else:
                return self.__getitem__(element)
        else:
            print(("WARNING: Element ", element, " does not exist"))
            return {}

    def getElementType(
            self,
            typ: str | list | tuple,
            param: str | None = None,
    ) -> dict | list | Any:
        """
        Gets all elements of the specified type, or the parameter of each of those elements

        Parameters
        ----------
        typ: list or str or tuple
            Type or list of types to get
        param: str or None
            Parameters to retrieve; if `None`, get the entire object

        Returns
        -------
        dict or list or Any
            Get `param` for all elements, or all elements, or recall this function recursively if
            `param` is a list or tuple
        """
        if isinstance(typ, (list, tuple)):
            return [self.getElementType(t, param=param) for t in typ]
        if isinstance(param, (list, tuple)):
            return zip(*[self.getElementType(typ, param=p) for p in param])
            # return [item for sublist in all_elements for item in sublist]
        return [
            (
                {"name": element, **self.elementObjects[element].model_dump()}
                if param is None
                else getattr(self.elementObjects[element], param)
            )
            for element in list(self.elementObjects.keys())
            if self.elementObjects[element].objecttype.lower() == typ.lower()
        ]

    def setElementType(
            self,
            typ: str,
            setting: str,
            values: Any,
    ) -> None:
        """
        Modifies the specified parameter of each element of a given type

        Parameters
        ----------
        typ: str
            All elements of a given type to set
        setting: str
            Parameter in those elements to set
        values: Any
            Values to set on those elements

        Raises
        ------
        ValueError
            If there is a mismatch between the length of `values` and the number of elements of that type
        """
        elems = self.getElementType(typ)
        if len(elems) == len(values):
            for e, v in zip(elems, values):
                setattr(self[e["name"]], setting, v)
        else:
            raise ValueError

    def modifyElement(
            self,
            elementName: str,
            parameter: str | list | dict,
            value: Any = None,
    ) -> None:
        """
        Modifies an element parameter

        Parameters
        ----------
        elementName: str
            Name of element to modify
        parameter: list or str or dict
            Parameter to modify
        value:
            Value to set on that element
        """
        if isinstance(parameter, dict) and value is None:
            for p, v in parameter.items():
                self.modifyElement(elementName, p, v)
        elif isinstance(parameter, list) and isinstance(value, list):
            if len(parameter) != len(value):
                raise ValueError("parameter and value must be of the same length")
            for p, v in zip(parameter, value):
                self.modifyElement(elementName, p, v)
        elif elementName in self.groupObjects:
            self.groupObjects[elementName].change_Parameter(parameter, value)
        elif elementName in self.elementObjects:
            setattr(self.elementObjects[elementName], parameter, value)

    def modifyElements(
            self, elementNames: str | list,
            parameter: str | list | dict,
            value: Any = None,
    ) -> None:
        """
        Modifies parameters for multiple elements

        Parameters
        ----------
        elementNames: str or list
            Name(s) of element to modify
        parameter: list or str or dict
            Parameter to modify
        value:
            Value to set on those elements
        """
        if isinstance(elementNames, str) and elementNames.lower() == "all":
            elementNames = self.elementObjects.keys()
        for elem in elementNames:
            self.modifyElement(elem, parameter, value)

    def modifyElementType(
            self,
            elementType: str,
            parameter: str,
            value: Any,
    ) -> None:
        """
        Modifies an element or a list of elements of a given type

        Parameters
        ----------
        elementType: str
            Type of element to modify
        parameter: str
            Parameter of that element type to modify
        value: Any
            Value to set on that element(s)
        """
        elems = self.getElementType(elementType)
        for elementName in [e["name"] for e in elems]:
            self.modifyElement(elementName, parameter, value)

    def modifyLattice(
            self,
            latticeName: str,
            parameter: str | list | dict,
            value: Any = None,
    ) -> None:
        """
        Modify a lattice definition,

        Parameters
        ----------
        latticeName: str
            Name of lattice to modify
        parameter: str or list or dict
            Parameter(s) to update with their values
        value: Any
            Value to update
        """
        if isinstance(parameter, dict) and value is None:
            for p, v in parameter.items():
                self.modifyLattice(latticeName, p, v)
        elif isinstance(parameter, list) and isinstance(value, list):
            for p, v in parameter:
                self.modifyLattice(latticeName, p, v)
        elif latticeName in self.latticeObjects:
            setattr(self.latticeObjects[latticeName], parameter, value)

    def modifyLattices(
            self,
            latticeNames: str | list,
            parameter: str | list | dict,
            value: Any = None,
    ) -> None:
        """
        Modify a lattice definition for a list of lattices

        Parameters
        ----------
        latticeNames: str or list
            Name of lattice(s) to modify
        parameter: str or list or dict
            Parameter(s) to update with their values
        value: Any
            Value to update
        """
        if isinstance(latticeNames, str) and latticeNames.lower() == "all":
            latticeNames = self.latticeObjects.keys()
        for latt in latticeNames:
            self.modifyLattice(latt, parameter, value)

    def add_Generator(
            self,
            default: str | None = None,
            **kwargs,
    ) -> None:
        """
        Add a file generator based on a keyword dictionary.
        Sets :attr:`~generator` to the :class:`~SimulationFramework.Codes.Generators.Generators.frameworkGenerator`.

        Also sets the "generator" in :attr:`~latticeObjects` to this generator.

        Parameters
        ----------
        default: str or None
            Name of generator code
        """
        if "code" in kwargs:
            if kwargs["code"].lower() == "gpt":
                code = GPTGenerator
            else:
                code = ASTRAGenerator
        else:
            code = ASTRAGenerator
        if default in generator_keywords["defaults"]:
            self.generator = code(
                self.executables,
                self.global_parameters,
                **merge_two_dicts(kwargs, generator_keywords["defaults"][default]),
            )
        else:
            self.generator = code(self.executables, self.global_parameters, **kwargs)
        self.latticeObjects["generator"] = self.generator

    def change_generator(
            self,
            generator: str,
    ) -> None:
        """
        Changes the generator from one type to another.

        Parameters
        ----------
        generator: str
            The generator code to which the generator object should be changed.
        """
        old_kwargs = self.generator.kwargs
        if generator.lower() == "gpt":
            generator = GPTGenerator(
                self.executables, self.global_parameters, **old_kwargs
            )
        else:
            generator = ASTRAGenerator(
                self.executables, self.global_parameters, **old_kwargs
            )
        self.latticeObjects["generator"] = generator

    def loadParametersFile(self, file):
        pass

    def saveParametersFile(self, file: str, parameters: dict | list | tuple) -> None:
        """Saves a list of parameters to a file"""
        output = {}
        if isinstance(parameters, dict):
            try:
                output.update({parameters["name"]: {k1: v1 for k1, v1 in parameters.items() if v1 not in disallowed}})
            except KeyError:
                warn("parameters dictionary must contain 'name' key")
        elif isinstance(parameters, (list, tuple)):
            try:
                for k in parameters:
                    output.update({k["name"]: {}})
                    output[k["name"]].update({subk: k[subk] for subk in k if subk not in disallowed})
            except TypeError:
                warn("parameters must be a dictionary or a list of dictionaries containing a 'name' key")
        else:
            warn("could not parse parameters; they should be a dict, list or tuple")
        with open(file, "w") as yaml_file:
            yaml.default_flow_style = True
            yaml.dump(output, yaml_file)

    def set_lattice_prefix(
            self,
            lattice: str,
            prefix: str,
    ) -> None:
        """
        Sets the 'prefix' parameter for a lattice in :attr:`~latticeObjects`,
        which determines where it looks for its starting beam distribution.


        Parameters
        ----------
        lattice: str
            Name of lattice
        prefix: str
            Lattice prefix
        """
        if lattice in self.latticeObjects:
            self.latticeObjects[lattice].prefix = prefix
        else:
            warn(f"{lattice} not found in latticeObjects; valid lattices are {list(self.latticeObjects.keys())}")

    def set_lattice_sample_interval(
            self,
            lattice: str,
            interval: int,
    ) -> None:
        """
        Sets the 'sample_interval' parameter for a lattice, which determines the sampling of the distribution.
        See :attr:`~SimulationFramework.Framework_objects.frameworkLattice.sample_interval`.

        Parameters
        ----------
        lattice: str
            Name of lattice
        interval: int
            Sampling interval in units of 2 ** (3 * interval)
        """
        if lattice in self.latticeObjects:
            self.latticeObjects[lattice].sample_interval = interval
        else:
            warn(f"{lattice} not found in latticeObjects; valid lattices are {list(self.latticeObjects.keys())}")

    def __getitem__(self, key: str) -> Any:
        if key in list(self.elementObjects.keys()):
            return self.elementObjects.get(key)
        elif key in list(self.latticeObjects.keys()):
            return self.latticeObjects.get(key)
        elif key in list(self.groupObjects.keys()):
            return self.groupObjects.get(key)
        else:
            try:
                return getattr(self, key)
            except Exception:
                return None

    @property
    def elements(self) -> list:
        """
        Returns a list of all element names from :attr:`~elementObjects`

        Returns
        -------
        list
            List of element names
        """
        return list(self.elementObjects.keys())

    @property
    def groups(self) -> list:
        """
        Returns a list of all group names from :attr:`~groupObjects`

        Returns
        -------
        list
            List of group names
        """
        return list(self.groupObjects.keys())

    @property
    def lines(self) -> list:
        """
        Returns a list of all lattice names

        Returns
        -------
        list
            List of lattice names
        """
        return list(self.latticeObjects.keys())

    @property
    def lattices(self) -> list:
        """
        Returns a list of all lattice names

        Returns
        -------
        list
            List of lattice names
        """
        return self.lines

    @property
    def commands(self) -> list:
        """
        Returns a list of all command object names

        Returns
        -------
        list
            List of command object names
        """
        return list(self.commandObjects.keys())

    def getSValues(self) -> list:
        """
        Returns a list of S values for the current lattice from :attr:`~latticeObjects`;
        see :func:`~SimulationFramework.Framework_objects.frameworkLattice.getSValues`.

        Returns
        -------
        list
            S values for all elements
        """
        s0 = 0
        allS = []
        for lo in self.latticeObjects.values():
            try:
                latticeS = [a + s0 for a in lo.getSValues()]
                allS = allS + latticeS
                s0 = allS[-1]
            except Exception:
                pass
        return allS

    def getSValuesElements(self) -> list:
        """
        Returns a list of (name, element, s) tuples for the current machine from :attr:`~latticeObjects`;
        see :func:`~SimulationFramework.Framework_objects.frameworkLattice.getSNamesElems`.

        Returns
        -------
        list
            Element names, element object and its S position
        """
        s0 = 0
        allS = []
        for lo in self.latticeObjects:
            if not lo == "generator":
                names, elems, svals = self.latticeObjects[lo].getSNamesElems()
                latticeS = [a + s0 for a in svals]
                selems = list(zip(names, elems, latticeS))
                allS = allS + selems
                s0 = latticeS[-1]
        return allS

    def getZValuesElements(self) -> list:
        """
        Returns a list of (name, element, z) tuples for the current machine from :attr:`~latticeObjects`;
        see :func:`~SimulationFramework.Framework_objects.frameworkLattice.getZNamesElems`.

        Returns
        -------
        list
            Element names, element object and its Z position
        """
        allZ = []
        for lo in self.latticeObjects:
            if not lo == "generator":
                names, elems, zvals = self.latticeObjects[lo].getZNamesElems()
                zelems = list(zip(names, elems, zvals))
                allZ = allZ + zelems
        return list(sorted(allZ, key=lambda x: x[2][0]))

    def track(
        self,
        files: list | None = None,
        startfile: str | None = None,
        endfile: str | None = None,
        preprocess: bool = True,
        write: bool = True,
        track: bool = True,
        postprocess: bool = True,
        save_summary: bool = True,
        frameworkDirec: bool = False,
        check_lattice: bool = True,
    ) -> Any | None:
        """
        Tracks the current machine, or a subset based on the 'files' list.
        The lattice is checked (:func:`~check_lattice`) and saved (:func:`~save_lattice`), the settings file
        is saved (:func:`~save_settings`), and then each line in the lattice is tracked with
        the code specified.

        Parameters
        ----------
        files: list or None
            List of files (lattice names) to track; if `None`, track all
        startfile: str or None
            Initial lattice name for tracking; if `None`, track all
        endfile: str or None
            Final lattice name for tracking; if `None`, track all
        preprocess: bool
            Call :func:`~SimulationFramework.Framework_objects.frameworkLattice.preProcess` before
            tracking each line
        write: bool
            Write each lattice file
        track: bool
            Track each lattice
        postprocess: bool
            Call :func:`~SimulationFramework.Framework_objects.frameworkLattice.postProcess` after
            tracking each line
        save_summary: bool
            Save beam and Twiss summary files
        frameworkDirec: bool
            If True, return a :class:`~SimulationFramework.Framework.frameworkDirectory` object
        check_lattice: bool
            Call :func:`~check_lattice` before tracking

        Returns
        -------
        :class:`~SimulationFramework.Framework.frameworkDirectory` or None
            Framework directory object if `frameworkDirec` is True
        """
        if check_lattice:
            if not self.check_lattice():
                raise Exception("Lattice Error - check definitions")
        self.save_lattice(directory=self.subdirectory, filename="lattice.yaml")
        self.save_settings(
            directory=self.subdirectory,
            filename="settings.def",
            elements={"filename": "lattice.yaml"},
        )
        self.tracking = True
        self.progress = 0
        if files is None:
            files = (
                ["generator"] + self.lines
                if not hasattr(self, "generator")
                else self.lines
            )
        if startfile is not None and startfile in files:
            index = files.index(startfile)
            files = files[index:]
        if endfile is not None and endfile in files:
            index = files.index(endfile)
            files = files[: index + 1]
        if self.verbose:
            pbar = tqdm(total=len(files) * 4)
        percentage_step = 100 / len(files)
        for i in range(len(files)):
            base_percentage = 100 * (i / len(files))
            lattice_name = files[i]
            self.progress = base_percentage
            if lattice_name == "generator" and hasattr(self, "generator"):
                latt = self.generator
                base_description = "Generator[" + self.generator.code + "]"
            else:
                latt = self.latticeObjects[lattice_name]
                base_description = lattice_name + "[" + latt.code + "]"
            if self.verbose:
                pbar.set_description(base_description + ":              ")  # noqa E701
            if preprocess and lattice_name != "generator":
                if self.verbose:
                    pbar.set_description(
                        base_description + ": pre-process  "
                    )  # noqa E701
                latt.preProcess()
                self.progress = base_percentage + 0.25 * percentage_step
            if self.verbose:
                pbar.update()  # noqa E701
            if write:
                if self.verbose:
                    pbar.set_description(
                        base_description + ": write        "
                    )  # noqa E701
                latt.write()
                self.progress = base_percentage + 0.5 * percentage_step
            if self.verbose:
                pbar.update()  # noqa E701
            if track:
                if self.verbose:
                    pbar.set_description(
                        base_description + ": track        "
                    )  # noqa E701
                latt.run()
                self.progress = base_percentage + 0.75 * percentage_step
            if self.verbose:
                pbar.update()  # noqa E701
            if postprocess:
                if self.verbose:
                    pbar.set_description(
                        base_description + ": post-process "
                    )  # noqa E701
                latt.postProcess()
                self.progress = base_percentage + 1 * percentage_step
            if self.verbose:
                pbar.update()  # noqa E701
        if self.verbose:
            pbar.set_description(base_description + ": Finished! ")
            pbar.close()
        if save_summary:
            self.save_summary_files()
        self.tracking = False
        if frameworkDirec:
            return frameworkDirectory(
                directory=self.subdirectory,
                twiss=True,
                beams=True,
                verbose=self.verbose,
            )

    def postProcess(
        self,
        files: list | None = None,
        startfile: str | None = None,
        endfile: str | None = None,
    ) -> None:
        """
        Post-processes the tracking files and converts them to HDF5.
        See :func:`~SimulationFramework.Framework_objects.frameworkLattice.postProcess` and the same function
        in the child classes for specific codes.

        Parameters
        ----------
        files: list or None
            List of lattice names; if `None`, process all
        startfile: str or None
            Starting lattice object; if `None`, process from the start
        endfile: str or None
            End lattice object; if `None`, process to the end
        """
        if files is None:
            files = (
                ["generator"] + self.lines
                if not hasattr(self, "generator")
                else self.lines
            )
        if startfile is not None and startfile in files:
            index = files.index(startfile)
            files = files[index:]
        if endfile is not None and endfile in files:
            index = files.index(endfile)
            files = files[: index + 1]
        for i in range(len(files)):
            latt = files[i]
            if latt == "generator" and hasattr(self, "generator"):
                self.generator.postProcess()
            else:
                self.latticeObjects[latt].postProcess()

    def save_summary_files(self, twiss: bool = True, beams: bool = True) -> None:
        """
        Saves HDF5 summary files for the Twiss and/or Beam files using
        :func:`~SimulationFramework.Modules.Twiss.load_directory` and
        :func:`~SimulationFramework.Modules.Beams.save_HDF5_summary_file`

        Parameters
        ----------
        twiss: bool
            If True, save `Twiss_Summary.hdf5` in :attr:`~subdirectory`
        beams: bool
            If True, save `Beam_Summary.hdf5` in :attr:`~subdirectory`
        """
        if twiss:
            t = rtf.load_directory(self.subdirectory)
            t.save_HDF5_twiss_file(self.subdirectory + "/" + "Twiss_Summary.hdf5")
        if beams:
            rbf.save_HDF5_summary_file(
                self.subdirectory, self.subdirectory + "/" + "Beam_Summary.hdf5"
            )

    def pushRunSettings(self) -> None:
        """
        Updates the 'Run Settings' in each of the lattices
        """
        for ln, latticeObject in self.latticeObjects.items():
            if isinstance(latticeObject, tuple(latticeClasses)):
                latticeObject.updateRunSettings(self.runSetup)

    def setNRuns(self, nruns: int) -> None:
        """
        Sets the number of simulation runs to a new value for all lattice objects.
        See :func:`~SimulationFramework.Framework.runSetup.setNRuns`.

        Parameters
        ----------
        nruns: int
            Number of runs to set up
        """
        self.runSetup.setNRuns(nruns)
        self.pushRunSettings()

    def setSeedValue(self, seed: int) -> None:
        """
        Sets the random number seed to a new value for all lattice objects

        See :func:`~SimulationFramework.Framework.runSetup.setSeedValue`.

        Parameters
        ----------
        seed: int
            Random number seed
        """
        self.runSetup.setSeedValue(seed)
        self.pushRunSettings()

    def loadElementErrors(self, file: str) -> None:
        """
        Load element errors file; see :func:`~SimulationFramework.Framework.runSetup.loadElementErrors`

        Parameters
        ----------
        file: str
            Errors file
        """
        self.runSetup.loadElementErrors(file)
        self.pushRunSettings()

    def setElementScan(
            self,
            name: str,
            item: str,
            scanrange: list,
            multiplicative: bool = False,
    ) -> None:
        """
        Define a parameter scan for a single parameter of a given machine element.
        See :class:`~SimulationFramework.Framework.runSetup.setElementScan`

        Parameters
        ----------
        name: str
            Name of element to scan
        item: str
            Parameter of that element to scan
        scanrange: list
            List of values to set
        multiplicative: bool
            Flag to indicate whether settings are multiplicative or additive with respect to the original value
        """
        self.runSetup.setElementScan(
            name=name, item=item, scanrange=scanrange, multiplicative=multiplicative
        )
        self.pushRunSettings()

    def _addLists(self, list1: list, list2: list) -> list:
        """Adds elements piecewise in two lists"""
        return [a + b for a, b in zip(list1, list2)]

    def offsetElements(
        self, x: int | float = 0, y: int | float = 0, z: int | float = 0
    ) -> None:
        """
        Moves all elements by the set amount in (x, y, z) space.
        Updates :attr:`~elementObjects`.

        Parameters
        ----------
        x: int | float
            x offset
        y: int | float
            y offset
        z: int | float
            z offset
        """
        offset = [x, y, z]
        for latt in self.lines:
            if (
                self.latticeObjects[latt].file_block is not None
                and "output" in self.latticeObjects[latt].file_block
                and "zstart" in self.latticeObjects[latt].file_block["output"]
            ):
                self.latticeObjects[latt].file_block["output"]["zstart"] += z
        for elem in self.elements:
            self.elementObjects[elem].position_start = self._addLists(
                self.elementObjects[elem].position_start, offset
            )
            self.elementObjects[elem].position_end = self._addLists(
                self.elementObjects[elem].position_end, offset
            )


class frameworkDirectory(BaseModel):
    """
    Class to load a tracking run from a directory and read the Beam and Twiss files and make them available
    """

    directory: str | None = None
    """Directory from which to load beam and Twiss files"""

    twiss: bool | rtf.twiss = True
    """Flag to indicate whether to load Twiss files"""

    beams: bool | rbf.beamGroup = False
    """Flag to indicate whether to load beam files"""

    verbose: bool = False
    """Flag to print status updates"""

    settings: str = "settings.def"
    """Framework settings filename"""

    changes: str = "changes.yaml"
    """Lattice changes filename"""

    rest_mass: float | None = None
    """Particle rest mass"""

    framework: Framework | None = None
    """:class:`~SimulationFramework.Framework.Framework` instance"""

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super(frameworkDirectory, self).__init__(
            *args,
            **kwargs,
        )
        if not isinstance(self.framework, Framework):
            directory = "." if self.directory is None else os.path.abspath(self.directory)
            self.framework = Framework(**kwargs,)
            self.framework.loadSettings(directory + "/" + self.settings)
        else:
            self.framework = self.framework
            if self.directory is None:
                directory = os.path.abspath(self.framework.subdirectory)
            else:
                directory = self.directory

        if os.path.exists(directory + "/" + self.changes):
            self.framework.load_changes_file(directory + "/" + self.changes)
        if self.beams:
            self.beams = rbf.load_HDF5_summary_file(
                os.path.join(directory, "Beam_Summary.hdf5")
            )
            if len(self.beams) < 1:
                print("No Summary File! Globbing...")
                self.beams = rbf.load_directory(directory)
            if self.rest_mass is None:
                if len(self.beams.param("particle_rest_energy")) > 0:
                    rest_mass = self.beams.param("particle_rest_energy")[0][0]
                else:
                    rest_mass = constants.m_e
            else:
                rest_mass = self.rest_mass
            self.twiss = rtf.twiss(rest_mass=rest_mass)
        else:
            self.beams = None
            self.twiss = rtf.twiss()
        if self.twiss:
            self.twiss.load_directory(directory)

    if use_matplotlib:

        def plot(self, *args, **kwargs):
            """
            Return a plot object; see :func:`~SimulationFramework.Modules.plotting.plotting.plot`.
            """
            return groupplot.plot(self, *args, **kwargs)

        def general_plot(self, *args, **kwargs):
            """
            Return a general_plot object; see :func:`~SimulationFramework.Modules.plotting.plotting.general_plot`.
            """
            return groupplot.general_plot(self, *args, **kwargs)

    def __repr__(self):
        return repr(
            {"framework": self.framework, "twiss": self.twiss, "beams": self.beams}
        )

    def save_summary_files(self, twiss: bool = True, beams: bool = True):
        """
        Save summary files in framework directory;
        see :func:`~SimulationFramework.Framework.Framework.save_summary_files`.

        Parameters
        ----------
        twiss: bool
            If True, save `Twiss_Summary.hdf5` in :attr:`~directory`
        beams: bool
            If True, save `Beam_Summary.hdf5` in :attr:`~directory`
        """
        self.framework.save_summary_files(twiss=twiss, beams=beams)

    def getScreen(self, screen: str) -> rbf.beam | None:
        """
        Get a beam object for the given screen;
        see :func:`~SimulationFramework.Modules.Beams.beamGroup.getScreen`

        Parameters
        ----------
        screen: str
            Name of screen

        Returns
        -------
        :class:`~SimulationFramework.Modules.Beams.beam`
            The beam object from `screen`

        Raises
        ------
        ValueError
            If `beams` is not a :class:`~SimulationFramework.Modules.Beams.beamGroup` object
        """
        if isinstance(self.beams, rbf.beamGroup):
            return self.beams.getScreen(screen)
        else:
            raise ValueError("Beam files have not been read in")

    def getScreenNames(self) -> dict:
        """
        Get beam objects from all screens

        Returns
        -------
        Dict
            The :class:`~SimulationFramework.Modules.Beams.beam` objects from the screen keyed by name

        Raises
        ------
        ValueError
            If `beams` is not a :class:`~SimulationFramework.Modules.Beams.beamGroup` object
        """
        if isinstance(self.beams, rbf.beamGroup):
            return self.beams.getScreens()
        else:
            raise ValueError("Beam files have not been read in")

    def element(self, element: str, field: str | None = None) -> Any | frameworkElement:
        """
        Get an element definition from the framework object.

        Parameters
        ----------
        element: str
            Element to retrieve
        field: str | None
            Field of that element to retrieve

        Returns
        -------
        Any or frameworkElement
            Get the `field` of `element`, or the entire
            :class:`~SimulationFramework.Framework_objects.frameworkElement` if not `field`
        """
        elem = self.framework.getElement(element)
        if field:
            try:
                return getattr(elem, field)
            except AttributeError:
                warn(f"{elem} does not have field {field}; returning entire element")
                return elem
        else:
            pprint(
                {
                    k.replace("object", ""): v
                    for k, v in elem.items()
                    if k not in disallowed
                }
            )
            return elem


def load_directory(
    directory: str = ".", twiss: bool = True, beams: bool = False, **kwargs
) -> frameworkDirectory:
    """Load a directory from a SimFrame tracking run and return a frameworkDirectory object"""
    fw = frameworkDirectory(
        directory=directory, twiss=twiss, beams=beams, **kwargs
    )
    return fw
