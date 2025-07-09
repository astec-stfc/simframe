import os
import sys
import yaml
import inspect
from typing import Any
from pprint import pprint
import numpy as np
from .Modules.merge_two_dicts import merge_two_dicts
from .Modules import Beams as rbf
from .Modules import Twiss as rtf
from .Modules import constants
from .Codes import Executables as exes
from .Codes.Generators.Generators import (
    ASTRAGenerator,
    GPTGenerator,
    generator_keywords,
)
from .Framework_objects import runSetup
from . import Framework_lattices as frameworkLattices
from . import Framework_elements as frameworkElements
from .Framework_Settings import FrameworkSettings
from .FrameworkHelperFunctions import (
    _rotation_matrix,
    clean_directory,
    convert_numpy_types,
)

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


class Framework(Munch):

    def __init__(
        self,
        directory: str = "test",
        master_lattice: str | None = None,
        simcodes: str | None = None,
        overwrite: bool | None = None,
        runname: str = "CLARA_240",
        clean: bool = False,
        verbose: bool = True,
        sddsindex: int = 0,
        delete_output_files: bool = False,
    ):
        super(Framework, self).__init__()
        gptlicense = os.environ["GPTLICENSE"] if "GPTLICENSE" in os.environ else ""
        astra_use_wsl = os.environ["WSL_ASTRA"] if "WSL_ASTRA" in os.environ else 1
        self.global_parameters = {
            "beam": rbf.beam(sddsindex=sddsindex),
            "GPTLICENSE": gptlicense,
            "delete_tracking_files": delete_output_files,
            "astra_use_wsl": astra_use_wsl,
        }
        self.verbose = verbose
        self.subdir = directory
        self.clean = clean
        self.elementObjects = dict()
        self.latticeObjects = dict()
        self.commandObjects = dict()
        self.groupObjects = dict()
        self.progress = 0
        self.tracking = False
        self.basedirectory = os.getcwd()
        self.filedirectory = os.path.dirname(os.path.abspath(__file__))
        self.overwrite = overwrite
        self.runname = runname
        if self.subdir is not None:
            self.setSubDirectory(self.subdir)
        self.setMasterLatticeLocation(master_lattice)
        self.setSimCodesLocation(simcodes)

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

    def change_subdirectory(self, *args, **kwargs):
        self.setSubDirectory(*args, **kwargs)

    def setSubDirectory(self, dir: str) -> None:
        """Set subdirectory for tracking"""
        self.subdirectory = os.path.abspath(dir)
        self.global_parameters["master_subdir"] = self.subdirectory
        if not os.path.exists(self.subdirectory):
            os.makedirs(self.subdirectory, exist_ok=True)
        else:
            if self.clean is True:
                clean_directory(self.subdirectory)
        if self.overwrite is None:
            self.overwrite = True

    def setMasterLatticeLocation(self, master_lattice: str | None = None) -> None:
        """Set the location of the MasterLattice package"""
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
        """Set the location of the SimCodes package"""
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

    def load_Elements_File(self, input: str | list | tuple) -> None:
        """Load a file with element definitions"""
        if isinstance(input, (list, tuple)):
            filename = input
        else:
            filename = [input]
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
        self, filename: str | None = None, settings: FrameworkSettings | None = None
    ) -> None:
        """Load Lattice Settings from file or dictionary"""
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
            self.original_elementObjects[e] = unmunchify(self.elementObjects[e])
        self.original_elementObjects["generator"] = unmunchify(self["generator"])

    def save_settings(
        self,
        filename: str | None = None,
        directory: str = ".",
        elements: dict | None = None,
    ) -> None:
        """Save Lattice Settings to a file"""
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
        """Create an instance of a <code>Lattice class"""
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
        self, elementtype: str | None = None, elements: list | None = None
    ) -> dict:
        """Detect lattice changes from the original loaded lattice and return a dictionary of changes"""
        disallowed = [
            "allowedkeywords",
            "keyword_conversion_rules_elegant",
            "keyword_conversion_rules_ocelot",
            "objectdefaults",
            "global_parameters",
        ]
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
                    new = unmunchify(self.elementObjects[e])
                elif e in self.groupObjects:
                    new = self.groupObjects[e]
                if new is not None:
                    if e not in changedict:
                        changedict[e] = {}
                    changedict[e][k] = convert_numpy_types(new[k])
        else:
            for e in changeelements:
                if e in self.elementObjects:
                    unmunched_element = unmunchify(self.elementObjects[e])
                elif e == "generator":
                    unmunched_element = unmunchify(self["generator"])
                if not self.original_elementObjects[e] == unmunched_element:
                    orig = self.original_elementObjects[e]
                    new = unmunched_element
                    try:
                        changedict[e] = {
                            k: convert_numpy_types(new[k])
                            for k in new
                            if k in orig
                            and not new[k] == orig[k]
                            and k not in disallowed
                        }
                        changedict[e].update(
                            {
                                k: convert_numpy_types(new[k])
                                for k in new
                                if k not in orig and k not in disallowed
                            }
                        )
                        if changedict[e] == {}:
                            del changedict[e]
                    except Exception:
                        print("##### ERROR IN CHANGE ELEMS: ")  # , e, new)
                        pass
        return changedict

    def save_changes_file(
        self,
        filename: str | None = None,
        type: str | None = None,
        elements: dict | None = None,
        dictionary: bool = False,
    ) -> dict | None:
        """Save a file, or returns a dictionary, of detected changes in the lattice from the loaded version"""
        if filename is None:
            pre, ext = os.path.splitext(os.path.basename(self.settingsFilename))
            filename = pre + "_changes.yaml"
        changedict = self.detect_changes(elementtype=type, elements=elements)
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
        """Save to a file, or returns a dictionary, of a lattice"""
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
        disallowed = [
            "allowedkeywords",
            "keyword_conversion_rules_elegant",
            "keyword_conversion_rules_ocelot",
            "objectdefaults",
            "global_parameters",
            "objectname",
            "subelement",
            "beam",
        ]
        for e in elements:
            new = unmunchify(self.elementObjects[e])
            try:
                if (
                    "subelement" in new and not new["subelement"]
                ) or "subelement" not in new:
                    latticedict[e] = {
                        k.replace("object", ""): convert_numpy_types(new[k])
                        for k in new
                        if k not in disallowed
                    }
                    if "sub_elements" in new:
                        for subelem in new["sub_elements"]:
                            newsub = self.elementObjects[subelem]
                            latticedict[e]["sub_elements"][subelem] = {
                                k.replace("object", ""): convert_numpy_types(newsub[k])
                                for k in newsub
                                if k not in disallowed
                            }
            except Exception:
                print("##### ERROR IN CHANGE ELEMS: ", e, new)
                pass
        if dictionary:
            return dic
        else:
            with open(directory + "/" + filename, "w") as yaml_file:
                yaml.default_flow_style = True
                yaml.dump(dic, yaml_file)

    def load_changes_file(
        self, filename: str | None = None, apply: bool = True, verbose: bool = False
    ) -> dict:
        """Loads a saved changes file and applies the settings to the current lattice.
        Returns a list of changes.
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
        """Applies a dictionary of changes to the current lattice"""
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
        """Checks that there are no positioning errors in the lattice and returns True/False"""
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
                print('check_lattice error:', elem.objectname, cend, end, cend - end)
        return noerror

    def check_lattice_drifts(self, decimals: int = 4) -> bool:
        """Checks that there are no positioning errors in the lattice and returns True/False"""
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
                print('check_lattice_drifts error:', elem.objectname, cend, end, cend - end)
        return noerror

    def change_Lattice_Code(
        self, latticename: str, code: str, exclude: str | list | tuple | None = None
    ) -> None:
        """Changes the tracking code for a given lattice"""
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
        self, elementname: str, element: dict, subelement: bool = False, parent: str = None
    ) -> None:
        """Reads an element definition and creates the element and any sub-elements"""
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
        self, name: str | None = None, type: str | None = None, **kwargs
    ) -> dict:
        """Instantiates and adds the element definition to the list of all elements"""
        if name is None:
            if "name" not in kwargs:
                raise NameError("Element does not have a name")
            else:
                name = kwargs["name"]
        try:
            element = getattr(frameworkElements, type)(
                name, type, global_parameters=self.global_parameters, **kwargs
            )
            element.update_field_definition()
        except Exception as e:
            print('add_Element error:', e)
            print('add_Element error:', type, name, kwargs)
        self.elementObjects[name] = element
        return element
        # except Exception as e:
        #     raise NameError('Element \'%s\' does not exist' % type)

    def replace_Element(
        self, name: str | None = None, type: str | None = None, **kwargs
    ) -> dict:
        """Replaces and element type with a new type and updates the definitions"""
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
        element = getattr(frameworkElements, type)(name, type, **new_properties)
        # print element
        self.elementObjects[name] = element
        return element
        # except Exception as e:
        #     raise NameError('Element \'%s\' does not exist' % type)

    def getElement(self, element: str, param: str | None = None) -> dict | Any:
        """Returns the element object or a parameter of that element"""
        if self.__getitem__(element) is not None:
            if param is not None:
                param = param.lower()
                return getattr(self.__getitem__(element), param)
            else:
                return self.__getitem__(element)
        else:
            print(("WARNING: Element ", element, " does not exist"))
            return {}

    def getElementType(self, type: str, param: str | None = None) -> dict | Any:
        """Gets all elements of the specified type, or the parameter of each of those elements"""
        if isinstance(type, (list, tuple)):
            return [self.getElementType(t, param=param) for t in type]
        if isinstance(param, (list, tuple)):
            return zip(*[self.getElementType(type, param=p) for p in param])
            # return [item for sublist in all_elements for item in sublist]
        return [
            (
                {"name": element, **self.elementObjects[element]}
                if param is None
                else self.elementObjects[element][param]
            )
            for element in list(self.elementObjects.keys())
            if self.elementObjects[element].objecttype.lower() == type.lower()
        ]

    def setElementType(self, type: str, setting: str, values: Any) -> None:
        """Modifies the specified parameter of each element of a given type"""
        elems = self.getElementType(type)
        if len(elems) == len(values):
            for e, v in zip(elems, values):
                e[setting] = v
        else:
            raise ValueError

    def modifyElement(
        self, elementName: str, parameter: str | list | dict, value: Any = None
    ) -> None:
        """Modifies an element parameter"""
        if isinstance(parameter, dict) and value is None:
            for p, v in parameter.items():
                self.modifyElement(elementName, p, v)
        elif isinstance(parameter, (list, set)) and value is None:
            for p, v in parameter:
                self.modifyElement(elementName, p, v)
        elif elementName in self.groupObjects:
            self.groupObjects[elementName].change_Parameter(parameter, value)
        elif elementName in self.elementObjects:
            setattr(self.elementObjects[elementName], parameter, value)

    def modifyElements(
        self, elementNames: str | list, parameter: str | list | dict, value: Any = None
    ) -> None:
        """Modifies an element parameter for a list of elements"""
        if isinstance(elementNames, str) and elementNames.lower() == "all":
            elementNames = self.elementObjects.keys()
        for elem in elementNames:
            self.modifyElement(elem, parameter, value)

    def modifyElementType(self, elementType: str, parameter: str, value: Any) -> None:
        """Modifies an element for a list of elements of a given type"""
        elems = self.getElementType(elementType)
        for elementName in [e["name"] for e in elems]:
            self.modifyElement(elementName, parameter, value)

    def modifyLattice(
        self, latticeName: str, parameter: str | list | dict, value: Any = None
    ) -> None:
        """Modify a lattice definition"""
        if isinstance(parameter, dict) and value is None:
            for p, v in parameter.items():
                self.modifyLattice(latticeName, p, v)
        elif isinstance(parameter, (list, set)) and value is None:
            for p, v in parameter:
                self.modifyLattice(latticeName, p, v)
        elif latticeName in self.latticeObjects:
            setattr(self.latticeObjects[latticeName], parameter, value)

    def modifyLattices(
        self, latticeNames: str | list, parameter: str | list | dict, value: Any = None
    ) -> None:
        """Modify a lattice definition for a list of lattices"""
        if isinstance(latticeNames, str) and latticeNames.lower() == "all":
            latticeNames = self.latticeObjects.keys()
        for latt in latticeNames:
            self.modifyLattice(latt, parameter, value)

    def add_Generator(self, default: str | None = None, **kwargs) -> None:
        """Add a file generator based on a keyword dictionary"""
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

    def change_generator(self, generator: str) -> None:
        """Changes the generator from one type to another"""
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
            for k, v in list(parameters.items()):
                output[k] = {}
                if isinstance(v, (list, tuple)):
                    for p in v:
                        output[k][p] = getattr(self[k], p)
                else:
                    output[k][v] = getattr(self[k], v)
        elif isinstance(parameters, (list, tuple)):
            for k, v in parameters:
                output[k] = {}
                if isinstance(v, (list, tuple)):
                    for p in v:
                        output[k][p] = getattr(self[k], p)
                else:
                    output[k][v] = getattr(self[k], v)
        with open(file, "w") as yaml_file:
            yaml.default_flow_style = True
            yaml.dump(output, yaml_file)

    def set_lattice_prefix(self, lattice: str, prefix: str) -> None:
        """Sets the 'prefix' parameter for a lattice, which determines where it looks for its starting beam distribution"""
        if lattice in self.latticeObjects:
            self.latticeObjects[lattice].prefix = prefix

    def set_lattice_sample_interval(self, lattice: str, interval: int) -> None:
        """Sets the 'sample_interval' parameter for a lattice, which determines the sampling of the distribution"""
        if lattice in self.latticeObjects:
            self.latticeObjects[lattice].sample_interval = interval

    def __getitem__(self, key: str) -> Any:
        if key in super(Framework, self).__getitem__("elementObjects"):
            return self.elementObjects.get(key)
        elif key in super(Framework, self).__getitem__("latticeObjects"):
            return self.latticeObjects.get(key)
        elif key in super(Framework, self).__getitem__("groupObjects"):
            return self.groupObjects.get(key)
        else:
            try:
                return super(Framework, self).__getitem__(key)
            except Exception:
                return None

    @property
    def elements(self) -> list:
        """Returns a list of all element objects"""
        return list(self.elementObjects.keys())

    @property
    def groups(self) -> list:
        """Returns a list of all group objects"""
        return list(self.groupObjects.keys())

    @property
    def lines(self) -> list:
        """Returns a list of all lattice objects"""
        return list(self.latticeObjects.keys())

    @property
    def lattices(self) -> list:
        return self.lines

    @property
    def commands(self) -> list:
        """Returns a list of all command objects"""
        return list(self.commandObjects.keys())

    def getSValues(self) -> list:
        """returns a list of S values for the current machine"""
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
        """Returns a list of (name, element, s) tuples for the current machine"""
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
        """Returns a list of (name, element, Z) tuples for the current machine"""
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
        frameworkDirectory: bool = False,
        check_lattice: bool = True,
    ) -> None:
        """Tracks the current machine, or a subset based on the 'files' list"""
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
        if frameworkDirectory:
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
        """Post-processes the tracking files and converts them to HDF5"""
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
        """Saves HDF5 summary files for the Twiss and/or Beam files"""
        t = rtf.load_directory(self.subdirectory)
        try:
            t.save_HDF5_twiss_file(self.subdirectory + "/" + "Twiss_Summary.hdf5")
        except Exception:
            pass
        try:
            rbf.save_HDF5_summary_file(
                self.subdirectory, self.subdirectory + "/" + "Beam_Summary.hdf5"
            )
        except Exception:
            pass

    def pushRunSettings(self) -> None:
        """Updates the 'Run Settings' in each of the lattices"""
        for ln, latticeObject in self.latticeObjects.items():
            if isinstance(latticeObject, tuple(latticeClasses)):
                latticeObject.updateRunSettings(self.runSetup)

    def setNRuns(self, nruns: int) -> None:
        """sets the number of simulation runs to a new value for all lattice objects"""
        self.runSetup.setNRuns(nruns)
        self.pushRunSettings()

    def setSeedValue(self, seed: int) -> None:
        """sets the random number seed to a new value for all lattice objects"""
        self.runSetup.setSeedValue(seed)
        self.pushRunSettings()

    def loadElementErrors(self, file: str) -> None:
        self.runSetup.loadElementErrors(file)
        self.pushRunSettings()

    def setElementScan(
        self, name: str, item: str, scanrange: list, multiplicative: bool = False
    ) -> None:
        """define a parameter scan for a single parameter of a given machine element"""
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
        """Moves all elements by the set amount in (x, y, z) space"""
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


class frameworkDirectory(Munch):
    """Class to load a tracking run from a directory and read the Beam and Twiss files and make them available"""

    def __init__(
        self,
        directory: str | None = None,
        twiss: bool = True,
        beams: bool = False,
        verbose: bool = False,
        settings: str = "settings.def",
        changes: str = "changes.yaml",
        rest_mass: float | None = None,
        framework: Framework | None = None,
        **kwargs,
    ) -> None:
        super(frameworkDirectory, self).__init__()
        if framework is None:
            directory = "." if directory is None else os.path.abspath(directory)
            self.framework = Framework(
                directory, clean=False, verbose=verbose, **kwargs
            )
            self.framework.loadSettings(directory + "/" + settings)
        else:
            self.framework = framework
            if directory is None:
                directory = os.path.abspath(self.framework.subdirectory)

        if os.path.exists(directory + "/" + changes):
            self.framework.load_changes_file(directory + "/" + changes)
        if beams:
            self.beams = rbf.load_HDF5_summary_file(
                os.path.join(directory, "Beam_Summary.hdf5")
            )
            if len(self.beams) < 1:
                print("No Summary File! Globbing...")
                self.beams = rbf.load_directory(directory)
            if rest_mass is None:
                if len(self.beams.param("particle_rest_energy")) > 0:
                    rest_mass = self.beams.param("particle_rest_energy")[0][0]
                else:
                    rest_mass = constants.m_e
            self.twiss = rtf.twiss(rest_mass=rest_mass)
        else:
            self.beams = None
            self.twiss = rtf.twiss()
        if twiss:
            self.twiss.load_directory(directory)

    if use_matplotlib:

        def plot(self, *args, **kwargs):
            """Return a plot object"""
            return groupplot.plot(self, *args, **kwargs)

        def general_plot(self, *args, **kwargs):
            """Return a general_plot object"""
            return groupplot.general_plot(self, *args, **kwargs)

    def __repr__(self):
        return repr(
            {"framework": self.framework, "twiss": self.twiss, "beams": self.beams}
        )

    def save_summary_files(self, twiss: bool = True, beams: bool = True):
        """Save summary files in framework directory"""
        self.framework.save_summary_files(twiss=twiss, beams=beams)

    def getScreen(self, screen: str) -> dict:
        """Get a beam object for the given screen"""
        if self.beams:
            return self.beams.getScreen(screen)

    def getScreenNames(self) -> list:
        """Get all screen names in the beam object"""
        if self.beams:
            return self.beams.getScreens()
        return []

    def element(self, element: str, field: str | None = None) -> dict:
        """Get an element definition from the framework object"""
        elem = self.framework.getElement(element)
        if field:
            return elem[field]
        else:
            disallowed = [
                "allowedkeywords",
                "keyword_conversion_rules_elegant",
                "keyword_conversion_rules_ocelot",
                "objectdefaults",
                "global_parameters",
                "objectname",
                "subelement",
                "beam",
            ]
            return pprint(
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
        directory=directory, twiss=twiss, beams=beams, verbose=True, **kwargs
    )
    return fw
