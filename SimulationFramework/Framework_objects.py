import os
import subprocess
import yaml
from munch import Munch
from .Modules.merge_two_dicts import merge_two_dicts
from .Modules.MathParser import MathParser
from .FrameworkHelperFunctions import (
    chunks,
    expand_substitution,
    checkValue,
    chop,
    dot
)
from .FrameworkHelperFunctions import _rotation_matrix
from .Modules.Fields import field
from .Codes import Executables as exes
from .Codes.Ocelot import ocelot_conversion
from ocelot.cpbd.elements import Marker, Aperture
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
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

type_conversion_rules_Ocelot = ocelot_conversion.ocelot_conversion_rules

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
    """class defining settings for simulations that include multiple runs"""

    def __init__(self):
        # define the number of runs and the random number seed
        self.nruns = 1
        self.seed = 0

        # init errorElement and elementScan settings as None
        self.elementErrors = None
        self.elementScan = None

    def setNRuns(self, nruns):
        """sets the number of simulation runs to a new value"""
        # enforce integer argument type
        if isinstance(nruns, (int, float)):
            self.nruns = int(nruns)
        else:
            raise TypeError(
                "Argument nruns passed to runSetup instance must be an integer"
            )

    def setSeedValue(self, seed):
        """sets the random number seed to a new value for all lattice objects"""
        # enforce integer argument type
        if isinstance(seed, (int, float)):
            self.seed = int(seed)
        else:
            raise TypeError("Argument seed passed to runSetup must be an integer")

    def loadElementErrors(self, file):
        # load error definitions from markup file
        if isinstance(file, str) and (".yaml" in file):
            with open(file, "r") as infile:
                error_setup = dict(yaml.safe_load(infile))
        # define errors from dictionary
        elif isinstance(file, dict):
            error_setup = file

        # assign the element error definitions
        self.elementErrors = error_setup["elements"]
        self.elementScan = None

        # set the number of runs and random number seed, if available
        if "nruns" in error_setup:
            self.setNRuns(error_setup["nruns"])
        if "seed" in error_setup:
            self.setSeedValue(error_setup["seed"])

    def setElementScan(self, name, item, scanrange, multiplicative=False):
        """define a parameter scan for a single parameter of a given machine element"""
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


class frameworkLattice(BaseModel):
    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

    name: str
    file_block: Dict
    elementObjects: Dict
    groupObjects: Dict
    runSettings: runSetup
    settings: Dict
    executables: exes.Executables
    global_parameters: Dict
    allow_negative_drifts: bool = False
    _csr_enable: bool = True
    csrDrifts: bool = True
    lscDrifts: bool = True
    lsc_bins: int = 20
    lsc_high_frequency_cutoff_start: float = -1
    lsc_high_frequency_cutoff_end: float = -1
    lsc_low_frequency_cutoff_start: float = -1
    lsc_low_frequency_cutoff_end: float = -1
    _sample_interval: int = 1
    globalSettings: Dict = {"charge": None}
    groupSettings: Dict = {}
    allElements: List = []

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
        self.globalSettings = (
            settings["global"] if settings["global"] is not None else {"charge": None}
        )
        self.groupSettings = (
            file_block["groups"]
            if "groups" in file_block and file_block["groups"] is not None
            else {}
        )
        self.update_groups()
        self._sample_interval = (
            self.file_block["input"]["sample_interval"]
            if "input" in self.file_block
            and "sample_interval" in self.file_block["input"]
            else 1
        )
        self.objectname = self.name

        # define settings for simulations with multiple runs
        self.updateRunSettings(runSettings)

    def insert_element(self, index, element):
        for i, _ in enumerate(range(len(self.elements))):
            k, v = self.elements.popitem(False)
            self.elements[element.objectname if i == index else k] = element

    @property
    def csr_enable(self):
        return self._csr_enable

    @csr_enable.setter
    def csr_enable(self, csr):
        self.csrDrifts = csr
        self._csr_enable = csr

    @property
    def sample_interval(self):
        return self._sample_interval

    @sample_interval.setter
    def sample_interval(self, interval):
        # print('Setting new sample_interval = ', interval)
        self._sample_interval = interval

    @property
    def prefix(self):
        if "input" not in self.file_block:
            self.file_block["input"] = {}
        if "prefix" not in self.file_block["input"]:
            self.file_block["input"]["prefix"] = ""
        return self.file_block["input"]["prefix"]

    @prefix.setter
    def prefix(self, prefix):
        if "input" not in self.file_block:
            self.file_block["input"] = {}
        self.file_block["input"]["prefix"] = prefix

    def update_groups(self):
        for g in list(self.groupSettings.keys()):
            if g in self.groupObjects:
                setattr(self, g, self.groupObjects[g])
                if self.groupSettings[g] is not None:
                    self.groupObjects[g].update(**self.groupSettings[g])

    def getElement(self, element, param=None):
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
            print(("WARNING: Element ", element, " does not exist"))
            return {}

    def getElementType(self, type, param=None) -> list | tuple | zip:
        if isinstance(type, (list, tuple)):
            return [self.getElementType(t, param=param) for t in type]
        if isinstance(param, (list, tuple)):
            return zip(*[self.getElementType(type, param=p) for p in param])
        return [
            self.elements[element] if param is None else self.elements[element][param]
            for element in list(self.elements.keys())
            if self.elements[element].objecttype.lower() == type.lower()
        ]

    def setElementType(self, type, setting, values):
        elems = self.getElementType(type)
        if len(elems) == len(values):
            for e, v in zip(elems, values):
                e[setting] = v
        else:
            raise ValueError

    @property
    def quadrupoles(self):
        return self.getElementType("quadrupole")

    @property
    def cavities(self):
        return self.getElementType("cavity")

    @property
    def solenoids(self):
        return self.getElementType("solenoid")

    @property
    def dipoles(self):
        return self.getElementType("dipole")

    @property
    def kickers(self):
        return self.getElementType("kicker")

    @property
    def dipoles_and_kickers(self):
        return sorted(
            self.getElementType("dipole") + self.getElementType("kicker"),
            key=lambda x: x.position_end[2],
        )

    @property
    def wakefields(self):
        return self.getElementType("wakefield")

    @property
    def wakefields_and_cavity_wakefields(self):
        cavities = [cav for cav in self.getElementType("cavity") if
                    (hasattr(cav, 'longitudinal_wakefield') and cav['longitudinal_wakefield'] is not None and cav['longitudinal_wakefield'] != '')
                    or
                    (hasattr(cav, 'transverse_wakefield') and cav['transverse_wakefield'] is not None and cav['transverse_wakefield'] != '')
                    or
                    (hasattr(cav, 'wakefield_definition') and cav['wakefield_definition'] is not None and cav['wakefield_definition'] != '')]
        wakes = self.getElementType("wakefield")
        return cavities + wakes

    @property
    def screens(self):
        return self.getElementType("screen")

    @property
    def screens_and_bpms(self):
        return sorted(
            self.getElementType("screen")
            + self.getElementType("beam_position_monitor"),
            key=lambda x: x.position_start[2],
        )

    @property
    def screens_and_markers_and_bpms(self) -> list:
        """Return all Screens and BPMs"""
        return sorted(
            self.getElementType("screen")
            + self.getElementType("marker")
            + self.getElementType("beam_position_monitor"),
            key=lambda x: x.position_start[2],
        )

    @property
    def apertures(self):
        return sorted(
            self.getElementType("aperture") + self.getElementType("collimator"),
            key=lambda x: x.position_start[2],
        )

    @property
    def lines(self):
        return list(self.lineObjects.keys())

    @property
    def start(self):
        if "start_element" in self.file_block["output"]:
            return self.file_block["output"]["start_element"]
        elif "zstart" in self.file_block["output"]:
            for e in list(self.elementObjects.keys()):
                if (
                    self.elementObjects[e].position_start[2]
                    == self.file_block["output"]["zstart"]
                ):
                    return e
        else:
            return self.elementObjects[0]

    @property
    def startObject(self):
        return self.elementObjects[self.start]

    @property
    def end(self):
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
    def endObject(self):
        return self.elementObjects[self.end]

    @property
    def elements(self):
        index_start = self.allElements.index(self.start)
        index_end = self.allElements.index(self.end)
        f = dict(
            [
                [e, self.elementObjects[e]]
                for e in self.allElements[index_start: index_end + 1]
            ]
        )
        return f

    def write(self):
        pass

    def run(self):
        """Run the code with input 'filename'"""
        command = self.executables[self.code] + [self.name]
        with open(
            os.path.relpath(
                self.global_parameters["master_subdir"]
                + "/"
                + self.name
                + ".log",
                ".",
            ),
            "w",
        ) as f:
            subprocess.call(
                command, stdout=f, cwd=self.global_parameters["master_subdir"]
            )

    def getInitialTwiss(self):
        """Get the initial Twiss parameters from the file block"""
        if "input" in self.file_block and "twiss" in self.file_block["input"] and self.file_block["input"]["twiss"]:
            alpha_x = self.file_block["input"]["twiss"]["alpha_x"] if "alpha_x" in self.file_block["input"]["twiss"] else False
            alpha_y = self.file_block["input"]["twiss"]["alpha_y"] if "alpha_y" in self.file_block["input"]["twiss"] else False
            beta_x = self.file_block["input"]["twiss"]["beta_x"] if "beta_x" in self.file_block["input"]["twiss"] else False
            beta_y = self.file_block["input"]["twiss"]["beta_y"] if "beta_y" in self.file_block["input"]["twiss"] else False
            nemit_x = self.file_block["input"]["twiss"]["nemit_x"] if "nemit_x" in self.file_block["input"]["twiss"] else False
            nemit_y = self.file_block["input"]["twiss"]["nemit_y"] if "nemit_y" in self.file_block["input"]["twiss"] else False
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
                    }
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
                }
            }

    def preProcess(self):
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

    def createDrifts(self):
        """Insert drifts into a sequence of 'elements'"""
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

        for e, d in driftdata:
            if (
                e[1]["objecttype"] == "screen"
                or e[1]["objecttype"] == "beam_position_monitor"
            ) and round(e[1]["length"] / 2, 6) > 0:
                name = e[0] + "-drift-01"
                newdrift = drifttype(
                    name,
                    global_parameters=self.global_parameters,
                    **{
                        "length": round(e[1]["length"] / 2, 6),
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
                )
                newelements[name] = newdrift
                newelements[e[0]] = e[1]
                name = e[0] + "-drift-02"
                newdrift = drifttype(
                    name,
                    global_parameters=self.global_parameters,
                    **{
                        "length": round(e[1]["length"] / 2, 6),
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
                )
                newelements[name] = newdrift
            else:
                newelements[e[0]] = e[1]
            if e[1]["objecttype"] == "dipole":
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
                    print("Element with error = ", e[0])
                    print(d)
                    raise exc
                if self.allow_negative_drifts or (round(length, 6) > 0 and vector < 1e-6):
                    elementno += 1
                    name = "drift" + str(elementno)
                    middle = [(a + b) / 2.0 for a, b in zip(d[0], d[1])]
                    newdrift = drifttype(
                        name,
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
                        }
                    )
                    newelements[name] = newdrift
                elif length < 0 or vector > 1e-6:
                    raise Exception("Lattice has negative drifts!", self.allow_negative_drifts, e[0], e[1], length)
        return newelements

    def getSValues(self, drifts: bool = True, as_dict: bool = False, at_entrance=False):
        elems = self.createDrifts()
        s = [0]
        for e in list(elems.values()):
            s.append(s[-1] + e.length)
        s = s[:-1] if at_entrance else s[1:]
        if as_dict:
            return dict(zip([e.objectname for e in elems.values()], s))
        return list(s)

    def getZValues(self, drifts: bool = True, as_dict: bool = False):
        if drifts:
            elems = self.createDrifts()
        else:
            elems = self.elements
        if as_dict:
            return {e.objectname: [e.start[2], e.end[2]] for e in elems.values()}
        return [[e.start[2], e.end[2]] for e in elems.values()]

    def getNames(self, drifts: bool = True):
        if drifts:
            elems = self.createDrifts()
        else:
            elems = self.elements
        return [e.objectname for e in list(elems.values())]

    def getElems(self, drifts: bool = True, as_dict: bool = False):
        if drifts:
            elems = self.createDrifts()
        else:
            elems = self.elements
        if as_dict:
            return {e.objectname: e for e in list(elems.values())}
        return [e for e in list(elems.values())]

    def getSNames(self):
        s = self.getSValues()
        names = self.getNames()
        return list(zip(names, s))

    def getSNamesElems(self):
        s = self.getSValues()
        names = self.getNames()
        elems = self.getElems()
        return names, elems, s

    def getZNamesElems(self):
        z = self.getZValues()
        names = self.getNames()
        elems = self.getElems()
        return names, elems, z

    def findS(self, elem) -> list:
        if elem in self.allElements:
            sNames = self.getSNames()
            return [a for a in sNames if a[0] == elem]
        return []

    def updateRunSettings(self, runSettings):
        if isinstance(runSettings, runSetup):
            self.runSettings = runSettings
        else:
            raise TypeError(
                "runSettings argument passed to frameworkLattice.updateRunSettings is not a runSetup instance"
            )


class frameworkObject(BaseModel):
    class Config:
        extra = "allow"
        arbitrary_types_allowed = True
    objectname: str
    objecttype: str
    objectdefaults: Dict = {}
    allowedkeywords: List | Dict = {}

    def __init__(
            self,
            objectname,
            objecttype,
            *args,
            **kwargs
    ):
        frameworkObject.objectname = objectname
        frameworkObject.objecttyp = objecttype
        super(frameworkObject, self).__init__(
            objectname=objectname,
            objecttype=objecttype,
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
            print(("Unknown type = ", objecttype))
            raise NameError
        self.allowedkeywords = [x.lower() for x in self.allowedkeywords]
        for key, value in list(kwargs.items()):
            self.add_property(key, value)

    def change_Parameter(self, key, value):
        setattr(self, key, value)

    def add_property(self, key, value):
        key = key.lower()
        if key in self.allowedkeywords:
            try:
                setattr(self, key, value)
            except Exception as e:
                print('add_property error:', (self.objecttype, "[", key, "]: ", e))

    def add_properties(self, **keyvalues):
        for key, value in keyvalues.items():
            key = key.lower()
            if key in self.allowedkeywords:
                try:
                    setattr(self, key, value)
                except Exception as e:
                    print('add_properties error:', (self.objecttype, "[", key, "]: ", e))

    def add_default(self, key, value):
        self.objectdefaults[key] = value

    @property
    def parameters(self):
        return list(self.keys())

    @property
    def objectproperties(self):
        return self

    def __getitem__(self, key):
        lkey = key.lower()
        defaults = super(frameworkObject, self).__getitem__("objectdefaults")
        if lkey in defaults:
            try:
                return super(frameworkObject, self).__getitem__(lkey)
            except Exception:
                return defaults[lkey]
        else:
            try:
                return super(frameworkObject, self).__getitem__(lkey)
            except Exception:
                try:
                    return super(frameworkObject, self).__getattribute__(key)
                except Exception:
                    return None

    def __repr__(self):
        string = ""
        for k, v in list(self.items()):
            string += "{} ({})".format(k, v) + "\n"
        return string


class frameworkCommand(frameworkObject):

    def __init__(self, objectname=None, objecttype=None, **kwargs):
        if objectname is None:
            objectname = objecttype
        super(frameworkCommand, self).__init__(objectname, objecttype, **kwargs)
        if objecttype not in commandkeywords:
            raise NameError("Command '%s' does not exist" % objecttype)

    def write_Elegant(self):
        string = "&" + self.objecttype + "\n"
        for key in commandkeywords[self.objecttype]:
            if (
                key.lower() in self.objectproperties
                and not key == "objectname"
                and not key == "objecttype"
                and not self.objectproperties[key.lower()] is None
            ):
                string += (
                    "\t" + key + " = " + str(self.objectproperties[key.lower()]) + "\n"
                )
        string += "&end\n"
        return string

    def write_MAD8(self):
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
    def __init__(self, name, elementObjects, type, elements, **kwargs):
        super(frameworkGroup, self).__init__()
        self.objectname = name
        self.type = type
        self.elements = elements
        self.allElementObjects = elementObjects.elementObjects
        self.allGroupObjects = elementObjects.groupObjects

    def update(self, **kwargs):
        pass

    def get_Parameter(self, p):
        try:
            isinstance(type(self).p, p)
            return getattr(self, p)
        except Exception:
            if self.elements[0] in self.allGroupObjects:
                return self.allGroupObjects[self.elements[0]][p]
            return self.allElementObjects[self.elements[0]][p]

    def change_Parameter(self, p, v):
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
        return [self.allElementObjects[e].objectname for e in self.elements]

    def __str__(self):
        return str([self.allElementObjects[e].objectname for e in self.elements])

    def __getitem__(self, key):
        return self.get_Parameter(key)

    def __setitem__(self, key, value):
        return self.change_Parameter(key, value)


class element_group(frameworkGroup):
    def __init__(self, name, elementObjects, type, elements, **kwargs):
        super().__init__(name, elementObjects, type, elements, **kwargs)

    def __str__(self):
        return str([self.allElementObjects[e] for e in self.elements])


class r56_group(frameworkGroup):
    def __init__(self, name, elementObjects, type, elements, ratios, keys, **kwargs):
        super().__init__(name, elementObjects, type, elements, **kwargs)
        self.ratios = ratios
        self.keys = keys
        self._r56 = None

    def __str__(self):
        return str({e: k for e, k in zip(self.elements, self.keys)})

    def get_Parameter(self, p):
        if str(p) == "r56":
            return self.r56
        else:
            super().get_Parameter(p)

    @property
    def r56(self):
        return self._r56

    @r56.setter
    def r56(self, r56):
        # print('Changing r56!', self._r56)
        self._r56 = r56
        data = {"r56": self._r56}
        parser = MathParser(data)
        values = [parser.parse(e) for e in self.ratios]
        # print('\t', list(zip(self.elements, self.keys, values)))
        for e, k, v in zip(self.elements, self.keys, values):
            self.updateElements(e, k, v)

    def updateElements(self, element, key, value):
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
    def __init__(self, name, elementObjects, type, elements, **kwargs):
        super(chicane, self).__init__(name, elementObjects, type, elements, **kwargs)
        self.ratios = (1, -1, -1, 1)

    def update(self, **kwargs):
        if "dipoleangle" in kwargs:
            self.set_angle(kwargs["dipoleangle"])
        if "width" in kwargs:
            self.change_Parameter("width", kwargs["width"])
        if "gap" in kwargs:
            self.change_Parameter("gap", kwargs["gap"])

    @property
    def angle(self):
        obj = [self.allElementObjects[e] for e in self.elements]
        return float(obj[0].angle)

    @angle.setter
    def angle(self, theta):
        "using setter! angle = ", theta
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

    def set_angle(self, a):
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
    def __init__(self, name, elementObjects, type, elements, **kwargs):
        super(s_chicane, self).__init__(name, elementObjects, type, elements, **kwargs)
        self.ratios = (-1, 2, -2, 1)


class frameworkCounter(dict):
    def __init__(self, sub={}):
        super(frameworkCounter, self).__init__()
        self.sub = sub

    def counter(self, type):
        type = self.sub[type] if type in self.sub else type
        if type not in self:
            return 1
        return self[type] + 1

    def value(self, type):
        type = self.sub[type] if type in self.sub else type
        if type not in self:
            return 1
        return self[type]

    def add(self, type, n=1):
        type = self.sub[type] if type in self.sub else type
        if type not in self:
            self[type] = n
        else:
            self[type] += n
        return self[type]

    def subtract(self, type):
        type = self.sub[type] if type in self.sub else type
        if type not in self:
            self[type] = 0
        else:
            self[type] = self[type] - 1 if self[type] > 0 else 0
        return self[type]


class frameworkElement(frameworkObject):

    def __init__(self, elementName=None, elementType=None, **kwargs):
        super().__init__(elementName, elementType, **kwargs)
        self.add_default("length", 0)
        self.add_property("position_errors", [0, 0, 0])
        self.add_property("rotation_errors", [0, 0, 0])
        self.add_default("global_rotation", [0, 0, 0])
        self.add_default("rotation", [0, 0, 0])
        self.add_default("starting_rotation", 0)
        self.keyword_conversion_rules_elegant = keyword_conversion_rules_elegant[
            "general"
        ]
        if elementType in keyword_conversion_rules_elegant:
            self.keyword_conversion_rules_elegant = merge_two_dicts(
                self.keyword_conversion_rules_elegant,
                keyword_conversion_rules_elegant[elementType],
            )
        if elementType in keyword_conversion_rules_elegant:
            self.keyword_conversion_rules_elegant = merge_two_dicts(
                self.keyword_conversion_rules_elegant,
                keyword_conversion_rules_elegant[elementType],
            )
        self.keyword_conversion_rules_ocelot = keyword_conversion_rules_ocelot[
            "general"
        ]
        if elementType in keyword_conversion_rules_ocelot:
            self.keyword_conversion_rules_ocelot = merge_two_dicts(
                self.keyword_conversion_rules_ocelot,
                keyword_conversion_rules_ocelot[elementType],
            )

    def __mul__(self, other):
        return [self.objectproperties for x in range(other)]

    def __rmul__(self, other):
        return [self.objectproperties for x in range(other)]

    def __neg__(self):
        return self

    def __repr__(self):
        disallowed = [
            "allowedkeywords",
            "keyword_conversion_rules_elegant",
            "objectdefaults",
            "global_parameters",
            "objectname",
            "subelement",
        ]
        return repr(
            {k.replace("object", ""): v for k, v in self.items() if k not in disallowed}
        )

    @property
    def propertiesDict(self):
        disallowed = [
            "allowedkeywords",
            "keyword_conversion_rules_elegant",
            "objectdefaults",
            "global_parameters",
            "subelement",
        ]
        return {
            k.replace("object", ""): v for k, v in self.items() if k not in disallowed
        }

    @property
    def x(self):
        return self.position_start[0]

    @x.setter
    def x(self, x):
        self.position_start[0] = x
        self.position_end[0] = x

    @property
    def y(self):
        return self.position_start[1]

    @y.setter
    def y(self, y):
        self.position_start[1] = y
        self.position_end[1] = y

    @property
    def z(self):
        return self.position_start[2]

    @z.setter
    def z(self, z):
        self.position_start[2] = z
        self.position_end[2] = z

    @property
    def dx(self):
        return self.position_errors[0]

    @dx.setter
    def dx(self, x):
        self.position_errors[0] = x

    @property
    def dy(self):
        return self.position_errors[1]

    @dy.setter
    def dy(self, y):
        self.position_errors[1] = y

    @property
    def dz(self):
        return self.position_errors[2]

    @dz.setter
    def dz(self, z):
        self.position_errors[2] = z

    @property
    def x_rot(self):
        return self.global_rotation[1]

    @property
    def y_rot(self):
        return self.global_rotation[2] + self.starting_rotation

    @property
    def z_rot(self):
        return self.global_rotation[0]

    @property
    def dx_rot(self):
        return self.rotation_errors[1]

    @dx_rot.setter
    def dx_rot(self, x):
        self.rotation_errors[1] = x

    @property
    def dy_rot(self):
        return self.rotation_errors[2]

    @dy_rot.setter
    def dy_rot(self, y):
        self.rotation_errors[2] = y

    @property
    def dz_rot(self):
        return self.rotation_errors[0]

    @dz_rot.setter
    def dz_rot(self, z):
        self.rotation_errors[0] = z

    @property
    def tilt(self):
        return self.dz_rot

    @property
    def PV(self):
        if hasattr(self, "PV_root"):
            return self.PV_root
        else:
            return self.objectName

    @property
    def get_field_amplitude(self):
        if hasattr(self, "field_scale") and isinstance(self.field_scale, (int, float)):
            return float(self.field_scale) * float(
                expand_substitution(self, self.field_amplitude)
            )
        else:
            return float(expand_substitution(self, self.field_amplitude))

    def get_field_reference_position(self):
        if hasattr(self, "field_reference_position") and self.field_reference_position is not None:
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
    def theta(self):
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
    def rotation_matrix(self):
        return _rotation_matrix(self.theta)

    def rotated_position(self, pos=[0, 0, 0], offset=None, theta=None):
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
    def start(self):
        return self.position_start

    @property
    def position_start(self):
        middle = np.array(self.centre)
        start = middle - self.rotated_position(
            np.array([0, 0, self.length / 2.0]),
            offset=self.starting_offset,
            theta=self.y_rot,
        )
        return start

    @property
    def position_middle(self):
        return self.centre

    @property
    def middle(self):
        return self.centre

    @property
    def end(self):
        return self.position_end

    @property
    def position_end(self):
        start = np.array(self.position_start)
        end = start + self.rotated_position(
            np.array([0, 0, self.length]), offset=self.starting_offset, theta=self.y_rot
        )
        return end

    def relative_position_from_centre(self, vec=[0, 0, 0]):
        middle = np.array(self.centre)
        return middle + self.rotated_position(
            np.array(vec), offset=self.starting_offset, theta=self.y_rot
        )

    def relative_position_from_start(self, vec=[0, 0, 0]):
        start = np.array(self.position_start)
        return start + self.rotated_position(
            np.array(vec), offset=self.starting_offset, theta=self.y_rot
        )

    def update_field_definition(self) -> None:
        """Updates the field definitions to allow for the relative sub-directory location"""
        if hasattr(self, "field_definition") and self.field_definition is not None and isinstance(self.field_definition, str):
            # print('update_field_definition', self.objectname, self.field_definition, self.field_type, self.frequency, self.Structure_Type)
            self.field_definition = field(
                filename=expand_substitution(self, self.field_definition),
                field_type=self.field_type,
                frequency=self.frequency,
                cavity_type=self.Structure_Type,
                n_cells=self.n_cells
            )
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
                n_cells=self.n_cells
            )

    def _write_ASTRA_dictionary(self, d, n=1):
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

    def write_ASTRA(self, n, **kwargs):
        return self._write_ASTRA(n, **kwargs)

    def generate_field_file_name(self, param, code):
        if hasattr(param, 'filename'):
            basename = os.path.basename(param.filename).replace('"', "").replace("'", "")
            # location = os.path.abspath(
            #     expand_substitution(self, param.filename)
            #     .replace("\\", "/")
            #     .replace('"', "")
            #     .replace("'", "")
            # )
            efield_basename = os.path.abspath(
                self.global_parameters["master_subdir"].replace("\\", "/")
                + "/"
                + basename.replace("\\", "/")
            )
            return os.path.basename(param.write_field_file(code=code, location=efield_basename))
        else:
            pass
            # print(f'param does not have a filename: {param}')
        return None

    def _write_Elegant(self):
        wholestring = ""
        etype = self._convertType_Elegant(self.objecttype)
        string = self.objectname + ": " + etype
        k1 = self.k1 if self.k1 is not None else 0
        k2 = self.k2 if self.k2 is not None else 0
        keydict = merge_two_dicts(
            {"k1": k1, "k2": k2},
            merge_two_dicts(self.objectproperties, self.objectdefaults),
        )
        for key, value in keydict.items():
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

    def write_Elegant(self):
        if not self.subelement:
            return self._write_Elegant()

    def _convertType_Elegant(self, etype):
        return (
            type_conversion_rules_Elegant[etype]
            if etype in type_conversion_rules_Elegant
            else etype
        )

    def _convertKeyword_Elegant(self, keyword):
        return (
            self.keyword_conversion_rules_elegant[keyword]
            if keyword in self.keyword_conversion_rules_elegant
            else keyword
        )

    def _write_Ocelot(self):
        obj = type_conversion_rules_Ocelot[self.objecttype](eid=self.objectname)
        k1 = self.k1 if self.k1 is not None else 0
        k2 = self.k2 if self.k2 is not None else 0
        keydict = merge_two_dicts(
            {"k1": k1, "k2": k2},
            merge_two_dicts(self.objectproperties, self.objectdefaults),
        )
        for key, value in keydict.items():
            if (key not in ["name", "type", "commandtype"]) and (
                not type(obj) in [Aperture, Marker]
            ):
                value = (
                    getattr(self, key)
                    if hasattr(self, key) and getattr(self, key) is not None
                    else value
                )
                setattr(obj, self._convertKeword_Ocelot(key), value)
        return obj

    def write_Ocelot(self):
        if not self.subelement:
            return self._write_Ocelot()

    def _convertType_Ocelot(self, etype):
        return (
            type_conversion_rules_Ocelot[etype]
            if etype in type_conversion_rules_Ocelot
            else etype
        )

    def _convertKeword_Ocelot(self, keyword):
        return (
            self.keyword_conversion_rules_ocelot[keyword]
            if keyword in self.keyword_conversion_rules_ocelot
            else keyword
        )

    def _write_CSRTrack(self, n=0, **kwargs):
        pass

    def write_CSRTrack(self, n=0, **kwargs):
        return self._write_CSRTrack(self, n, **kwargs)

    def write_GPT(self, Brho, ccs="wcs", *args, **kwargs):
        return self._write_GPT(Brho, ccs, *args, **kwargs)

    def _write_GPT(self, Brho, ccs="wcs", *args, **kwargs):
        return ""

    def gpt_coordinates(self, position, rotation):
        x, y, z = chop(position, 1e-6)
        psi, phi, theta = rotation
        output = ""
        for c in [-x, y, z]:
            output += str(c) + ", "
        output += "cos(" + str(theta) + "), 0, -sin(" + str(theta) + "), 0, 1 ,0"
        return output

    def gpt_ccs(self, ccs):
        return ccs

    def array_names_string(self):
        array_names = (
            self.default_array_names if self.array_names is None else self.array_names
        )
        return ", ".join(['"' + name + '"' for name in array_names])


class getGrids(object):

    def __init__(self):
        self.powersof8 = np.asarray([2 ** (j) for j in range(1, 20)])

    def getGridSizes(self, x):
        self.x = abs(x)
        self.cuberoot = int(round(self.x ** (1.0 / 3)))
        return max([4, self.find_nearest(self.powersof8, self.cuberoot)])

    def find_nearest(self, array, value):
        self.array = array
        self.value = value
        self.idx = (np.abs(self.array - self.value)).argmin()
        return self.array[self.idx]


class csrdrift(frameworkElement):

    def __init__(self, name=None, type="csrdrift", **kwargs):
        super().__init__(name, type, **kwargs)
        self.add_default("lsc_interpolate", 1)

    def _write_Elegant(self):
        wholestring = ""
        etype = self._convertType_Elegant(self.objecttype)
        string = self.objectname + ": " + etype
        for key, value in list(
            merge_two_dicts(self.objectproperties, self.objectdefaults).items()
        ):
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

    def __init__(self, name=None, type="lscdrift", **kwargs):
        super().__init__(name, type, **kwargs)


class edrift(csrdrift):

    def __init__(self, name=None, type="edrift", **kwargs):
        super().__init__(name, type, **kwargs)