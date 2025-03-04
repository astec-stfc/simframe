import os
import re
import yaml
from copy import copy
from itertools import groupby
from pysdds import read as sddsread
from counter import Counter
from SimulationFramework.FrameworkHelperFunctions import chop  # type: ignore

with open(
    os.path.dirname(os.path.abspath(__file__))
    + "/../../Codes/type_conversion_rules.yaml",
    "r",
) as infile:
    type_conversion_rules = yaml.safe_load(infile)
    type_conversion_rules_Elegant = type_conversion_rules["elegant"]
    type_conversion_rules_Names = type_conversion_rules["name"]
elementtypes = {v.lower(): k.lower() for k, v in type_conversion_rules_Elegant.items()}
elementtypes["mark"] = "marker"
elementtypes["drif"] = "drift"


class SDDS_Param:

    sdds_param_columns = [
        "ElementName",
        "ElementParameter",
        "ParameterValue",
        "ParameterValueString",
        "ElementType",
    ]

    def __init__(self, filename: str = None, page: int = 0):
        if filename is not None:
            self.import_sdds_param_file(filename)

    def select_param_strings(self, param: list):
        newparam = list(copy(param))
        if len(newparam[3]) > 0:
            del newparam[2]
        else:
            del newparam[3]
        return newparam

    def import_sdds_param_file(self, filename: str, page: int = 0) -> list:
        self.filename = filename
        sdds_par = sddsread(filename)
        data_columns = list(
            zip(*[sdds_par.col(a).data[page] for a in self.sdds_param_columns])
        )
        data_columns = [self.select_param_strings(a) for a in data_columns]
        sdds_par = [
            a
            for a in data_columns
            if "DRIFT" not in a[-1] and a[-1].lower() in elementtypes.keys()
        ]
        groupbygroups = groupby(sdds_par, key=lambda x: re.sub("-C$", "", x[0]))
        rules = {k: {a[1]: a[2] for a in list(g)} for k, g in groupbygroups}
        groupbygroups = groupby(sdds_par, key=lambda x: re.sub("-C$", "", x[0]))
        types = {k: {"TYPE": list(g)[0][-1]} for k, g in groupbygroups}
        self.groups = {k: {**types[k], **rules[k]} for k in rules.keys()}


class SDDS_Floor:

    sdds_position_columns = [
        "ElementName",
        "X",
        "Y",
        "Z",
    ]

    sdds_angle_columns = [
        "ElementName",
        "phi",
        "psi",
        "theta",
    ]

    def __init__(self, filename: str = None, page: int = 0, prefix: str = "."):
        [
            setattr(self, c, [])
            for c in (self.sdds_position_columns + self.sdds_angle_columns)
        ]
        self.prefix = prefix
        self.counter = Counter()
        if filename is not None:
            self.floor_data = self.import_sdds_floor_file(filename, page)

    def get_duplicate_element_names(self) -> list:
        return [k for k, g in groupby(sorted(self.ElementName)) if len(list(g)) > 1]

    def number_element(self, elem):
        if elem in self.duplicates:
            no = self.counter.counter(elem)
            self.counter.add(elem)
            return elem + self.prefix + str(no)
        return elem

    def import_sdds_floor_file(self, filename: str, page: int = 0) -> list:
        self.filename = filename
        sdds_par = sddsread(filename)
        for a in self.sdds_position_columns + self.sdds_angle_columns:
            setattr(self, a, sdds_par.col(a).data[page])
        self.counter = Counter()
        self.duplicates = self.get_duplicate_element_names()
        self.ElementName = [self.number_element(e) for e in self.ElementName]
        # print(self.ElementName)
        # exit()
        self.rawpositiondata = {
            e: list(map(float, chop([x, y, z], 1e-6)))
            for e, x, y, z in list(
                zip(*[getattr(self, a) for a in self.sdds_position_columns])
            )
        }
        self.rawangledata = {
            e: list(map(float, chop([phi, psi, theta], 1e-6)))
            for e, phi, psi, theta in list(
                zip(*[getattr(self, a) for a in self.sdds_angle_columns])
            )
        }
        self.data = {
            e: {"end": self.rawpositiondata[e], "end_rotation": self.rawangledata[e]}
            for e in self.ElementName
        }

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        print(f"{key} missing!")
