import os
import re
import yaml
from copy import deepcopy
from itertools import groupby
from .counter import Counter
from pydantic import BaseModel, ValidationInfo, field_validator
from typing import List, Dict
import numpy as np
from .sdds_classes_APS import SDDS_Floor
from SimulationFramework.FrameworkHelperFunctions import _rotation_matrix, chop  # type: ignore
from SimulationFramework.Modules.merge_two_dicts import merge_dicts  # type: ignore


class noflow_list(list):
    pass


def noflow_list_rep(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)


class flow_list(list):
    pass


def flow_list_rep(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.add_representer(noflow_list, noflow_list_rep)
yaml.add_representer(flow_list, flow_list_rep)

elementregex = (
    r"[\"]?([a-zA-Z][a-zA-Z0-9\~\@\$\%\^\&\-\_\+\=\{\}\[\]\\\|\/\?\<\>\.\:]*)[\"]?"
)
propertiesregex = r"(?:([^=,&\s]+)\s*=\s*([^,&]+))"
lineregex = r"[\s]*:[\s]*LINE[\s]*=[\s]*"

with open(
    os.path.dirname(os.path.abspath(__file__))
    + "/../../Codes/type_conversion_rules.yaml",
    "r",
) as infile:
    type_conversion_rules = yaml.safe_load(infile)
    type_conversion_rules_Elegant = type_conversion_rules["elegant"]
    type_conversion_rules_Names = type_conversion_rules["name"]
elementtypes = {v.lower(): k.lower() for k, v in type_conversion_rules_Elegant.items()}
elementtypes["drift"] = "drift"
elementtypes["mark"] = "marker"
elementtypes["drif"] = "drift"
elementtypes["edrift"] = "edrift"
elementtypes["csrdrift"] = "csrdrift"
elementtypes["lscdrift"] = "lscdrift"
elementtypes["sext"] = "sextupole"
elementtypes["koct"] = "octupole"

with open(
    os.path.dirname(os.path.abspath(__file__))
    + "/../../Codes/Elegant/keyword_conversion_rules_elegant.yaml",
    "r",
) as infile:
    keyword_conversion_rules_elegant = yaml.safe_load(infile)
keywordrules = {
    kwkey: {v.lower(): k.lower() for k, v in kwvalue.items()}
    for kwkey, kwvalue in keyword_conversion_rules_elegant.items()
}

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Elements/elementkeywords.yaml",
    "r",
) as infile:
    elementkeywords = yaml.safe_load(infile)
simframerules = {
    kwkey: {k.lower(): k.lower() for k in kwvalue["keywords"].keys()}
    for kwkey, kwvalue in elementkeywords.items()
}


def getRules(type: str) -> dict:
    sftype = (
        elementtypes[type.lower()] if type.lower() in elementtypes else type.lower()
    )
    sfrules = (
        merge_dicts(simframerules["common"], simframerules[sftype])
        if sftype in simframerules
        else simframerules["common"]
    )
    return (
        merge_dicts(keywordrules[sftype], keywordrules["general"], sfrules)
        if sftype in keywordrules
        else merge_dicts(keywordrules["general"], sfrules)
    )


def rotate_vector(
    vec: list[float, float, float], angle: float
) -> list[float, float, float]:
    return np.dot(vec, _rotation_matrix(angle))


def convert_numpy_types(v):
    if isinstance(v, (np.ndarray, list, tuple)):
        return flow_list([convert_numpy_types(elem) for elem in v])
    elif isinstance(v, (np.float64, np.float32, np.float16)):
        return float(v)
    elif isinstance(
        v,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(v)
    else:
        return v


class ElegantElement(BaseModel):
    name: str
    properties: dict
    _cavity_updated: bool = False

    _reserved_keys = [
        "type",
        "global_rotation",
        "centre",
    ]

    @field_validator("properties", mode="before")
    @classmethod
    def validate_middle(cls, v: dict) -> dict:
        if "start" in v:
            angle = float(v["angle"]) if "angle" in v else 0
            theta = float(v["global_rotation"][2]) if "global_rotation" in v else 0
            length = float(v["l"]) if "l" in v else 0
            start = v["start"]
            if abs(angle) > 0:
                v["centre"] = convert_numpy_types(
                    list(
                        map(
                            float,
                            chop(
                                start
                                + rotate_vector(
                                    np.array(
                                        [0, 0, length * np.tan(0.5 * angle) / angle]
                                    ),
                                    -theta,
                                )
                            ),
                        )
                    )
                )
            else:
                v["centre"] = convert_numpy_types(
                    list(
                        map(
                            float,
                            chop(
                                start
                                + rotate_vector(np.array([0, 0, length / 2.0]), -theta)
                            ),
                        )
                    )
                )
        cls.__convert_K_to_KL(cls, v)
        return v

    def __convert_K_to_KL(self, v) -> None:
        for n in range(1, 9):
            if "k" + str(n) in v and "l" in v:
                v["k" + str(n) + "l"] = float(v["k" + str(n)]) * float(v["l"])
                del v["k" + str(n)]

    def add_cavity_fields(self):
        if self.type == "cavity" and not self._cavity_updated:
            self.properties["phase"] = convert_numpy_types(
                90 - float(self.properties["phase"])
            )
            cells = 0
            self.properties["field_amplitude"] = convert_numpy_types(
                np.sqrt(2) * float(self.properties["field_amplitude"])
            ) / ((4.1 + cells) * float(self.properties["cell_length"]))
            self._cavity_updated = True

    def update(self, properties: dict) -> None:
        self.properties.update(properties)
        self.validate_middle(self.properties)

    def replace_element_type(self, replacements: dict) -> None:
        if self.properties["type"].lower() not in replacements.values():
            if self.properties["type"].lower() in replacements.keys():
                self.properties["type"] = replacements[self.properties["type"].lower()]
            else:
                raise ValueError(
                    f'"{self.properties["type"].lower()}" type is not allowed!', self
                )

    def replace_keys(self, replacements: dict) -> None:
        if self.type in replacements:
            rules = merge_dicts(replacements[self.type], replacements["general"])
        else:
            rules = replacements["general"]
        for k, newk in rules.items():
            if k in self.properties:
                # print('converting property', self.name, k, newk)
                self.properties[newk] = self.properties.pop(k)

    def filter_properties(self, filter: list) -> dict:
        if self.type in filter:
            rules = merge_dicts(filter[self.type], filter["common"])
        else:
            rules = filter["common"]
        for k in list(self.properties.keys()):
            if k not in rules and k not in self._reserved_keys:
                # print("deleting property", self.name, k)
                del self.properties[k]

    @property
    def length(self):
        return float(self.properties["l"]) if "l" in self.properties else 0

    @property
    def angle(self):
        return (
            float(self.properties["angle"])
            if "angle" in self.properties and abs(float(self.properties["angle"])) > 0
            else 0
        )

    @property
    def theta(self):
        return (
            float(self.properties["global_rotation"][2])
            if "global_rotation" in self.properties
            and abs(float(self.properties["global_rotation"][2])) > 0
            else 0
        )

    @property
    def start(self):
        return self.properties["start"]

    @property
    def end(self):
        return self.properties["end"]

    @property
    def middle(self):
        return self.properties["centre"]

    @property
    def type(self):
        return self.properties["type"]


class ElegantLattice(BaseModel):
    name: str
    lattice: List[str]
    elements: Dict[str, ElegantElement]
    floor: dict
    prefix: str = "."

    @field_validator("elements", mode="before")
    @classmethod
    def validate_elements(
        cls, v: Dict[str, ElegantElement], info: ValidationInfo
    ) -> Dict[str, ElegantElement]:
        lattice = info.data["lattice"]
        for elem in v:
            if elem not in lattice:
                raise ValueError(f"{elem} is not in lattice!?")
        for elem in lattice:
            if elem not in v:
                raise ValueError(f"{elem} is not in elements!?")
        return v

    def __get_duplicate_element_names(self) -> list:
        return [k for k, g in groupby(sorted(self.lattice)) if len(list(g)) > 1]

    def model_post_init(self, ___context):
        counter = Counter()
        duplicates = self.__get_duplicate_element_names()
        new_lattice = []
        new_elements = {}
        startpos = [0, 0, 0]
        startangle = [0, 0, 0]
        for elem in self.lattice:
            if elem in duplicates:
                no = counter.counter(elem)
                counter.add(elem)
                name = elem + self.prefix + str(no)
            else:
                name = elem
            new_elements[name] = deepcopy(self.elements[elem])
            new_elements[name].name = name
            properties = {
                k: convert_numpy_types(v) for k, v in self.floor[name].items()
            }
            properties.update({"start": convert_numpy_types(startpos)})
            properties.update({"global_rotation": convert_numpy_types(startangle)})
            new_elements[name].update(properties)
            new_lattice += [name]
            startpos = self.floor[name]["end"]
            startangle = self.floor[name]["end_rotation"]
        self.lattice = new_lattice
        self.elements = new_elements

    def bends(self):
        return [
            self.elements[elem]
            for elem in self.lattice
            if abs(float(self.elements[elem].angle)) > 0
        ]

    def quadrupoles(self):
        return [
            self.elements[elem]
            for elem in self.lattice
            if "k1l" in self.elements[elem].properties
            and abs(float(self.elements[elem].properties["k1l"])) > 0
        ]

    def cavities(self):
        return [
            self.elements[elem]
            for elem in self.lattice
            if (
                "frequency" in self.elements[elem].properties
                and abs(float(self.elements[elem].properties["frequency"])) > 0
            )
            or (
                "freq" in self.elements[elem].properties
                and abs(float(self.elements[elem].properties["freq"])) > 0
            )
        ]

    def index(self, element: str | ElegantElement) -> int | None:
        element = element.name if isinstance(element, ElegantElement) else element
        return self.lattice.index(element)

    def element_index(self, element: str | ElegantElement, offset: int = 0):
        idx = self.index(element)
        return self.elements[self.lattice[idx + offset]]

    def replace_keys(self, replacements: dict) -> None:
        for elem in self.elements.values():
            elem.replace_keys(replacements)

    def replace_element_types(self, replacements: dict) -> None:
        for elem in self.elements.values():
            elem.replace_element_type(replacements)

    def filter_element_properties(self, filter: list) -> dict:
        for elem in self.elements.values():
            elem.filter_properties(filter)

    def update_cavities(self):
        for elem in self.elements.values():
            if elem.type == "cavity":
                elem.add_cavity_fields()


class ReadElegantLattice:

    def __init__(
        self,
        lattice_file: str = "ukxfel_save.lte",
        floor_file: str = "ukxfel.flr",
        base_dir: str = ".",
        allowed_element_types: list | None = None,
    ):
        self.base_dir = base_dir
        self.allowed_element_types = allowed_element_types
        if lattice_file is not None and floor_file is not None:
            self.lattice_file = lattice_file
            self.floor_file = floor_file
            self.load_lattice(self.lattice_file, self.floor_file)

    def flatten(self, A):
        rt = []
        for item in A:
            if isinstance(item, list):
                rt.extend(self.flatten(item))
            else:
                rt.append(item)
        return rt

    def expand_lattice(self, elem: str, lattices: dict) -> str:
        if elem in lattices.keys():
            return [self.expand_lattice(e, lattices) for e in lattices[elem]]
        return elem

    @property
    def lattice_names(self):
        return re.findall(elementregex + lineregex, "\n".join(self.latticestrings))

    def load_lattice(
        self, lattice_file: str, floor_file: str, lattice_name: str | None = None
    ):
        self.lattice_file = lattice_file
        self.floor_file = floor_file
        self.floor = SDDS_Floor(os.path.join(self.base_dir, floor_file))
        self.lattice_path = os.path.join(self.base_dir, lattice_file)
        join = False
        with open(self.lattice_path, "r") as file:
            lattice = []
            for line in file:
                if join:
                    lattice[-1] = lattice[-1][:-1] + line.rstrip()
                else:
                    lattice += [line.rstrip()]
                if line.rstrip()[-2:] == ",&":
                    join = True
                else:
                    join = False
        self.latticestrings = lattice + []
        self.get_lattice_lines()
        self.get_elements()
        self.lattices = {}
        for lattice_name, lattice in self.latticelists.items():
            elements = self.get_elements_in_lattice(lattice_name)
            self.lattices[lattice_name] = ElegantLattice(
                name=lattice_name,
                lattice=lattice,
                elements=elements,
                floor=self.floor.data,
            )

    def __convert_to_float_if_possible(self, string: str) -> str | float:
        regex_int = re.match(r"^-?(\d+)$", string)
        regex_float = re.match(r"^-?\d+(?:\.\d+)$", string)
        if regex_int is not None:
            return int(regex_int.group(0))
        elif regex_float is not None:
            return float(regex_float.group(0))
        return str(string)

    def get_elements(self):
        self.elements = {}
        removed_elements = []
        for ls in self.latticestrings:
            nametypematch = re.findall(elementregex + r": ([\w]+)", ls)
            if len(nametypematch) > 0:
                name, type = nametypematch[0]
                if type.lower() != "line":
                    if (
                        self.allowed_element_types is None
                        or len(self.allowed_element_types) == 0
                        or type.lower() in self.allowed_element_types
                    ):
                        properties = {"type": type.lower()}
                        properties.update(
                            {
                                k.lower(): convert_numpy_types(
                                    self.__convert_to_float_if_possible(v)
                                )
                                for k, v in re.findall(propertiesregex, ls)
                            }
                        )
                        self.elements[name] = ElegantElement(
                            name=name, properties=properties
                        )
                    else:
                        properties = {
                            k.lower(): v for k, v in re.findall(propertiesregex, ls)
                        }
                        if (
                            "l" in properties
                            and abs(float(properties["l"])) > 0
                        ):
                            raise ValueError(
                                f'"{type.lower()}" type is not allowed and length > 0 - aborting!'
                            )
                        else:
                            removed_elements += [name]
                            print(
                                f'"{type.lower()}" type is not allowed but ignored'
                            )
        self.remove_elements_from_lattices(removed_elements)
        return self.elements

    def get_lattice_lines(self):
        lattice_names = self.lattice_names
        lattices = {}
        for lattname in lattice_names:
            lattices[lattname] = re.findall(
                elementregex,
                [
                    re.sub(lattname + lineregex, "", latt)
                    for latt in self.latticestrings
                    if len(latt) > len(lattname) and latt[: len(lattname)] == lattname
                ][0],
            )
        for lattname, lattice in lattices.items():
            lattices[lattname] = self.flatten(
                [self.expand_lattice(elem, lattices) for elem in lattice]
            )
        self.latticelists = lattices

    def remove_elements_from_lattices(self, removed_elements: list) -> None:
        for lattice_name, lattice in self.latticelists.items():
            self.latticelists[lattice_name] = [
                elem for elem in lattice if elem not in removed_elements
            ]

    def get_elements_in_lattice(self, lattice: str):
        return {elem: self.elements[elem] for elem in self.latticelists[lattice]}

    def to_YAML(self, lattice: str, stream=None):
        yaml_dict = {"elements": {}}
        elements_dict = yaml_dict["elements"]
        for elemname, elem in self.lattices[lattice].elements.items():
            if 'drif' not in elem.type:
                elements_dict[elemname] = elem.properties
        if stream is not None:
            yaml.dump(yaml_dict, stream, default_flow_style=False)
        return yaml.dump(yaml_dict, default_flow_style=False)
