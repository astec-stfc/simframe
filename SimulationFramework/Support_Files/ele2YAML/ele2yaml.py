import os
from pprint import pprint
from copy import copy
from itertools import groupby
import yaml
import re
import numpy as np
from SimulationFramework.Modules.merge_two_dicts import merge_dicts
from SimulationFramework.FrameworkHelperFunctions import _rotation_matrix
# from SimulationFramework.Framework_objects import frameworkCounter
from elegant_lattice import ReadElegantLattice
from sdds_classes import SDDS_Param, SDDS_Floor
import pysdds

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/../../Codes/type_conversion_rules.yaml",
    "r",
) as infile:
    type_conversion_rules = yaml.safe_load(infile)
    type_conversion_rules_Elegant = type_conversion_rules["elegant"]
    type_conversion_rules_Names = type_conversion_rules["name"]
elementtypes = {v.lower(): k.lower() for k, v in type_conversion_rules_Elegant.items()}
elementtypes["mark"] = "marker"
elementtypes["drif"] = "drift"

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


def applyRules(assoc: dict, rules: dict) -> dict:
    return dict(
        [
            [rules[key], value] if key in rules else [key, value]
            for key, value in assoc.items()
        ]
    )


def rotate_vector(
    vec: list[float, float, float], angle: float
) -> list[float, float, float]:
    return np.dot(vec, _rotation_matrix(angle))


def positioncentre(
    length: float,
    position: list[float, float, float],
    angles: list[float, float, float],
) -> list[float, float, float]:
    if angles[2] > 0:
        vector = [length * np.tan(0.5 * angles[2]) / angles[2]]
    else:
        vector = [0, 0, length / 2]
    return position + rotate_vector(vector, angles[0])


def modifyCavity(cavity: dict) -> dict:
    cavity["phase"] = 90 - cavity["phase"]
    cells = 0
    cavity["field_amplitude"] = (np.sqrt(2) * cavity["field_amplitude"]) / (
                                (4.1 + cells) * cavity["cell_length"]
    )
    cavity["longitudinal_wakefield_enable"] = 1
    cavity["transverse_wakefield_enable"] = 1
    cavity["tcolumn"] = cavity["tcolumn"]
    cavity["body_focus_model"] = "None"
    return {k: v.replace("./output/", "") for k, v in cavity.items()}


def elemiddle(name, elem, start):
    angle = elem["angle"] if "angle" in elem and abs(elem["angle"]) > 0 else 0
    if abs(angle) > 0:
        return start + [0, 0, elem["length"] * np.tan(0.5 * elem("angle")) / elem["angle"]]
    else:
        return centerassoc[name]


base_dir = "C:\\Users\\jkj62\\Documents\\GitHub\\SimFrame_Examples\\CLARA\\Ocelot\\"


if __name__ == "__main__":
    elatt = ReadElegantLattice(lattice_file='ukxfel_save.lte', base_dir=base_dir)
    # print(elatt.get_elements())
    lattice = elatt.lattices['L0001'].number_duplicate_elements().lattice
    floor = SDDS_Floor(os.path.join(base_dir, 'ukxfel.flr'))
    # print(lattice)
    # print(floor.data)
    print({e: floor[e] for e in lattice})

