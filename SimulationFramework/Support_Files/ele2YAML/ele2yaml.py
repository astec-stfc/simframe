import numpy as np
import yaml
from .elegant_lattice import (
    ReadElegantLattice,
    elementtypes,
    keywordrules,
    simframerules,
)

_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG


def ndarray_representer(dumper: yaml.Dumper, array: np.ndarray) -> yaml.Node:
    return dumper.represent_list(array.tolist())


def dict_representer(dumper, data):
    return dumper.represent_dict(iter(list(data.items())))


def dict_constructor(loader, node):
    return dict(loader.construct_pairs(node))


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
yaml.add_representer(dict, dict_representer)
yaml.add_constructor(_mapping_tag, dict_constructor)
yaml.Dumper.ignore_aliases = lambda *args: True


def convert_lattice(lattice_file, line, base_dir, floor_file):
    elatt = ReadElegantLattice(
        lattice_file=lattice_file,
        base_dir=base_dir,
        allowed_element_types=elementtypes.keys(),
        floor_file=floor_file,
    )
    lattice = elatt.lattices[line]
    lattice.replace_element_types(elementtypes)
    lattice.replace_keys(keywordrules)
    lattice.filter_element_properties(simframerules)
    lattice.update_cavities()
    return elatt


base_dir = "C:\\Users\\jkj62\\Documents\\GitHub\\SimFrame_Examples\\CLARA\\Ocelot\\"


if __name__ == "__main__":
    elatt = convert_lattice(
        lattice_file="ukxfel_save.lte", line="L0001", base_dir=base_dir
    )
    with open("test.yaml", "w") as stream:
        elatt.to_YAML("L0001", stream)
