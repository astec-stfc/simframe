import os
import re
from itertools import groupby
from counter import Counter
from pydantic import BaseModel
from typing import List, Dict

elementregex = r'[\"]?([a-zA-Z][a-zA-Z0-9\~\@\$\%\^\&\-\_\+\=\{\}\[\]\\\|\/\?\<\>\.\:]*)[\"]?'
propertiesregex = r'(?:([^=,&\s]+)\s*=\s*([^,&]+))'


class ElegantLattice(BaseModel):
    name: str
    lattice: List[str]
    elements: Dict[str, dict]

    def get_duplicate_element_names(self) -> list:
        return [k for k, g in groupby(sorted(self.lattice)) if len(list(g)) > 1]

    def number_duplicate_elements(self, prefix: str = '.') -> None:
        counter = Counter()
        duplicates = self.get_duplicate_element_names()
        new_lattice = []
        new_elements = {}
        for elem in self.lattice:
            if elem in duplicates:
                no = counter.counter(elem)
                counter.add(elem)
                name = elem + prefix + str(no)
                new_elements[name] = self.elements[elem]
            else:
                name = elem
            new_lattice += [name]
        return ElegantLattice(name=self.name, lattice=new_lattice, elements=new_elements)


class ReadElegantLattice():

    def __init__(self, lattice_file: str = None, base_dir: str = '.'):
        self.base_dir = base_dir
        if lattice_file is not None:
            self.lattice_file = lattice_file
            self.load_lattice(self.lattice_file)

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
        return re.findall(elementregex + r': LINE = ', '\n'.join(self.latticestrings))

    def load_lattice(self, lattice_file: str = 'ukxfel_save.lte', lattice_name: str | None = None):
        self.lattice_path = os.path.join(self.base_dir, lattice_file)
        join = False
        with open(self.lattice_path, 'r') as file:
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
            self.lattices[lattice_name] = ElegantLattice(name=lattice_name, lattice=lattice, elements=elements)

    def get_elements(self):
        self.elements = {}
        for ls in self.latticestrings:
            nametypematch = re.findall(elementregex + r': ([\w]+)', ls)
            if len(nametypematch) > 0:
                name, type = nametypematch[0]
                if type.lower() != "line":
                    self.elements[name] = {'TYPE': type}
                    properties = re.findall(propertiesregex, ls)
                    self.elements[name].update({k: v for k, v in properties})
        return self.elements

    def get_lattice_lines(self):
        lattice_names = self.lattice_names
        lattices = {}
        for lattname in lattice_names:
            lattices[lattname] = re.findall(elementregex, [latt.replace(lattname + ': LINE = ', '') for latt in self.latticestrings if len(latt) > len(lattname) and latt[:len(lattname)] == lattname][0])
        for lattname, lattice in lattices.items():
            lattices[lattname] = self.flatten([self.expand_lattice(elem, lattices) for elem in lattice])
        self.latticelists = lattices

    def get_elements_in_lattice(self, lattice: str):
        return {elem: self.elements[elem] for elem in self.latticelists[lattice]}
