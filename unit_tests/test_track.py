from importlib.metadata import PackageNotFoundError

from SimulationFramework.Framework_elements import *
from SimulationFramework.Framework import runSetup
from SimulationFramework.Framework_Settings import FrameworkSettings
import SimulationFramework.Modules.Beams as rbf  # noqa E402
import SimulationFramework.Modules.Twiss as rtf
import SimulationFramework.Codes.Executables as exes
from SimulationFramework.Framework_lattices import elegantLattice, astraLattice, ocelotLattice, gptLattice
from test_beam import simple_beam
import pytest
from copy import deepcopy
import os
import shutil

@pytest.fixture
def test_fodo_elements():
    begin = {
        "objectname": "BEGIN",
        "objecttype": "marker",
        "centre": [0, 0, 0.0],
    }

    q1 = {
        "objectname": "QUAD1",
        "objecttype": "quadrupole",
        "centre": [0, 0, 0.05],
        "length": 0.1,
        "k1l": -2,
    }

    q2 = {
        "objectname": "QUAD2",
        "objecttype": "quadrupole",
        "centre": [0, 0, 0.2],
        "length": 0.1,
        "k1l": 4,
    }

    q3 = {
        "objectname": "QUAD3",
        "objecttype": "quadrupole",
        "centre": [0, 0, 0.35],
        "length": 0.1,
        "k1l": -1,
        "bore": 0.037,
    }

    end = {
        "objectname": "END",
        "objecttype": "marker",
        "centre": [0, 0, 0.4],
    }

    beginelem = marker(**begin)
    q1elem = quadrupole(**q1)
    q2elem = quadrupole(**q2)
    q3elem = quadrupole(**q3)
    endelem = marker(**end)

    elementobjects = {
        name: elem for name, elem in zip(
            ["BEGIN"] + [f"QUAD{i}" for i in range(1, 4)] + ["END"],
            [beginelem, q1elem, q2elem, q3elem, endelem, ],
        )
    }
    return elementobjects

# @pytest.mark.parametrize("code,lattice_class", [
#     ("elegant", elegantLattice),
#     ("astra", astraLattice),
#     ("ocelot", ocelotLattice),
#     # ("gpt", gptLattice),
# ])
def prepare_lattice(simple_beam, test_fodo_elements, code, lattice_class, remove=True):
    if not os.path.isdir(f'./fodo/{code}'):
        os.makedirs(f'./fodo/{code}')
    simple_beam.write_HDF5_beam_file(f'./fodo/{code}/BEGIN.hdf5')
    groupobjects = {}
    rs = runSetup()
    settings = FrameworkSettings()
    try:
        import SimCodes  # type: ignore

        SimCodesLocation = os.path.dirname(SimCodes.__file__) + "/"
    except ImportError:
        raise PackageNotFoundError("SimCodes package must be installed for test to work")
    global_parameters = {
        "beam": deepcopy(simple_beam),
        "simcodes_location": SimCodesLocation,
        "master_subdir": f'./fodo/{code}',
        "master_lattice_location": f'./fodo/{code}',
        "GPTLICENSE": "1115315511",
    }
    executables = exes.Executables(global_parameters)
    latticedict = {
        "FODO": {
            "code": code,
            "input": {},
            "output": {
                "start_element": "BEGIN",
                "end_element": "END",
            }
        }
    }
    latattrs = {
        "name": "FODO",
        "file_block": latticedict["FODO"],
        "elementObjects": test_fodo_elements,
        "groupObjects": groupobjects,
        "runSettings": rs,
        "settings": settings,
        "executables": executables,
        "global_parameters": global_parameters,
        "lscDrifts": False,
        "csrDrifts": False,
    }
    lattice = lattice_class(**latattrs)
    if remove:
        shutil.rmtree(f'./fodo/{code}')
    return lattice

@pytest.mark.parametrize("code,lattice_class", [
    ("elegant", elegantLattice),
    ("astra", astraLattice),
    ("ocelot", ocelotLattice),
    # ("gpt", gptLattice),
])
def test_track(simple_beam, test_fodo_elements, code, lattice_class):
    lattice = prepare_lattice(simple_beam, test_fodo_elements, code, lattice_class, remove=False)
    lattice.preProcess()
    lattice.write()
    lattice.run()
    lattice.postProcess()

