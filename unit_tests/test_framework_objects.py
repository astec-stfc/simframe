from SimulationFramework.Framework_elements import *
from SimulationFramework.Framework_objects import (
    frameworkLattice,
    frameworkElement,
)
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

def all_subclasses(cls):
    subclasses = cls.__subclasses__()
    for subclass in subclasses:
        subclasses += all_subclasses(subclass)
    return subclasses

def test_framework_object():
    element = frameworkObject(
        objectname="ELEMENT",
        objecttype="marker",
        centre=[0, 0, 0],
    )
    with pytest.raises(NameError):
        frameworkElement(
            objectname="ELEMENT",
            objecttype="element",
            centre=[0, 0, 0],
        )
    with pytest.raises(ValueError):
        frameworkElement(
            objectname=1,
            objecttype="element",
            centre=[0, 0, 0],
        )
    with pytest.raises(ValueError):
        frameworkElement(
            objectname="ELEMENT",
            objecttype=1,
            centre=[0, 0, 0],
        )
    element = frameworkElement(
        objectname="ELEMENT",
        objecttype="marker",
        centre=[0, 0, 0],
    )
    element.centre= -1


def test_framework_elements():
    exclude = ["global", "gpt", "astra", "csrtrack"]
    for elem in all_subclasses(frameworkElement):
        if not any(x in elem.__name__ for x in exclude):
            element = elem(
                objectname="ELEMENT",
                objecttype=elem.__name__,
                centre=[0, 0, 0],
            )
            assert element.x == element.y == element.z == element.dx == element.dy == element.dz == 0