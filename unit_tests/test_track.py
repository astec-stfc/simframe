from SimulationFramework.Framework_elements import *
from SimulationFramework.Framework_objects import frameworkLattice
from SimulationFramework.Framework import runSetup
from SimulationFramework.Framework_Settings import FrameworkSettings
import SimulationFramework.Modules.Beams as rbf  # noqa E402
import SimulationFramework.Modules.Twiss as rtf
import SimulationFramework.Codes.Executables as exes
from SimulationFramework.Framework_lattices import elegantLattice, astraLattice, ocelotLattice, gptLattice
import pytest
from test_beam import simple_beam