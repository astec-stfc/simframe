"""
SimFrame Elements Module

This module defines classes representing specific accelerator lattice elements, all of which inherit from
:class:`~SimulationFramework.Framework_objects.frameworkElement`. Each element has a function for creating
strings or python objects representing that element for the codes supported, and is able to convert
the generic keywords associated with that class to names that are understood by each code.

Classes:
    - :class:`~SimulationFramework.Elements.dipole.dipole`: Dipole magnet.
    - :class:`~SimulationFramework.Elements.kicker.kicker`: Kicker magnet.
    - :class:`~SimulationFramework.Elements.quadrupole.quadrupole`: Quadrupole magnet.
    - :class:`~SimulationFramework.Elements.sextupole.sextupole`: Sextupole magnet.
    - :class:`~SimulationFramework.Elements.octupole.octupole`: Octupole magnet.
    - :class:`~SimulationFramework.Elements.cavity.cavity`: RF cavity.
    - :class:`~SimulationFramework.Elements.wakefield.wakefield`: Wakefield.
    - :class:`~SimulationFramework.Elements.rf_deflecting_cavity.rf_deflecting_cavity`: \
    RF deflecting cavity.
    - :class:`~SimulationFramework.Elements.solenoid.solenoid`: Solenoid magnet.
    - :class:`~SimulationFramework.Elements.aperture.aperture`: Aperture.
    - :class:`~SimulationFramework.Elements.scatter.scatter`: Scatter object.
    - :class:`~SimulationFramework.Elements.scatter.scatter`: Cleaner object.
    - :class:`~SimulationFramework.Elements.wall_current_monitor.wall_current_monitor`: \
    Wall current monitor.
    - :class:`~SimulationFramework.Elements.integrated_current_transformer.integrated_current_transformer`: \
    Integrated current transformer.
    - :class:`~SimulationFramework.Elements.faraday_cup.faraday_cup`: Faraday cup.
    - :class:`~SimulationFramework.Elements.screen.screen`: Diagnostics screen.
    - :class:`~SimulationFramework.Elements.monitor.monitor`: Monitor object.
    - :class:`~SimulationFramework.Elements.faraday_cup.faraday_cup`: Faraday cup.
    - :class:`~SimulationFramework.Elements.watch_point.watch_point`: Watch point.
    - :class:`~SimulationFramework.Elements.beam_position_monitor.beam_position_monitor`: Beam position monitor.
    - :class:`~SimulationFramework.Elements.bunch_length_monitor.bunch_length_monitor`: Bunch length monitor.
    - :class:`~SimulationFramework.Elements.beam_arrival_monitor.beam_arrival_monitor`: Beam arrival monitor.
    - :class:`~SimulationFramework.Elements.collimator.collimator`: Collimator.
    - :class:`~SimulationFramework.Elements.rcollimator.rcollimator`: Rectangular collimator.
    - :class:`~SimulationFramework.Elements.apcontour.apcontour`: Contour.
    - :class:`~SimulationFramework.Elements.center.center`: Center object.
    - :class:`~SimulationFramework.Elements.marker.marker`: Marker object.
    - :class:`~SimulationFramework.Elements.drift.drift`: Drift.
    - :class:`~SimulationFramework.Elements.shutter.shutter`: Shutter.
    - :class:`~SimulationFramework.Elements.valve.valve`: Vacuum valve.
    - :class:`~SimulationFramework.Elements.bellows.bellows`: Bellows.
    - :class:`~SimulationFramework.Elements.fel_modulator.fel_modulator`: FEL modulator.
    - :class:`~SimulationFramework.Elements.wiggler.wiggler`: Wiggler.
    - :class:`~SimulationFramework.Elements.gpt_ccs.gpt_ccs`: GPT coordinate system.
    - :class:`~SimulationFramework.Elements.global_error.global_error`: Global error object.
    - :class:`~SimulationFramework.Elements.charge.charge`: Bunch charge.
"""

from SimulationFramework.Elements.dipole import dipole  # noqa F401
from SimulationFramework.Elements.kicker import kicker  # noqa F401
from SimulationFramework.Elements.quadrupole import quadrupole  # noqa F401
from SimulationFramework.Elements.sextupole import sextupole  # noqa F401
from SimulationFramework.Elements.octupole import octupole  # noqa F401
from SimulationFramework.Elements.cavity import cavity  # noqa F401
from SimulationFramework.Elements.wakefield import (  # noqa F401
    wakefield,
)
from SimulationFramework.Elements.rf_deflecting_cavity import (  # noqa F401
    rf_deflecting_cavity,
)
from SimulationFramework.Elements.solenoid import solenoid  # noqa F401
from SimulationFramework.Elements.aperture import aperture  # noqa F401
from SimulationFramework.Elements.scatter import scatter  # noqa F401
from SimulationFramework.Elements.cleaner import cleaner  # noqa F401
from SimulationFramework.Elements.wall_current_monitor import (  # noqa F401
    wall_current_monitor,
)
from SimulationFramework.Elements.integrated_current_transformer import (  # noqa F401
    integrated_current_transformer,
)
from SimulationFramework.Elements.faraday_cup import faraday_cup  # noqa F401
from SimulationFramework.Elements.screen import screen  # noqa F401
from SimulationFramework.Elements.monitor import monitor  # noqa F401
from SimulationFramework.Elements.watch_point import watch_point  # noqa F401
from SimulationFramework.Elements.beam_position_monitor import (  # noqa F401
    beam_position_monitor,
)
from SimulationFramework.Elements.beam_arrival_monitor import (  # noqa F401
    beam_arrival_monitor,
)
from SimulationFramework.Elements.bunch_length_monitor import (  # noqa F401
    bunch_length_monitor,
)
from SimulationFramework.Elements.collimator import collimator  # noqa F401
from SimulationFramework.Elements.rcollimator import rcollimator  # noqa F401
from SimulationFramework.Elements.apcontour import apcontour  # noqa F401
from SimulationFramework.Elements.center import center  # noqa F401
from SimulationFramework.Elements.marker import marker  # noqa F401
from SimulationFramework.Elements.drift import drift  # noqa F401
from SimulationFramework.Elements.shutter import shutter  # noqa F401
from SimulationFramework.Elements.valve import valve  # noqa F401
from SimulationFramework.Elements.bellows import bellows  # noqa F401
from SimulationFramework.Elements.fel_modulator import fel_modulator  # noqa F401
from SimulationFramework.Elements.wiggler import wiggler  # noqa F401
from SimulationFramework.Elements.charge import charge  # noqa F401
from SimulationFramework.Elements.gpt_ccs import gpt_ccs  # noqa F401
from SimulationFramework.Elements.global_error import global_error  # noqa F401
from SimulationFramework.Framework_objects import (
    chicane,
    s_chicane,
    r56_group,
    element_group,
)  # noqa F401

disallowed_keywords = [
    "allowedkeywords",
    "conversion_rules_elegant",
    "conversion_rules_ocelot",
    "objectdefaults",
    "global_parameters",
    "objectname",
    "beam",
]
