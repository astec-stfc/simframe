import os
import numpy as np
from .. import constants
from ..units import UnitValue
from ocelot.cpbd.io import load_particle_array, save_particle_array
from ocelot.cpbd.beam import ParticleArray


def read_ocelot_beam_file(self, filename):
    self.filename = filename
    self.code = "OCELOT"
    self._beam.particle_rest_energy_eV = self.E0_eV
    parray = load_particle_array(filename)
    self._beam.particle_mass = UnitValue(
        np.full(len(parray.x()), constants.m_e),
        units="kg",
    )
    self._beam.particle_charge = UnitValue(parray.q_array, units="C")
    self._beam.particle_mass = UnitValue(np.full(len(parray.x()), constants.m_e), units="kg")
    self._beam.particle_rest_energy = UnitValue(
        (
                self._beam.particle_mass * constants.speed_of_light ** 2
        ),
        units="J",
    )
    self._beam.particle_rest_energy_eV = UnitValue(
        (
                self._beam.particle_rest_energy / constants.elementary_charge
        ),
        units="eV",
    )
    # self._beam.gamma = UnitValue(parray.gamma, units="")
    self._beam.x = UnitValue(parray.x(), units="m")
    self._beam.y = UnitValue(parray.y(), units="m")
    self._beam.t = UnitValue((parray.s + parray.tau()) / constants.speed_of_light, units="s")
    # self._beam.p = UnitValue(parray.energies, units="eV/c")
    self._beam.px = UnitValue(parray.px() * parray.energies * 1e9 * self.q_over_c, units="kg*m/s")
    self._beam.py = UnitValue(parray.py() * parray.energies * 1e9 * self.q_over_c, units="kg*m/s")
    cp = parray.energies * 1e9
    self._beam.pz = UnitValue(
        (
            self.q_over_c * cp / np.sqrt(parray.px() ** 2 + parray.py() ** 2 + 1)
        ),
        units="kg*m/s",
    )
    self._beam.set_total_charge(-1 * abs(np.sum(parray.q_array)))
    self._beam.z = UnitValue(
        (-1 * self._beam.Bz * constants.speed_of_light) * (
            self._beam.t - np.mean(self._beam.t)
        ),
        units="m",
    )  # np.full(len(self.t), 0)
    self._beam.nmacro = UnitValue(np.full(len(self._beam.x), 1), units="")


def write_ocelot_beam_file(self, filename: str=None, write: bool=True):
    """Save an npz file for ocelot."""
    # {x, xp, y, yp, t, p, particleID}
    if filename is None:
        fn = os.path.splitext(self.filename)
        filename = fn[0].strip(".ocelot").strip(".openpmd") + ".ocelot.npz"
    E = self.energy.mean().val * 1e-9
    x = self.x.val
    y = self.y.val
    xp = self.cpx.val / self.cpz.val
    yp = self.cpy.val / self.cpz.val
    p = ((self.energy.val * 1e-9) - E) / E
    tau = (self.t.val - np.mean(self.t.val)) * constants.speed_of_light
    s = np.mean(self.t.val) * constants.speed_of_light

    rparticles = np.array([x, xp, y, yp, tau, p])
    q_array = np.array([np.abs(float(self.Q.val / len(x))) for _ in x])
    parray = ParticleArray(n=len(x))
    parray.rparticles = rparticles
    parray.q_array = q_array
    parray.E = E
    parray.s = s
    if write:
        save_particle_array(filename, parray)
    return parray
