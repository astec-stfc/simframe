import numpy as np
from .. import constants
from ..units import UnitValue
from ocelot.cpbd.io import load_particle_array, save_particle_array
from ocelot.cpbd.beam import ParticleArray


def read_ocelot_beam_file(self, filename):
    self.filename = filename
    self["code"] = "OCELOT"
    self._beam["particle_rest_energy_eV"] = self.E0_eV
    parray = load_particle_array(filename)
    self._beam["particle_mass"] = np.full(
        len(parray.x()), UnitValue(constants.m_e, "kg")
    )
    self._beam["particle_charge"] = parray.q_array
    self._beam["gamma"] = parray.gamma
    self._beam["x"] = parray.x()
    self._beam["y"] = parray.y()
    self._beam["t"] = (parray.s + parray.tau()) / constants.speed_of_light
    self._beam["p"] = parray.energies
    self._beam["px"] = parray.px() * parray.energies * 1e9 * self.q_over_c
    self._beam["py"] = parray.py() * parray.energies * 1e9 * self.q_over_c
    cp = (self._beam["p"]) * 1e9
    self._beam["pz"] = (
        self.q_over_c * cp / np.sqrt(parray.px() ** 2 + parray.py() ** 2 + 1)
    )
    self._beam["total_charge"] = -1 * abs(np.sum(parray.q_array))
    self._beam["z"] = (1 * self._beam.Bz * constants.speed_of_light) * (
        self._beam["t"] - np.mean(self._beam["t"])
    )  # np.full(len(self.t), 0)
    self._beam["charge"] = np.full(
        len(self._beam["x"]), self._beam["total_charge"] / len(self._beam["x"])
    )
    self._beam["nmacro"] = np.full(len(self._beam["x"]), 1)


def write_ocelot_beam_file(self, filename, write=True):
    """Save an npz file for ocelot."""
    # {x, xp, y, yp, t, p, particleID}
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
