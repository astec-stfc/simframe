import numpy as np
from .. import constants
from ocelot.cpbd.io import load_particle_array, save_particle_array
from ocelot.cpbd.beam import ParticleArray

def read_ocelot_beam_file(self, filename):
    self.filename = filename
    self['code'] = "OCELOT"
    parray = load_particle_array(filename)
    self._beam['x'] = parray.x()
    self._beam['y'] = parray.y()
    self._beam['t'] = parray.tau() / constants.speed_of_light
    self._beam['p'] = parray.p()
    self._beam['px'] = parray.px()
    cp = (self._beam['p']) * 1e9
    cpz = cp / np.sqrt(self._beam['px'] ** 2 + self._beam['py'] ** 2 + 1)
    cpx = self._beam['xp'] * cpz
    cpy = self._beam['yp'] * cpz
    self._beam['px'] = cpx * self.q_over_c
    self._beam['py'] = cpy * self.q_over_c
    self._beam['pz'] = cpz * self.q_over_c
    self._beam['total_charge'] = np.sum(parray.q_array)
    self._beam['z'] = (-1 * self._beam.Bz * constants.speed_of_light) * (
                self._beam.t - np.mean(self._beam.t))  # np.full(len(self.t), 0)
    self._beam['charge'] = np.full(len(self._beam['z']), self._beam['total_charge'] / len(self._beam['x']))
    self._beam['nmacro'] = np.full(len(self._beam['z']), 1)

def write_ocelot_beam_file(self, filename, write=True):
    """Save an npz file for ocelot."""
    # {x, xp, y, yp, t, p, particleID}
    E = np.mean(self.mean_energy)
    x = self.x
    y = self.y
    xp = self.px / self.pz
    yp = self.py / self.pz
    p = self.energy / E
    tau = self.t * constants.speed_of_light
    s = np.mean(tau)

    rparticles = np.array([x, xp, y, yp, tau, p])
    q_array = self.Q / self.nmacro
    parray = ParticleArray(n=len(x))
    parray.rparticles = rparticles
    parray.q_array = q_array
    parray.E = E * 1e-9
    parray.s = s
    if write:
        save_particle_array(filename, parray)
    return parray