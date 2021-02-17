import munch
import numpy as np
from ... import constants
from ...units import UnitValue, mean, std

class sigmas(munch.Munch):

    def __init__(self, beam):
        self.beam = beam

    @property
    def sigma_x(self):
        return self.Sx
    @property
    def sigma_y(self):
        return self.Sy
    @property
    def sigma_t(self):
        return self.St
    @property
    def sigma_z(self):
        return self.Sz
    @property
    def sigma_cp(self):
        return self.momentum_spread

    @property
    def Sx(self):
        return np.sqrt(self.beam.covariance(self.beam.x, self.beam.x))
    @property
    def Sy(self):
        return np.sqrt(self.beam.covariance(self.beam.y, self.beam.y))
    @property
    def Sz(self):
        return np.sqrt(self.beam.covariance(self.beam.z, self.beam.z))
    @property
    def St(self):
        return np.sqrt(self.beam.covariance(self.beam.t, self.beam.t))
    @property
    def momentum_spread(self):
        return std(self.beam.cp)
        # return self.beam.cp.std()/np.mean(self.beam.cp)
    @property
    def linear_chirp_z(self):
        return -1*std(self.beam.Bz*self.beam.speed_of_light*self.beam.t)/self.momentum_spread/100
