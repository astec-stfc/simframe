import munch
import numpy as np
from ... import constants
from ...units import UnitValue, mean, covariance

class centroids(munch.Munch):

    def __init__(self, beam):
        self.beam = beam

    @property
    def mean_x(self):
        return self.Cx
    @property
    def mean_y(self):
        return self.Cy
    @property
    def mean_t(self):
        return self.Ct
    @property
    def mean_z(self):
        return self.Cz
    @property
    def mean_cp(self):
        return self.Cp
    @property
    def mean_cxp(self):
        return self.Cpx
    @property
    def mean_cpy(self):
        return self.Cpy
    @property
    def mean_cpz(self):
        return self.Cpz

    @property
    def Cx(self):
        return mean(self.beam.x)
    @property
    def Cy(self):
        return mean(self.beam.y)
    @property
    def Cz(self):
        return mean(self.beam.z)
    @property
    def Ct(self):
        return mean(self.beam.t)
    @property
    def Cp(self):
        return mean(self.beam.cp)
    @property
    def Cpx(self):
        return mean(self.beam.cpx)
    @property
    def Cpy(self):
        return mean(self.beam.cpy)
    @property
    def Cpz(self):
        return mean(self.beam.cpz)
