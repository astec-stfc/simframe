import munch
import numpy as np
from ... import constants
from ...units import UnitValue

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
        return np.mean(self.beam.x)
    @property
    def Cy(self):
        return np.mean(self.beam.y)
    @property
    def Cz(self):
        return np.mean(self.beam.z)
    @property
    def Ct(self):
        return np.mean(self.beam.t)
    @property
    def Cp(self):
        return np.mean(self.beam.cp)
    @property
    def Cpx(self):
        return np.mean(self.beam.cpx)
    @property
    def Cpy(self):
        return np.mean(self.beam.cpy)
    @property
    def Cpz(self):
        return np.mean(self.beam.cpz)
    @property
    def Cxp(self):
        return np.mean(self.beam.xp)
    @property
    def Cyp(self):
        return np.mean(self.beam.yp)
