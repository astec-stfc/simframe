import numpy as np
from pydantic import BaseModel

class centroids(BaseModel):

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, beam, *args, **kwargs):
        super(centroids, self).__init__(*args, **kwargs)
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
    def mean_cpx(self):
        return self.Cpx

    @property
    def mean_cpy(self):
        return self.Cpy

    @property
    def mean_cpz(self):
        return self.Cpz

    @property
    def mean_energy(self):
        return self.CEn

    @property
    def mean_gamma(self):
        return self.Cgamma

    @property
    def mean_cp(self):
        return self.Ccp

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

    @property
    def Cgamma(self):
        return np.mean(self.beam.gamma)

    @property
    def Ccp(self):
        return np.mean(self.beam.cp)

    @property
    def CEn(self):
        return np.mean(self.beam.cp * self.beam.particle_rest_energy)
