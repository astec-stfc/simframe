import numpy as np
from pydantic import BaseModel, computed_field
from ...units import UnitValue

class centroids(BaseModel):

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, beam, *args, **kwargs):
        super(centroids, self).__init__(*args, **kwargs)
        self.beam = beam

    @computed_field
    @property
    def mean_x(self) -> UnitValue:
        return self.Cx

    @computed_field
    @property
    def mean_y(self) -> UnitValue:
        return self.Cy

    @computed_field
    @property
    def mean_t(self) -> UnitValue:
        return self.Ct

    @computed_field
    @property
    def mean_z(self) -> UnitValue:
        return self.Cz

    @computed_field
    @property
    def mean_cpx(self) -> UnitValue:
        return self.Cpx

    @computed_field
    @property
    def mean_cpy(self) -> UnitValue:
        return self.Cpy

    @computed_field
    @property
    def mean_cpz(self) -> UnitValue:
        return self.Cpz

    @computed_field
    @property
    def mean_energy(self) -> UnitValue:
        return self.CEn

    @computed_field
    @property
    def mean_gamma(self) -> UnitValue:
        return self.Cgamma

    @computed_field
    @property
    def mean_cp(self) -> UnitValue:
        return self.Ccp

    @computed_field
    @property
    def Cx(self) -> UnitValue:
        return np.mean(self.beam.x)

    @computed_field
    @property
    def Cy(self) -> UnitValue:
        return np.mean(self.beam.y)

    @computed_field
    @property
    def Cz(self) -> UnitValue:
        return np.mean(self.beam.z)

    @computed_field
    @property
    def Ct(self) -> UnitValue:
        return np.mean(self.beam.t)

    @computed_field
    @property
    def Cp(self) -> UnitValue:
        return np.mean(self.beam.cp)

    @computed_field
    @property
    def Cpx(self) -> UnitValue:
        return np.mean(self.beam.cpx)

    @computed_field
    @property
    def Cpy(self) -> UnitValue:
        return np.mean(self.beam.cpy)

    @computed_field
    @property
    def Cpz(self) -> UnitValue:
        return np.mean(self.beam.cpz)

    @computed_field
    @property
    def Cxp(self) -> UnitValue:
        return np.mean(self.beam.xp)

    @computed_field
    @property
    def Cyp(self) -> UnitValue:
        return np.mean(self.beam.yp)

    @computed_field
    @property
    def Cgamma(self) -> UnitValue:
        return np.mean(self.beam.gamma)

    @computed_field
    @property
    def Ccp(self) -> UnitValue:
        return np.mean(self.beam.cp)

    @computed_field
    @property
    def CEn(self) -> UnitValue:
        return np.mean(self.beam.cp * self.beam.particle_rest_energy)
