import numpy as np
from pydantic import BaseModel, computed_field
from ...units import UnitValue


class emittance(BaseModel):

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, beam, *args, **kwargs):
        super(emittance, self).__init__(*args, **kwargs)
        self.beam = beam

    # def __repr__(self):
    #     return repr({p: self.emittance(p) for p in ("x", "y")})

    def model_dump(self, *args, **kwargs):
        # Only include computed fields
        computed_keys = {
            f for f in self.__pydantic_decorators__.computed_fields.keys()
        }
        full_dump = super().model_dump(*args, **kwargs)
        return {k: v for k, v in full_dump.items() if k in computed_keys}

    @computed_field
    @property
    def ex(self) -> UnitValue:
        return self.emittance("x")

    @computed_field
    @property
    def ey(self) -> UnitValue:
        return self.emittance("y")

    @computed_field
    @property
    def enx(self) -> UnitValue:
        return self.normalized_emittance("x")

    @computed_field
    @property
    def eny(self) -> UnitValue:
        return self.normalized_emittance("y")

    @computed_field
    @property
    def ecx(self) -> UnitValue:
        return self.horizontal_emittance_corrected

    @computed_field
    @property
    def ecy(self) -> UnitValue:
        return self.vertical_emittance_corrected

    @computed_field
    @property
    def ecnx(self) -> UnitValue:
        return self.normalised_horizontal_emittance_corrected

    @computed_field
    @property
    def ecny(self) -> UnitValue:
        return self.normalised_vertical_emittance_corrected

    def emittance_calc(self, x, xp, p=None, units="m-rad") -> UnitValue:
        cov_x = self.beam.covariance(x, x)
        cov_xp = self.beam.covariance(xp, xp)
        cov_x_xp = self.beam.covariance(x, xp)
        emittance = (
            np.sqrt(cov_x * cov_xp - cov_x_xp**2)
            if (cov_x * cov_xp - cov_x_xp**2) > 0
            else 0
        )
        if p is not None:
            beta = np.mean(self.beam.Bz)
            gamma = np.mean(p) / (np.mean(self.beam.particle_rest_energy_eV) * beta)
            emittance = gamma * emittance
        return emittance

    def normalized_emittance(self, plane="x", corrected=False) -> UnitValue:
        if corrected:
            return self.emittance_calc(
                getattr(self.beam, plane + "c"),
                getattr(self.beam, plane + "pc"),
                self.beam.cpz,
            )
        else:
            return self.emittance_calc(
                getattr(self.beam, plane),
                getattr(self.beam, plane + "p"),
                self.beam.cpz,
            )

    def emittance(self, plane="x", corrected=False) -> UnitValue:
        if corrected:
            return self.emittance_calc(
                getattr(self.beam, plane + "c"), getattr(self.beam, plane + "pc"), None
            )
        else:
            return self.emittance_calc(
                getattr(self.beam, plane), getattr(self.beam, plane + "p"), None
            )

    @computed_field
    @property
    def normalized_horizontal_emittance(self) -> UnitValue:
        return self.emittance_calc(self.beam.x, self.beam.xp, self.beam.cp)

    @computed_field
    @property
    def normalized_vertical_emittance(self) -> UnitValue:
        return self.emittance_calc(self.beam.y, self.beam.yp, self.beam.cp)

    @computed_field
    @property
    def horizontal_emittance(self) -> UnitValue:
        return self.emittance_calc(self.beam.x, self.beam.xp)

    @computed_field
    @property
    def vertical_emittance(self) -> UnitValue:
        return self.emittance_calc(self.beam.y, self.beam.yp)

    @computed_field
    @property
    def horizontal_emittance_90(self) -> UnitValue:
        alpha = self.beam.twiss.alpha_x
        beta = self.beam.twiss.beta_x
        gamma = self.beam.twiss.gamma_x
        emiti = (
            gamma * self.beam.x**2
            + 2 * alpha * self.beam.x * self.beam.xp
            + beta * self.beam.xp * self.beam.xp
        )
        return sorted(emiti)[int(0.9 * len(emiti) - 0.5)]

    @computed_field
    @property
    def normalized_horizontal_emittance_90(self) -> UnitValue:
        emit = self.horizontal_emittance_90
        return np.mean(self.beam.cp / self.beam.E0_eV) * emit

    @computed_field
    @property
    def vertical_emittance_90(self) -> UnitValue:
        alpha = self.beam.twiss.alpha_y
        beta = self.beam.twiss.beta_y
        gamma = self.beam.twiss.gamma_y
        emiti = (
            gamma * self.beam.y**2
            + 2 * alpha * self.beam.y * self.beam.yp
            + beta * self.beam.yp * self.beam.yp
        )
        return sorted(emiti)[int(0.9 * len(emiti) - 0.5)]

    @computed_field
    @property
    def normalized_vertical_emittance_90(self) -> UnitValue:
        emit = self.vertical_emittance_90
        return np.mean(self.beam.cp / self.beam.E0_eV) * emit

    @computed_field
    @property
    def horizontal_emittance_corrected(self) -> UnitValue:
        xc = self.beam.eta_corrected(self.beam.x)
        xpc = self.beam.eta_corrected(self.beam.xp)
        return self.emittance_calc(xc, xpc)

    @computed_field
    @property
    def vertical_emittance_corrected(self) -> UnitValue:
        yc = self.beam.eta_corrected(self.beam.y)
        ypc = self.beam.eta_corrected(self.beam.yp)
        return self.emittance_calc(yc, ypc)

    @computed_field
    @property
    def normalised_horizontal_emittance_corrected(self) -> UnitValue:
        xc = self.beam.eta_corrected(self.beam.x)
        xpc = self.beam.eta_corrected(self.beam.xp)
        return self.emittance_calc(xc, xpc, self.beam.cp)

    @computed_field
    @property
    def normalised_vertical_emittance_corrected(self) -> UnitValue:
        yc = self.beam.eta_corrected(self.beam.y)
        ypc = self.beam.eta_corrected(self.beam.yp)
        return self.emittance_calc(yc, ypc, self.beam.cp)
