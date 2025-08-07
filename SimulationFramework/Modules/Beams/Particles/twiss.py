from pydantic import BaseModel, computed_field
import numpy as np
from ...units import UnitValue
from typing import Dict


class twiss(BaseModel):

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, beam, *args, **kwargs):
        super(twiss, self).__init__(*args, **kwargs)
        self.beam = beam

    def __repr__(self):
        return repr(self.normal)

    def __setattr__(self, name, value):
        if name == "my_field":
            if not isinstance(value, str):
                raise ValueError("my_field must be a string")
        super().__setattr__(name, value)

    @property
    def normal(self) -> Dict:
        return {
            p: getattr(self, p)
            for p in (
                "normalized_horizontal_emittance",
                "horizontal_emittance",
                "alpha_x",
                "beta_x",
                "normalized_vertical_emittance",
                "vertical_emittance",
                "alpha_y",
                "beta_y",
            )
        }

    @property
    def corrected(self) -> Dict:
        return {
            p + "_corrected": getattr(self, p + "_corrected")
            for p in (
                "normalized_horizontal_emittance",
                "horizontal_emittance",
                "alpha_x",
                "beta_x",
                "normalized_vertical_emittance",
                "vertical_emittance",
                "alpha_y",
                "beta_y",
            )
        }

    def model_dump(self, *args, **kwargs):
        # Only include computed fields
        computed_keys = {
            f for f in self.__pydantic_decorators__.computed_fields.keys()
        }
        full_dump = super().model_dump(*args, **kwargs)
        return {k: v for k, v in full_dump.items() if k in computed_keys}

    @computed_field
    @property
    def normalized_horizontal_emittance(self) -> UnitValue:
        return self.beam.emittance.normalized_horizontal_emittance

    @computed_field
    @property
    def normalized_vertical_emittance(self) -> UnitValue:
        return self.beam.emittance.normalized_vertical_emittance

    @computed_field
    @property
    def horizontal_emittance(self) -> UnitValue:
        return self.beam.emittance.horizontal_emittance

    @computed_field
    @property
    def vertical_emittance(self) -> UnitValue:
        return self.beam.emittance.vertical_emittance

    @computed_field
    @property
    def horizontal_emittance_corrected(self) -> UnitValue:
        return self.beam.emittance.horizontal_emittance_corrected

    @computed_field
    @property
    def vertical_emittance_corrected(self) -> UnitValue:
        return self.beam.emittance.vertical_emittance_corrected

    @computed_field
    @property
    def beta_x(self) -> UnitValue:
        return (
            self.beam.covariance(self.beam.x, self.beam.x) / self.horizontal_emittance
        )

    @computed_field
    @property
    def alpha_x(self) -> UnitValue:
        return (
            -1
            * self.beam.covariance(self.beam.x, self.beam.xp)
            / self.horizontal_emittance
        )

    @computed_field
    @property
    def gamma_x(self) -> UnitValue:
        return (
            self.beam.covariance(self.beam.xp, self.beam.xp) / self.horizontal_emittance
        )

    @computed_field
    @property
    def beta_y(self) -> UnitValue:
        return self.beam.covariance(self.beam.y, self.beam.y) / self.vertical_emittance

    @computed_field
    @property
    def alpha_y(self) -> UnitValue:
        return (
            -1
            * self.beam.covariance(self.beam.y, self.beam.yp)
            / self.vertical_emittance
        )

    @computed_field
    @property
    def gamma_y(self) -> UnitValue:
        return (
            self.beam.covariance(self.beam.yp, self.beam.yp) / self.vertical_emittance
        )

    @property
    def twiss_analysis(self) -> tuple:
        return (
            self.beam.emittance.horizontal_emittance,
            self.alpha_x,
            self.beta_x,
            self.gamma_x,
            self.beam.emittance.vertical_emittance,
            self.alpha_y,
            self.beta_y,
            self.gamma_y,
        )

    @computed_field
    @property
    def beta_x_corrected(self) -> UnitValue:
        xc = self.beam.eta_corrected(self.beam.x)
        return self.beam.covariance(xc, xc) / self.horizontal_emittance_corrected

    @computed_field
    @property
    def alpha_x_corrected(self) -> UnitValue:
        xc = self.beam.eta_corrected(self.beam.x)
        xpc = self.beam.eta_corrected(self.beam.xp)
        return -1 * self.beam.covariance(xc, xpc) / self.horizontal_emittance_corrected

    @computed_field
    @property
    def gamma_x_corrected(self) -> UnitValue:
        xpc = self.beam.eta_corrected(self.beam.xp)
        return self.beam.covariance(xpc, xpc) / self.horizontal_emittance_corrected

    @computed_field
    @property
    def beta_y_corrected(self) -> UnitValue:
        yc = self.beam.eta_corrected(self.beam.y)
        return self.beam.covariance(yc, yc) / self.vertical_emittance_corrected

    @computed_field
    @property
    def alpha_y_corrected(self) -> UnitValue:
        yc = self.beam.eta_corrected(self.beam.y)
        ypc = self.beam.eta_corrected(self.beam.yp)
        return -1 * self.beam.covariance(yc, ypc) / self.vertical_emittance_corrected

    @computed_field
    @property
    def gamma_y_corrected(self) -> UnitValue:
        ypc = self.beam.eta_corrected(self.beam.yp)
        return self.beam.covariance(ypc, ypc) / self.vertical_emittance_corrected

    @property
    def twiss_analysis_corrected(self) -> tuple:
        return (
            self.horizontal_emittance_corrected,
            self.alpha_x_corrected,
            self.beta_x_corrected,
            self.gamma_x_corrected,
            self.vertical_emittance_corrected,
            self.alpha_y_corrected,
            self.beta_y_corrected,
            self.gamma_y_corrected,
        )

    @computed_field
    @property
    def eta_x(self) -> UnitValue:
        return self.calculate_etax()[0]

    @computed_field
    @property
    def eta_xp(self) -> UnitValue:
        return self.calculate_etax()[1]

    def calculate_etax(self) -> tuple:
        p = self.beam.cpz
        pAve = np.mean(p)
        p = p / pAve - 1  # [(a / pAve) - 1 for a in p]
        S16, S66 = self.beam.covariance(self.beam.x, p), self.beam.covariance(p, p)
        eta1 = S16 / S66 if S66 else 0
        S26 = self.beam.covariance(self.beam.xp, p)
        etap1 = S26 / S66 if S66 else 0
        return eta1, etap1, np.mean(self.beam.t)
