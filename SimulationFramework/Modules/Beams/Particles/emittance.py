import numpy as np

class emittance():

    def __init__(self, beam):
        self.beam = beam

    def __repr__(self):
        return repr({p: self.emittance(p) for p in ('x', 'y')})

    @property
    def ex(self):
        return self.emittance('x')
    @property
    def ey(self):
        return self.emittance('y')
    @property
    def enx(self):
        return self.normalized_emittance('x')
    @property
    def eny(self):
        return self.normalized_emittance('y')

    def emittance_calc(self, x, xp, p=None):
        cov_x = self.beam.covariance(x, x)
        cov_xp = self.beam.covariance(xp, xp)
        cov_x_xp = self.beam.covariance(x, xp)
        emittance = np.sqrt(cov_x * cov_xp - cov_x_xp**2) if (cov_x * cov_xp - cov_x_xp**2) > 0 else 0
        if p is None:
            return emittance
        else:
            gamma = np.mean(p)/self.beam.E0_eV
            return gamma*emittance

    def normalized_emittance(self, plane='x', corrected=False):
        if corrected:
            return self.emittance_calc(getattr(self.beam, plane+'c'), getattr(self.beam, plane+'pc'), self.beam.cp)
        else:
            return self.emittance_calc(getattr(self.beam, plane), getattr(self.beam, plane+'p'), self.beam.cp)

    def emittance(self, plane='x', corrected=False):
        if corrected:
            return self.emittance_calc(getattr(self.beam, plane+'c'), getattr(self.beam, plane+'pc'), None)
        else:
            return self.emittance_calc(getattr(self.beam, plane), getattr(self.beam, plane+'p'), None)

    @property
    def normalized_horizontal_emittance(self):
        return self.emittance_calc(self.beam.x, self.beam.xp, self.beam.cp)
    @property
    def normalized_vertical_emittance(self):
        return self.emittance_calc(self.beam.y, self.beam.yp, self.beam.cp)
    @property
    def horizontal_emittance(self):
        return self.emittance_calc(self.beam.x, self.beam.xp)
    @property
    def vertical_emittance(self):
        return self.emittance_calc(self.beam.y, self.beam.yp)

    @property
    def horizontal_emittance_90(self):
        emit = self.beam.twiss.horizontal_emittance
        alpha = self.beam.twiss.alpha_x
        beta = self.beam.twiss.beta_x
        gamma = self.beam.twiss.gamma_x
        emiti = gamma * self.beam.x**2 + 2 * alpha * self.beam.x * self.beam.xp + beta * self.beam.xp * self.beam.xp
        return sorted(emiti)[int(0.9*len(emiti)-0.5)]
    @property
    def normalized_horizontal_emittance_90(self):
        emit = self.beam.horizontal_emittance_90
        return np.mean(self.beam.cp)/self.beam.E0_eV * emit
    @property
    def vertical_emittance_90(self):
        emit = self.beam.vertical_emittance
        alpha = self.beam.twiss.alpha_y
        beta = self.beam.twiss.beta_y
        gamma = self.beam.twiss.gamma_y
        emiti = gamma * self.beam.y**2 + 2 * alpha * self.beam.y * self.beam.yp + beta * self.beam.yp * self.beam.yp
        return sorted(emiti)[int(0.9*len(emiti)-0.5)]
    @property
    def normalized_vertical_emittance_90(self):
        emit = self.beam.vertical_emittance_90
        return np.mean(self.beam.cp)/self.beam.E0_eV * emit
