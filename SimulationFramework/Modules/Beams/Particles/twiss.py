import munch
import numpy as np

class twiss(munch.Munch):

    def __init__(self, beam):
        self.beam = beam

    def __repr__(self):
        return repr({p: getattr(self,p) for p in ('alpha_x', 'beta_x', 'alpha_y', 'beta_y')})

    @property
    def beta_x(self):
        return self.beam.covariance(self.beam.x,self.beam.x) / self.beam.emittance.horizontal_emittance
    @property
    def alpha_x(self):
        return -1*self.beam.covariance(self.beam.x,self.beam.xp) / self.beam.emittance.horizontal_emittance
    @property
    def gamma_x(self):
        return self.beam.covariance(self.beam.xp,self.beam.xp) / self.beam.emittance.horizontal_emittance
    @property
    def beta_y(self):
        return self.beam.covariance(self.beam.y,self.beam.y) / self.beam.emittance.vertical_emittance
    @property
    def alpha_y(self):
        return -1*self.beam.covariance(self.beam.y,self.beam.yp) / self.beam.emittance.vertical_emittance
    @property
    def gamma_y(self):
        return self.beam.covariance(self.beam.yp,self.beam.yp) / self.beam.emittance.vertical_emittance
    @property
    def twiss_analysis(self):
        return self.beam.emittance.horizontal_emittance, self.alpha_x, self.beta_x, self.gamma_x, self.beam.emittance.vertical_emittance, self.alpha_y, self.beta_y, self.gamma_y

    def eta_correlation(self, u):
        return self.beam.covariance(u,self.beam.p) / self.beam.covariance(self.beam.p, self.beam.p)

    def eta_corrected(self, u):
        return u - self.eta_correlation(u)*self.beam.p

    @property
    def horizontal_emittance_corrected(self):
        xc = self.eta_corrected(self.beam.x)
        xpc = self.eta_corrected(self.beam.xp)
        return self.beam.emittance.emittance_calc(xc, xpc)
    @property
    def vertical_emittance_corrected(self):
        yc = self.eta_corrected(self.beam.y)
        ypc = self.eta_corrected(self.beam.yp)
        return self.beam.emittance.emittance_calc(yc, ypc)
    @property
    def beta_x_corrected(self):
        xc = self.eta_corrected(self.beam.x)
        return self.beam.covariance(xc, xc) / self.horizontal_emittance_corrected
    @property
    def alpha_x_corrected(self):
        xc = self.eta_corrected(self.beam.x)
        xpc = self.eta_corrected(self.beam.xp)
        return -1*self.beam.covariance(xc, xpc) / self.horizontal_emittance_corrected
    @property
    def gamma_x_corrected(self):
        xpc = self.eta_corrected(self.beam.xp)
        return self.beam.covariance(xpc, xpc) / self.horizontal_emittance_corrected
    @property
    def beta_y_corrected(self):
        yc = self.eta_corrected(self.beam.y)
        return self.beam.covariance(yc,yc) / self.vertical_emittance_corrected
    @property
    def alpha_y_corrected(self):
        yc = self.eta_corrected(self.beam.y)
        ypc = self.eta_corrected(self.beam.yp)
        return -1*self.beam.covariance(yc, ypc) / self.vertical_emittance_corrected
    @property
    def gamma_y_corrected(self):
        ypc = self.eta_corrected(self.beam.yp)
        return self.beam.covariance(ypc,ypc) / self.vertical_emittance_corrected
    @property
    def twiss_analysis_corrected(self):
        return  self.horizontal_emittance_corrected, self.alpha_x_corrected, self.beta_x_corrected, self.gamma_x_corrected, \
                self.vertical_emittance_corrected, self.alpha_y_corrected, self.beta_y_corrected, self.gamma_y_corrected

    @property
    def eta_x(self):
        # print('etax = ', self.calculate_etax()[0])
        return self.calculate_etax()[0]

    @property
    def eta_xp(self):
        return self.calculate_etax()[1]

    def calculate_etax(self):
        p = self.beam.cpz
        pAve = np.mean(p)
        p = [(a / pAve) - 1 for a in p]
        S16, S66 = self.beam.covariance(self.beam.x, p), self.beam.covariance(p, p)
        eta1 = S16/S66 if S66 else 0
        S26 = self.beam.covariance(self.beam.xp, p)
        etap1 = S26/S66 if S66 else 0
        return eta1, etap1, np.mean(self.beam.t)
