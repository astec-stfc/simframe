import warnings
import numpy as np
from munch import Munch
from ... import constants
from .emittance import emittance as emittanceobject
from .twiss import twiss as twissobject
from .slice import slice as sliceobject
from .sigmas import sigmas as sigmasobject
from .centroids import centroids as centroidsobject
from ...UnitFloat import UnitArray, UnitFloat, unit_multiply

class Particles(Munch):

    properties = {
    'x': 'm',
    'y': 'm',
    'z': 'm',
    't': 's',
    'px': 'kg*m/s',
    'py': 'kg*m/s',
    'pz': 'kg*m/s',
    'p':  'kg*m/s'
    }

    particle_mass = UnitFloat(constants.m_e, 'kg')
    E0 = UnitFloat(particle_mass * constants.speed_of_light**2, 'J')
    E0_eV = UnitFloat(E0 / constants.elementary_charge, 'eV/c')
    q_over_c = UnitFloat(constants.elementary_charge / constants.speed_of_light, 'C/c')
    speed_of_light = UnitFloat(constants.speed_of_light, 'm/s')

    ''' ********************  Statistical Parameters  ************************* '''

    def __init__(self):
        super(Particles, self).__init__(self)

    def __getitem__(self, key):
        if isinstance(super(Particles, self).__getitem__(key),(list, tuple)):
            return np.array(super(Particles, self).__getitem__(key))
        else:
            try:
                return super(Particles, self).__getitem__(key)
            except KeyError:
                raise AttributeError(key)

    @property
    def slice(self):
        if not hasattr(self, '_slice'):
            self._slice = sliceobject(self)
        return self._slice

    @property
    def emittance(self):
        if not hasattr(self, '_emittance'):
            self._emittance = emittanceobject(self)
        return self._emittance

    @property
    def twiss(self):
        if not hasattr(self, '_twiss'):
            self._twiss = twissobject(self)
        return self._twiss

    @property
    def sigmas(self):
        if not hasattr(self, '_sigmas'):
            self._sigmas = sigmasobject(self)
        return self._sigmas

    @property
    def centroids(self):
        if not hasattr(self, '_mean'):
            self._mean = centroidsobject(self)
        return self._mean

    def covariance(self, u, up):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ans = 0
            if len(u) > 1 and len(up) > 1:
                ans = UnitArray(np.cov([u,up])[0,1], unit_multiply(u.units, up.units, divide=False))
            return ans

    def eta_correlation(self, u):
        return self.covariance(u,self.p) / self.covariance(self.p, self.p)

    def eta_corrected(self, u):
        return u - self.eta_correlation(u)*self.p

    @property
    def x(self):
        return UnitArray(self['x'], 'm')
    @property
    def y(self):
        return UnitArray(self['y'], 'm')
    @property
    def z(self):
        return UnitArray(self['z'], 'm')
    @property
    def px(self):
        return UnitArray(self['px'], 'kg*m/s')
    @property
    def py(self):
        return UnitArray(self['py'], 'kg*m/s')
    @property
    def pz(self):
        return UnitArray(self['pz'], 'kg*m/s')
    @property
    def t(self):
        return UnitArray(self['t'], 's')
    @property
    def xc(self):
        return UnitArray(self.eta_corrected(self.x), 'm')
    @property
    def xpc(self):
        return UnitArray(self.eta_corrected(self.xp), '')
    @property
    def yc(self):
        return UnitArray(self.eta_corrected(self.y), 'm')
    @property
    def ypc(self):
        return UnitArray(self.eta_corrected(self.yp), '')

    @property
    def cpx(self):
        return UnitArray(self['px'] / self.q_over_c, 'eV/c')
    @property
    def cpy(self):
        return UnitArray(self['py'] / self.q_over_c, 'eV/c')
    @property
    def cpz(self):
        return UnitArray(self['pz'] / self.q_over_c, 'eV/c')
    @property
    def deltap(self):
        return UnitArray((self.cp - np.mean(self.cp)) / np.mean(self.cp), '')
    @property
    def xp(self):
        return UnitArray(np.arctan(self.px/self.pz), 'rad')
    @property
    def yp(self):
        return UnitArray(np.arctan(self.py/self.pz), 'rad')
    @property
    def p(self):
        return UnitArray(self.cp * self.q_over_c, 'kg*m/s')
    @property
    def cp(self):
        return UnitArray(np.sqrt(self.cpx**2 + self.cpy**2 + self.cpz**2), 'eV/c')
    @property
    def Brho(self):
        return UnitArray(np.mean(self.p) / constants.elementary_charge, 'T*m')
    @property
    def gamma(self):
        return UnitArray(np.sqrt(1+(self.cp/self.E0_eV)**2), '')
    @property
    def BetaGamma(self):
        return UnitArray(self.cp/self.E0_eV, '')
    @property
    def energy(self):
        return UnitArray(self.gamma * self.E0_eV, 'eV')
    @property
    def vx(self):
        velocity_conversion = 1 / (constants.m_e * self.gamma)
        return UnitArray(velocity_conversion * self.px, 'm*s')
    @property
    def vy(self):
        velocity_conversion = 1 / (constants.m_e * self.gamma)
        return UnitArray(velocity_conversion * self.py, 'm*s')
    @property
    def vz(self):
        velocity_conversion = 1 / (constants.m_e * self.gamma)
        return UnitArray(velocity_conversion * self.pz, 'm*s')
    @property
    def Bx(self):
        return UnitArray(self.vx / constants.speed_of_light, '')
    @property
    def By(self):
        return UnitArray(self.vy / constants.speed_of_light, '')
    @property
    def Bz(self):
        return UnitArray(self.vz / constants.speed_of_light, '')
    @property
    def Q(self):
        return UnitArray(self['total_charge'], 'C')
    @property
    def total_charge(self):
        return UnitFloat(self['total_charge'], 'C')
    @property
    def charge(self):
        return UnitArray(self['charge'], 'C')
    # @property
    # def sigma_z(self):
    #     return self.rms(self.Bz*constants.speed_of_light*(self['t'] - np.mean(self['t'])))

    @property
    def kinetic_energy(self):
        return UnitArray(np.array((np.sqrt(self.E0**2 + self.cp**2) - self.E0**2)), 'J')

    @property
    def mean_energy(self):
        return UnitArray(np.mean(self.kinetic_energy), 'J')

    def computeCorrelations(self, x, y):
        xAve = np.mean(x)
        yAve = np.mean(y)
        C11 = 0
        C12 = 0
        C22 = 0
        for i, ii in enumerate(x):
            dx = x[i] - xAve
            dy = y[i] - yAve
            C11 += dx*dx
            C12 += dx*dy
            C22 += dy*dy
        C11 /= len(x)
        C12 /= len(x)
        C22 /= len(x)
        return C11, C12, C22

    def performTransformation(self, x, xp, beta=False, alpha=False, nEmit=False):
        p = self.cp
        pAve = np.mean(p)
        p = [a / pAve - 1 for a in p]
        eta1, etap1, _ = self.calculate_etax()
        for i, ii in enumerate(x):
            x[i] -= p[i] * eta1
            xp[i] -= p[i] * etap1

        S11, S12, S22 = self.computeCorrelations(x, xp)
        emit = np.sqrt(S11*S22 - S12**2)
        beta1 = S11/emit
        alpha1 = -S12/emit
        beta2 = beta if beta is not False else beta1
        alpha2 = alpha if alpha is not False else alpha1
        R11 = beta2/np.sqrt(beta1*beta2)
        R12 = 0
        R21 = (alpha1-alpha2)/np.sqrt(beta1*beta2)
        R22 = beta1/np.sqrt(beta1*beta2)
        if nEmit is not False:
            factor = np.sqrt(nEmit / (emit*pAve))
            R11 *= factor
            R12 *= factor
            R22 *= factor
            R21 *= factor
        for i, ii in enumerate(x):
            x0 = x[i]
            xp0 = xp[i]
            x[i] = R11 * x0 + R12 * xp0
            xp[i] = R21*x0 + R22*xp0
        return x, xp

    def rematchXPlane(self, beta=False, alpha=False, nEmit=False):
        x, xp = self.performTransformation(self.x, self.xp, beta, alpha, nEmit)
        self['x'] = x
        self['xp'] = xp

        cpz = self.cp / np.sqrt(self['xp']**2 + self.yp**2 + 1)
        cpx = self['xp'] * cpz
        cpy = self.yp * cpz
        self['px'] = cpx * self.q_over_c
        self['py'] = cpy * self.q_over_c
        self['pz'] = cpz * self.q_over_c

    def rematchYPlane(self, beta=False, alpha=False, nEmit=False):
        y, yp = self.performTransformation(self.y, self.yp, beta, alpha, nEmit)
        self['y'] = y
        self['yp'] = yp

        cpz = self.cp / np.sqrt(self.xp**2 + self['yp']**2 + 1)
        cpx = self.xp * cpz
        cpy = self['yp'] * cpz
        self['px'] = cpx * self.q_over_c
        self['py'] = cpy * self.q_over_c
        self['pz'] = cpz * self.q_over_c

    def performTransformationPeakISlice(self, xslice, xpslice, x, xp, beta=False, alpha=False, nEmit=False):
        p = self.cp
        pAve = np.mean(p)
        p = [a / pAve - 1 for a in p]
        eta1, etap1, _ = self.calculate_etax()
        for i, ii in enumerate(x):
            x[i] -= p[i] * eta1
            xp[i] -= p[i] * etap1

        S11, S12, S22 = self.computeCorrelations(xslice, xpslice)
        emit = np.sqrt(S11*S22 - S12**2)
        beta1 = S11/emit
        alpha1 = -S12/emit
        beta2 = beta if beta is not False else beta1
        alpha2 = alpha if alpha is not False else alpha1
        R11 = beta2/np.sqrt(beta1*beta2)
        R12 = 0
        R21 = (alpha1-alpha2)/np.sqrt(beta1*beta2)
        R22 = beta1/np.sqrt(beta1*beta2)
        if nEmit is not False:
            factor = np.sqrt(nEmit / (emit*pAve))
            R11 *= factor
            R12 *= factor
            R22 *= factor
            R21 *= factor
        for i, ii in enumerate(x):
            x0 = x[i]
            xp0 = xp[i]
            x[i] = R11 * x0 + R12 * xp0
            xp[i] = R21*x0 + R22*xp0
        return x, xp

    def rematchXPlanePeakISlice(self, beta=False, alpha=False, nEmit=False):
        peakIPosition = self.slice_max_peak_current_slice
        xslice = self.slice_data(self.x)[peakIPosition]
        xpslice = self.slice_data(self.xp)[peakIPosition]
        x, xp = self.performTransformationPeakISlice(xslice, xpslice, self.x, self.xp, beta, alpha, nEmit)
        self['x'] = x
        self['xp'] = xp

        cpz = self.cp / np.sqrt(self['xp']**2 + self.yp**2 + 1)
        cpx = self['xp'] * cpz
        cpy = self.yp * cpz
        self['px'] = cpx * self.q_over_c
        self['py'] = cpy * self.q_over_c
        self['pz'] = cpz * self.q_over_c

    def rematchYPlanePeakISlice(self, beta=False, alpha=False, nEmit=False):
        peakIPosition = self.slice_max_peak_current_slice
        yslice = self.slice_data(self.y)[peakIPosition]
        ypslice = self.slice_data(self.yp)[peakIPosition]
        y, yp = self.performTransformationPeakISlice(yslice, ypslice, self.y, self.yp, beta, alpha, nEmit)
        self['y'] = y
        self['yp'] = yp

        cpz = self.cp / np.sqrt(self.xp**2 + self['yp']**2 + 1)
        cpx = self.xp * cpz
        cpy = self['yp'] * cpz
        self['px'] = cpx * self.q_over_c
        self['py'] = cpy * self.q_over_c
        self['pz'] = cpz * self.q_over_c
