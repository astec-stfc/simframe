import warnings
from math import copysign
import numpy as np
from ... import constants
from .emittance import emittance as emittanceobject
from .twiss import twiss as twissobject
from .slice import slice as sliceobject
from .sigmas import sigmas as sigmasobject
from .centroids import centroids as centroidsobject
from .kde import kde as kdeobject

try:
    from .mve import MVE as MVEobject
except ImportError:
    pass
from ...units import UnitValue, unit_multiply
from pydantic import BaseModel
from typing import Dict

class Particles(BaseModel):

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    # properties = {
    #     "x": "m",
    #     "y": "m",
    #     "z": "m",
    #     "t": "s",
    #     "px": "kg*m/s",
    #     "py": "kg*m/s",
    #     "pz": "kg*m/s",
    #     "p": "kg*m/s",
    #     "particle_mass": "kg",
    #     "particle_rest_energy": "J",
    #     "particle_rest_energy_eV": "eV/c",
    #     "particle_charge": "C",
    # }

    # particle_mass = UnitValue(constants.m_e, "kg")
    # E0 = UnitValue(particle_mass * constants.speed_of_light**2, "J")
    # E0_eV = UnitValue(E0 / constants.elementary_charge, "eV/c")
    q_over_c: UnitValue = UnitValue(constants.elementary_charge / constants.speed_of_light, "C/c")
    speed_of_light: UnitValue = UnitValue(constants.speed_of_light, "m/s")
    mass: UnitValue | list | np.ndarray = None
    particle_rest_energy: UnitValue | list | np.ndarray = None
    particle_rest_energy_eV: UnitValue | list | np.ndarray = None
    particle_charge: UnitValue | list | np.ndarray = None
    charge: UnitValue | list | np.ndarray = None
    clock: UnitValue | list | np.ndarray = None
    t: UnitValue | list | np.ndarray = None
    total_charge: UnitValue | float = None
    x: UnitValue | list | np.ndarray = None
    y: UnitValue | list | np.ndarray = None
    z: UnitValue | list | np.ndarray = None
    px: UnitValue | list | np.ndarray = None
    py: UnitValue | list | np.ndarray = None
    pz: UnitValue | list | np.ndarray = None
    status: UnitValue | list | np.ndarray = None
    nmacro: int = None
    theta: UnitValue | float = None

    mass_index: Dict = {
        1: constants.m_e,  # electron
        2: constants.m_e,  # positron
        3: constants.m_p,  # proton
        4: constants.m_p,  # hydrogen ion
    }
    charge_sign_index: Dict = {1: -1, 2: 1, 3: 1, 4: 1}

    def sign(self, x):
        return copysign(1, x)

    def get_particle_index(self, m, q):
        if m == constants.m_e:
            if self.sign(q) > 0:
                # print('found positron')
                return 2
            # print('found electron')
            return 1
        elif m == constants.m_p:
            if self.sign(q) > 0:
                # print('found proton')
                return 3
            # print('found h-')
            return 4
        elif (0.9 * constants.m_e) < m < (1.1 * constants.m_e):
            if self.sign(q) < 0:
                # print('found electron')
                return 1
            else:
                # print('found positron')
                return 2
        elif (0.9 * constants.m_p) < m < (1.1 * constants.m_p):
            if self.sign(q) > 0:
                # print('found proton')
                return 3
            # print('found h-')
            return 4

    """ ********************  Statistical Parameters  ************************* """

    def __init__(self, *args, **kwargs):
        super(Particles, self).__init__(*args, **kwargs)
    #
    # def __getitem__(self, key):
    #     if isinstance(super(Particles, self).__getitem__(key), (list, tuple)):
    #         return np.array(super(Particles, self).__getitem__(key))
    #     else:
    #         try:
    #             return super(Particles, self).__getitem__(key)
    #         except KeyError:
    #             raise AttributeError(key)

    @property
    def slice(self):
        if not hasattr(self, "_slice"):
            self._slice = sliceobject(self)
        return self._slice

    @property
    def emittance(self):
        if not hasattr(self, "_emittance"):
            self._emittance = emittanceobject(self)
        return self._emittance

    @property
    def twiss(self):
        if not hasattr(self, "_twiss"):
            self._twiss = twissobject(self)
        return self._twiss

    @property
    def sigmas(self):
        if not hasattr(self, "_sigmas"):
            self._sigmas = sigmasobject(self)
        return self._sigmas

    @property
    def centroids(self):
        if not hasattr(self, "_mean"):
            self._mean = centroidsobject(self)
        return self._mean

    @property
    def kde(self):
        if not hasattr(self, "_kde"):
            self._kde = kdeobject(self)
        return self._kde

    @property
    def mve(self):
        if not hasattr(self, "_mve"):
            self._mve = MVEobject(self)
        return self._mve

    def covariance(self, u, up):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ans = 0
            if len(u) > 1 and len(up) > 1:
                ans = UnitValue(
                    np.cov([u, up])[0, 1],
                    unit_multiply(u.units, up.units, divide=False),
                )
            return ans

    def eta_correlation(self, u):
        return self.covariance(u, self.p) / self.covariance(self.p, self.p)

    def eta_corrected(self, u):
        return u - self.eta_correlation(u) * self.p

    def apply_mask(self, mask):
        prebeam = self.fullbeam
        cutbeam = prebeam[mask, :]
        self.x, self.y, self.z, self.px, self.py, self.pz = cutbeam.T

    @property
    def fullbeam(self):
        return np.array([self.x, self.y, self.z, self.px, self.py, self.pz]).T

    @property
    def particle_index(self):
        return [
            self.get_particle_index(m, q)
            for m, q in zip(self.particle_mass, self.particle_charge)
        ]

    @property
    def chargesign(self):
        return [self.sign(q) for q in self.charge]

    @property
    def xc(self):
        return UnitValue(self.eta_corrected(self.x), "m")

    @property
    def xpc(self):
        return UnitValue(self.eta_corrected(self.xp), "")

    @property
    def yc(self):
        return UnitValue(self.eta_corrected(self.y), "m")

    @property
    def ypc(self):
        return UnitValue(self.eta_corrected(self.yp), "")

    @property
    def cpx(self):
        return UnitValue(self.px / self.q_over_c, "eV/c")

    @property
    def cpy(self):
        return UnitValue(self.py / self.q_over_c, "eV/c")

    @property
    def cpz(self):
        return UnitValue(self.pz / self.q_over_c, "eV/c")

    @property
    def deltap(self):
        return (self.cp - np.mean(self.cp)) / np.mean(self.cp)

    @property
    def xp(self):
        return UnitValue(np.arctan(self.px / self.pz), "rad")

    @property
    def yp(self):
        return UnitValue(np.arctan(self.py / self.pz), "rad")

    @property
    def p(self):
        return UnitValue(self.cp * self.q_over_c, "kg*m/s")

    @property
    def cp(self):
        return UnitValue(np.sqrt(self.cpx**2 + self.cpy**2 + self.cpz**2), "eV/c")

    @property
    def Brho(self):
        return UnitValue(np.mean(self.pz) / constants.elementary_charge, "T*m")

    @property
    def gamma(self):
        return UnitValue(np.sqrt(1 + (self.cp / self.particle_rest_energy_eV) ** 2), "")

    @property
    def BetaGamma(self):
        return UnitValue(self.cp / self.particle_rest_energy_eV, "")

    @property
    def energy(self):
        return UnitValue(self.gamma * self.particle_rest_energy_eV, "eV")

    @property
    def Ex(self):
        return UnitValue(np.sqrt(self.particle_rest_energy_eV**2 + self.cpx**2), "eV")

    @property
    def Ey(self):
        return UnitValue(np.sqrt(self.particle_rest_energy_eV**2 + self.cpy**2), "eV")

    @property
    def Ez(self):
        return UnitValue(np.sqrt(self.particle_rest_energy_eV**2 + self.cpz**2), "eV")

    @property
    def Bx(self):
        return UnitValue(self.cpx / self.energy, "")

    @property
    def By(self):
        return UnitValue(self.cpy / self.energy, "")

    @property
    def Bz(self):
        return UnitValue(self.cpz / self.energy, "")

    @property
    def Q(self):
        return UnitValue(self.total_charge, "C")

    def set_total_charge(self, q):
        self.total_charge = q
        particle_q = q / (len(self.charge))
        self.charge = np.full(len(self.charge), particle_q)

    # @property
    # def sigma_z(self):
    #     return self.rms(self.Bz*constants.speed_of_light*(self['t'] - np.mean(self['t'])))

    @property
    def kinetic_energy(self):
        return UnitValue(
            np.array(
                (
                    np.sqrt(self.particle_rest_energy**2 + self.cp**2)
                    - self.particle_rest_energy**2
                )
            ),
            "J",
        )

    @property
    def mean_energy(self):
        return UnitValue(np.mean(self.kinetic_energy), "J")

    def computeCorrelations(self, x, y):
        return self.covariance(x, x), self.covariance(x, y), self.covariance(y, y)

    def performTransformation(self, x, xp, beta=False, alpha=False, nEmit=False):
        p = self.cp
        pAve = np.mean(p)
        gamma = np.mean(self.gamma)
        p = [a / pAve - 1 for a in p]
        eta1, etap1, _ = self.twiss.calculate_etax()
        for i, ii in enumerate(x):
            x[i] -= p[i] * eta1
            xp[i] -= p[i] * etap1

        S11, S12, S22 = self.computeCorrelations(x, xp)
        emit = np.sqrt(S11 * S22 - S12**2)
        beta1 = S11 / emit
        alpha1 = -S12 / emit
        beta2 = beta if beta is not False else beta1
        alpha2 = alpha if alpha is not False else alpha1
        R11 = beta2 / np.sqrt(beta1 * beta2)
        R12 = 0
        R21 = (alpha1 - alpha2) / np.sqrt(beta1 * beta2)
        R22 = beta1 / np.sqrt(beta1 * beta2)
        if nEmit is not False:
            factor = np.sqrt(float(nEmit) / (emit * gamma))
            R11 *= factor
            R12 *= factor
            R22 *= factor
            R21 *= factor
        for i, ii in enumerate(x):
            x0 = x[i]
            xp0 = xp[i]
            x[i] = R11 * x0 + R12 * xp0
            xp[i] = R21 * x0 + R22 * xp0
        return x, xp

    def rematchXPlane(self, beta=False, alpha=False, nEmit=False):
        if not (beta is False and alpha is False):
            x, xp = self.performTransformation(self.x, self.xp, beta, alpha, nEmit)
            self.x = x
            self.xp = xp

            cpz = self.cp / np.sqrt(self.xp ** 2 + self.yp**2 + 1)
            cpx = self.xp * cpz
            cpy = self.yp * cpz
            self.px = cpx * self.q_over_c
            self.py = cpy * self.q_over_c
            self.pz = cpz * self.q_over_c

    def rematchYPlane(self, beta=False, alpha=False, nEmit=False):
        if not (beta is False and alpha is False):
            y, yp = self.performTransformation(self.y, self.yp, beta, alpha, nEmit)
            self.y = y
            self.yp = yp

            cpz = self.cp / np.sqrt(self.xp**2 + self.yp ** 2 + 1)
            cpx = self.xp * cpz
            cpy = self.yp * cpz
            self.px = cpx * self.q_over_c
            self.py = cpy * self.q_over_c
            self.pz = cpz * self.q_over_c

    def performTransformationPeakISlice(
        self, xslice, xpslice, x, xp, beta=False, alpha=False, nEmit=False
    ):
        p = self.cp
        pAve = np.mean(p)
        p = [a / pAve - 1 for a in p]
        eta1, etap1, _ = self.twiss.calculate_etax()
        for i, ii in enumerate(x):
            x[i] -= p[i] * eta1
            xp[i] -= p[i] * etap1

        S11, S12, S22 = self.computeCorrelations(xslice, xpslice)
        emit = np.sqrt(S11 * S22 - S12**2)
        beta1 = S11 / emit
        alpha1 = -S12 / emit
        beta2 = beta if beta is not False else beta1
        alpha2 = alpha if alpha is not False else alpha1
        R11 = beta2 / np.sqrt(beta1 * beta2)
        R12 = 0
        R21 = (alpha1 - alpha2) / np.sqrt(beta1 * beta2)
        R22 = beta1 / np.sqrt(beta1 * beta2)
        if nEmit is not False:
            factor = np.sqrt(nEmit / (emit * pAve))
            R11 *= factor
            R12 *= factor
            R22 *= factor
            R21 *= factor
        for i, ii in enumerate(x):
            x0 = x[i]
            xp0 = xp[i]
            x[i] = R11 * x0 + R12 * xp0
            xp[i] = R21 * x0 + R22 * xp0
        return x, xp

    def rematchXPlanePeakISlice(self, beta=False, alpha=False, nEmit=False):
        peakIPosition = self.slice_max_peak_current_slice
        xslice = self.slice_data(self.x)[peakIPosition]
        xpslice = self.slice_data(self.xp)[peakIPosition]
        x, xp = self.performTransformationPeakISlice(
            xslice, xpslice, self.x, self.xp, beta, alpha, nEmit
        )
        self.x = x
        self.xp = xp

        cpz = self.cp / np.sqrt(self.xp ** 2 + self.yp**2 + 1)
        cpx = self.xp * cpz
        cpy = self.yp * cpz
        self.px = cpx * self.q_over_c
        self.py = cpy * self.q_over_c
        self.pz = cpz * self.q_over_c

    def rematchYPlanePeakISlice(self, beta=False, alpha=False, nEmit=False):
        peakIPosition = self.slice_max_peak_current_slice
        yslice = self.slice_data(self.y)[peakIPosition]
        ypslice = self.slice_data(self.yp)[peakIPosition]
        y, yp = self.performTransformationPeakISlice(
            yslice, ypslice, self.y, self.yp, beta, alpha, nEmit
        )
        self.y = y
        self.yp = yp

        cpz = self.cp / np.sqrt(self.xp**2 + self.yp ** 2 + 1)
        cpx = self.xp * cpz
        cpy = self.yp * cpz
        self.px = cpx * self.q_over_c
        self.py = cpy * self.q_over_c
        self.pz = cpz * self.q_over_c
