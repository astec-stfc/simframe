import numpy as np
from munch import Munch
import scipy.constants as constants
import SimulationFramework.Modules.minimumVolumeEllipse as mve
MVE = mve.EllipsoidTool()

class Particles(Munch):

    properties = {
    'z': 'm',
    'x': 'm',
    'y': 'm',
    'z': 'm',
    't': 's',
    'px': 'kg*m/s',
    'py': 'kg*m/s',
    'pz': 'kg*m/s',
    'p': 'kg*m/s'
    }

    particle_mass = constants.m_e
    E0 = particle_mass * constants.speed_of_light**2
    E0_eV = E0 / constants.elementary_charge
    q_over_c = (constants.elementary_charge / constants.speed_of_light)
    speed_of_light = constants.speed_of_light

    ''' ********************  Statistical Parameters  ************************* '''

    def __init__(self):
        Munch.__init__(self)
        self.twissFunctions = {}
        self.sliceProperties = {}
        self._tbins = []
        self._pbins = []

    def __getitem__(self, key):
        if isinstance(Munch.__getitem__(self, key),(list, tuple)):
            return np.array(Munch.__getitem__(self, key))
        else:
            return Munch.__getitem__(self, key)

    def kde_function(self, x, bandwidth=0.2, **kwargs):
        """Kernel Density Estimation with Scipy"""
        # Note that scipy weights its bandwidth by the covariance of the
        # input data.  To make the results comparable to the other methods,
        # we divide the bandwidth by the sample standard deviation here.
        # Taken from https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
        if not hasattr(self, '_kde_x') or not len(x) == len(self._kde_x) or not np.allclose(x, self._kde_x) or not bandwidth == self._kde_bandwidth:
            self._kde_x = x
            self._kde_bandwidth = bandwidth
            self._kde_function = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
        return self._kde_function

    def PDF(self, x, x_grid, bandwidth=0.2, **kwargs):
        kde = self.kde_function(x, bandwidth, **kwargs)
        return kde.evaluate(x_grid)

    def PDFI(self, x, x_grid, bandwidth=0.2, **kwargs):
        kde = self.kde_function(x, bandwidth, **kwargs)
        vals = kde.evaluate(x_grid)
        f = lambda bin, val: self.Q / len(self.t) * (val / bin)
        return vals#self.charge * vals / (2*abs(x_grid[1] - x_grid[0])) / len(self.t) #[f(x_grid[1] - x_grid[0], v) for v in vals]

    def CDF(self, x, x_grid, bandwidth=0.2, **kwargs):
        kde = self.kde_function(x, bandwidth, **kwargs)
        cdf = np.vectorize(lambda e: kde.integrate_box_1d(x_grid[0], e))
        return cdf(x_grid)

    def FWHM(self, X, Y, frac=0.5):
        frac = 1.0/frac if frac > 1 else frac
        d = Y - (max(Y) * frac)
        indexes = np.where(d > 0)[0]
        return abs(X[indexes][-1] - X[indexes][0]), indexes

    def covariance(self, u, up):
        # u2 = u - np.mean(u)
        # up2 = up - np.mean(up)
        # return np.mean(u2*up2) - np.mean(u2)*np.mean(up2)
        return float(np.cov([u,up])[0,1])

    def emittance_calc(self, x, xp, p=None):
        cov_x = self.covariance(x, x)
        cov_xp = self.covariance(xp, xp)
        cov_x_xp = self.covariance(x, xp)
        emittance = np.sqrt(cov_x * cov_xp - cov_x_xp**2) if (cov_x * cov_xp - cov_x_xp**2) > 0 else 0
        if p is None:
            return emittance
        else:
            gamma = np.mean(p)/self.E0_eV
            return gamma*emittance

    @property
    def volume(self):
        return self.volume6D(self.x, self.y, self.z-np.mean(self.z), self.cpx/self.cpz, self.cpy/self.cpz, ((self.cpz/np.mean(self.cp)) - 1))

    @property
    def density(self):
        return len(self.x) / self.volume

    def volume6D(self, x, y, t, xp, yp, cp):
        if len(x) < 10:
            return 1e6
        else:
            beam = list(zip(x, y, t, xp, yp, cp))
            return ConvexHull(beam, qhull_options='QJ').volume

    def mve_emittance(self, x, xp, p=None):
        (center, radii, rotation, hullP) = MVE.getMinVolEllipse(list(zip(x,xp)), .01)
        emittance = radii[0] * radii[1]
        if p is None:
            return emittance
        else:
            gamma = np.mean(p)/self.E0_eV
            return gamma*emittance

    def normalized_emittance(self, plane='x', corrected=False):
        if corrected:
            return self.emittance_calc(getattr(self, plane+'c'), getattr(self, plane+'pc'), self.cp)
        else:
            return self.emittance_calc(getattr(self, plane), getattr(self, plane+'p'), self.cp)

    def emittance(self, plane='x', corrected=False):
        if corrected:
            return self.emittance_calc(getattr(self, plane+'c'), getattr(self, plane+'pc'), None)
        else:
            return self.emittance_calc(getattr(self, plane), getattr(self, plane+'p'), None)

    def twiss(self, plane='x', corrected=False):
        functions = ['beta_', 'alpha_', 'gamma_']
        for t in functions:
            if corrected:
                    getattr(self, t+plane+'_corrected')
            else:
                getattr(self, t+plane)
        if corrected:
            return [self.twissFunctions[t+plane+'_corrected'] for t in functions]
        else:
            return [self.twissFunctions[t+plane] for t in functions]

    @property
    def normalized_horizontal_emittance(self):
        return self.emittance_calc(self.x, self.xp, self.cp)
    @property
    def normalized_vertical_emittance(self):
        return self.emittance_calc(self.y, self.yp, self.cp)
    @property
    def horizontal_emittance(self):
        return self.emittance_calc(self.x, self.xp)
    @property
    def vertical_emittance(self):
        return self.emittance_calc(self.y, self.yp)
    @property
    def normalized_mve_horizontal_emittance(self):
        return self.mve_emittance(self.x, self.xp, self.cp)
    @property
    def normalized_mve_vertical_emittance(self):
        return self.mve_emittance(self.y, self.yp, self.cp)
    @property
    def horizontal_mve_emittance(self):
        return self.mve_emittance(self.x, self.xp)
    @property
    def vertical_mve_emittance(self):
        return self.mve_emittance(self.y, self.yp)
    @property
    def horizontal_emittance_90(self):
        emit = self.horizontal_emittance
        alpha = self.alpha_x
        beta = self.beta_x
        gamma = self.gamma_x
        emiti = gamma * self.x**2 + 2 * alpha * self.x * self.xp + beta * self.xp * self.xp
        return sorted(emiti)[int(0.9*len(emiti)-0.5)]
    @property
    def normalized_horizontal_emittance_90(self):
        emit = self.horizontal_emittance_90
        return np.mean(self.cp)/self.E0_eV * emit
    @property
    def vertical_emittance_90(self):
        emit = self.vertical_emittance
        alpha = self.alpha_y
        beta = self.beta_y
        gamma = self.gamma_y
        emiti = gamma * self.y**2 + 2 * alpha * self.y * self.yp + beta * self.yp * self.yp
        return sorted(emiti)[int(0.9*len(emiti)-0.5)]
    @property
    def normalized_vertical_emittance_90(self):
        emit = self.vertical_emittance_90
        return np.mean(self.cp)/self.E0_eV * emit

    @property
    def beta_x(self):
        self.twissFunctions['beta_x'] = self.covariance(self.x,self.x) / self.horizontal_emittance
        return self.twissFunctions['beta_x']
    @property
    def alpha_x(self):
        self.twissFunctions['alpha_x'] = -1*self.covariance(self.x,self.xp) / self.horizontal_emittance
        return self.twissFunctions['alpha_x']
    @property
    def gamma_x(self):
        self.twissFunctions['gamma_x'] = self.covariance(self.xp,self.xp) / self.horizontal_emittance
        return self.twissFunctions['gamma_x']
    @property
    def beta_y(self):
        self.twissFunctions['beta_y'] = self.covariance(self.y,self.y) / self.vertical_emittance
        return self.twissFunctions['beta_y']
    @property
    def alpha_y(self):
        self.twissFunctions['alpha_y'] = -1*self.covariance(self.y,self.yp) / self.vertical_emittance
        return self.twissFunctions['alpha_y']
    @property
    def gamma_y(self):
        self.twissFunctions['gamma_y'] = self.covariance(self.yp,self.yp) / self.vertical_emittance
        return self.twissFunctions['gamma_y']
    @property
    def twiss_analysis(self):
        return self.horizontal_emittance, self.alpha_x, self.beta_x, self.gamma_x, self.vertical_emittance, self.alpha_y, self.beta_y, self.gamma_y

    def eta_correlation(self, u):
        return self.covariance(u,self.p) / self.covariance(self.p, self.p)

    def eta_corrected(self, u):
        return u - self.eta_correlation(u)*self.p

    @property
    def xc(self):
        return self.eta_corrected(self.x)
    @property
    def xpc(self):
        return self.eta_corrected(self.xp)
    @property
    def yc(self):
        return self.eta_corrected(self.y)
    @property
    def ypc(self):
        return self.eta_corrected(self.yp)

    @property
    def horizontal_emittance_corrected(self):
        xc = self.eta_corrected(self.x)
        xpc = self.eta_corrected(self.xp)
        return self.emittance_calc(xc, xpc)
    @property
    def vertical_emittance_corrected(self):
        yc = self.eta_corrected(self.y)
        ypc = self.eta_corrected(self.yp)
        return self.emittance_calc(yc, ypc)
    @property
    def beta_x_corrected(self):
        xc = self.eta_corrected(self.x)
        self.twissFunctions['beta_x_corrected'] = self.covariance(xc, xc) / self.horizontal_emittance_corrected
        return self.twissFunctions['beta_x_corrected']
    @property
    def alpha_x_corrected(self):
        xc = self.eta_corrected(self.x)
        xpc = self.eta_corrected(self.xp)
        self.twissFunctions['alpha_x_corrected'] = -1*self.covariance(xc, xpc) / self.horizontal_emittance_corrected
        return self.twissFunctions['alpha_x_corrected']
    @property
    def gamma_x_corrected(self):
        xpc = self.eta_corrected(self.xp)
        self.twissFunctions['gamma_x_corrected'] = self.covariance(xpc, xpc) / self.horizontal_emittance_corrected
        return self.twissFunctions['gamma_x_corrected']
    @property
    def beta_y_corrected(self):
        yc = self.eta_corrected(self.y)
        self.twissFunctions['beta_y_corrected'] = self.covariance(yc,yc) / self.vertical_emittance_corrected
        return self.twissFunctions['beta_y_corrected']
    @property
    def alpha_y_corrected(self):
        yc = self.eta_corrected(self.y)
        ypc = self.eta_corrected(self.yp)
        self.twissFunctions['alpha_y_corrected'] = -1*self.covariance(yc, ypc) / self.vertical_emittance_corrected
        return self.twissFunctions['alpha_y_corrected']
    @property
    def gamma_y_corrected(self):
        ypc = self.eta_corrected(self.yp)
        self.twissFunctions['gamma_y_corrected'] = self.covariance(ypc,ypc) / self.vertical_emittance_corrected
        return self.twissFunctions['gamma_y_corrected']
    @property
    def twiss_analysis_corrected(self):
        return  self.horizontal_emittance_corrected, self.alpha_x_corrected, self.beta_x_corrected, self.gamma_x_corrected, \
                self.vertical_emittance_corrected, self.alpha_y_corrected, self.beta_y_corrected, self.gamma_y_corrected

    @property
    def slice_length(self):
        return self._slicelength

    @slice_length.setter
    def slice_length(self, slicelength):
        self._slicelength = slicelength

    @property
    def slices(self):
        return self._slices

    @slices.setter
    def slices(self, slices):
        self.set_slices(slices)

    def set_slices(self, slices):
        twidth = (max(self.t) - min(self.t))
        # print('twidth = ', twidth)
        if twidth == 0:
            t = self.z / (-1 * self.Bz * constants.speed_of_light)
            twidth = (max(t) - min(t))
        if slices == 0:
            slices = int(twidth / 0.1e-12)
        # print('slices = ', slices)
        self._slices = slices
        self._slicelength = twidth / slices
        # if not hasattr(self,'_slicelength'):
        #     print('no slicelength even though I just set it')

    def bin_time(self):
        if not hasattr(self,'slice'):
            self.sliceProperties = {}
        if not self.slice_length > 0:
            # print('no slicelength', self.slice_length)
            self._slice_length = 0
            # print("Assuming slice length is 100 fs")
        twidth = (max(self.t) - min(self.t))
        if twidth == 0:
            t = self.z / (-1 * self.Bz * constants.speed_of_light)
            twidth = (max(t) - min(t))
        else:
            t = self.t
        if not self.slice_length > 0.0:
            self.slice_length = twidth / 20.0
        # print('slicelength =', self.slice_length)
        nbins = max([1,int(np.ceil(twidth / self.slice_length))])+2
        self._hist, binst =  np.histogram(t, bins=nbins, range=(min(t)-self.slice_length, max(t)+self.slice_length))
        self.sliceProperties['t_Bins'] = binst
        self._t_binned = np.digitize(t, self.sliceProperties['t_Bins'])
        self._tfbins = [[self._t_binned == i] for i in range(1, len(binst))]
        self._tbins = [np.array(self.t)[tuple(tbin)] for tbin in self._tfbins]
        self._cpbins = [np.array(self.cp)[tuple(tbin)] for tbin in self._tfbins]

    def bin_momentum(self, width=10**6):
        if not hasattr(self,'slice'):
            self.slice = {}
        pwidth = (max(self.cp) - min(self.cp))
        if width is None:
            self.slice_length_cp = pwidth / self.slices
        else:
            self.slice_length_cp = width
        nbins = max([1,int(np.ceil(pwidth / self.slice_length_cp))])+2
        self._hist, binst =  np.histogram(self.cp, bins=nbins, range=(min(self.cp)-self.slice_length_cp, max(self.cp)+self.slice_length_cp))
        self.sliceProperties['cp_Bins'] = binst
        self._cp_binned = np.digitize(self.cp, self.sliceProperties['cp_Bins'])
        self._tfbins = [np.array([self._cp_binned == i]) for i in range(1, len(binst))]
        self._cpbins = [self.cp[tuple(cpbin)] for cpbin in self._tfbins]
        self._tbins = [self.t[tuple(cpbin)] for cpbin in self._tfbins]

    @property
    def slice_bins(self):
        if not hasattr(self,'slice'):
            self.bin_time()
        bins = self.sliceProperties['t_Bins']
        return (bins[:-1] + bins[1:]) / 2
        # return [t.mean() for t in ]
    @property
    def slice_cpbins(self):
        if not hasattr(self,'slice'):
            self.bin_momentum()
        bins = self.sliceProperties['cp_Bins']
        return (bins[:-1] + bins[1:]) / 2
        # return [t.mean() for t in ]
    @property
    def slice_momentum(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        self.sliceProperties['Momentum'] = np.array([cpbin.mean() if len(cpbin) > 0 else 0 for cpbin in self._cpbins ])
        return self.sliceProperties['Momentum']
    @property
    def slice_momentum_spread(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        self.sliceProperties['Momentum_Spread'] = np.array([cpbin.std() if len(cpbin) > 0 else 0 for cpbin in self._cpbins])
        return self.sliceProperties['Momentum_Spread']
    @property
    def slice_relative_momentum_spread(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        self.sliceProperties['Relative_Momentum_Spread'] = np.array([100*cpbin.std()/cpbin.mean() if len(cpbin) > 0 else 0 for cpbin in self._cpbins])
        return self.sliceProperties['Relative_Momentum_Spread']

    def slice_data(self, data):
        return [data[tuple(tbin)] for tbin in self._tfbins]

    def emitbins(self, x, y):
        xbins = self.slice_data(x)
        ybins = self.slice_data(y)
        return list(zip(*[xbins, ybins, self._cpbins]))

    @property
    def slice_6D_Volume(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        xbins = self.slice_data(self.x)
        ybins = self.slice_data(self.y)
        zbins = self.slice_data(self.z-np.mean(self.z))
        pxbins = self.slice_data(self.cpx/self.cpz)
        pybins = self.slice_data(self.cpy/self.cpz)
        pzbins = self.slice_data(((self.cpz/np.mean(self.cp)) - 1))
        emitbins = list(zip(xbins, ybins, zbins, pxbins, pybins, pzbins))
        self.sliceProperties['6D_Volume'] = np.array([self.volume6D(*a) for a in emitbins])
        return self.sliceProperties['6D_Volume']
    @property
    def slice_density(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        xbins = self.slice_data(self.x)
        volume = self.slice_6D_Volume
        self.sliceProperties['Density'] = np.array([len(x)/v for x, v in zip(xbins, volume)])
        return self.sliceProperties['Density']
    @property
    def slice_horizontal_emittance(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        emitbins = self.emitbins(self.x, self.xp)
        self.sliceProperties['Horizontal_Emittance'] = np.array([self.emittance_calc(xbin, xpbin) if len(cpbin) > 0 else 0 for xbin, xpbin, cpbin in emitbins])
        return self.sliceProperties['Horizontal_Emittance']
    @property
    def slice_vertical_emittance(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        emitbins = self.emitbins(self.y, self.yp)
        self.sliceProperties['Vertical_Emittance'] = np.array([self.emittance_calc(ybin, ypbin) if len(cpbin) > 0 else 0 for ybin, ypbin, cpbin in emitbins])
        return self.sliceProperties['Vertical_Emittance']
    @property
    def slice_normalized_horizontal_emittance(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        emitbins = self.emitbins(self.x, self.xp)
        self.sliceProperties['Normalized_Horizontal_Emittance'] = np.array([self.emittance_calc(xbin, xpbin, cpbin) if len(cpbin) > 0 else 0 for xbin, xpbin, cpbin in emitbins])
        return self.sliceProperties['Normalized_Horizontal_Emittance']
    @property
    def slice_normalized_vertical_emittance(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        emitbins = self.emitbins(self.y, self.yp)
        self.sliceProperties['Normalized_Vertical_Emittance'] = np.array([self.emittance_calc(ybin, ypbin, cpbin) if len(cpbin) > 0 else 0 for ybin, ypbin, cpbin in emitbins])
        return self.sliceProperties['Normalized_Vertical_Emittance']
    @property
    def slice_normalized_mve_horizontal_emittance(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        emitbins = self.emitbins(self.x, self.xp)
        self.sliceProperties['Normalized_mve_Horizontal_Emittance'] = np.array([self.mve_emittance(xbin, xpbin, cpbin) if len(cpbin) > 0 else 0 for xbin, xpbin, cpbin in emitbins])
        return self.sliceProperties['Normalized_mve_Horizontal_Emittance']
    @property
    def slice_normalized_mve_vertical_emittance(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        emitbins = self.emitbins(self.y, self.yp)
        self.sliceProperties['Normalized_mve_Vertical_Emittance'] = np.array([self.mve_emittance(ybin, ypbin, cpbin) if len(cpbin) > 0 else 0 for ybin, ypbin, cpbin in emitbins])
        return self.sliceProperties['Normalized_mve_Vertical_Emittance']
    @property
    def slice_peak_current(self):
        if not hasattr(self,'_hist'):
            self.bin_time()
        f = lambda bin: self.Q / len(self.t) * (len(bin) / (max(bin) - min(bin))) if len(bin) > 1 else 0
        # f = lambda bin: len(bin) if len(bin) > 1 else 0
        self.sliceProperties['Peak_Current'] = np.array([f(bin) for bin in self._tbins])
        return abs(self.sliceProperties['Peak_Current'])
    @property
    def slice_max_peak_current_slice(self):
        peakI = self.slice_peak_current
        self.sliceProperties['Max_Peak_Current_Slice'] = list(abs(peakI)).index(max(abs(peakI)))
        return self.sliceProperties['Max_Peak_Current_Slice']

    @property
    def slice_beta_x(self):
        xbins = self.slice_data(self['x'])
        exbins =  self.slice_horizontal_emittance
        emitbins = list(zip(xbins, exbins))
        self.sliceProperties['slice_beta_x'] = np.array([self.covariance(x, x)/ex if ex > 0 else 0 for x, ex in emitbins])
        return self.sliceProperties['slice_beta_x']
    @property
    def slice_alpha_x(self):
        xbins = self.slice_data(self.x)
        xpbins = self.slice_data(self.xp)
        exbins =  self.slice_horizontal_emittance
        emitbins = list(zip(xbins, xpbins, exbins))
        self.sliceProperties['slice_alpha_x'] = np.array([-1*self.covariance(x, xp)/ex if ex > 0 else 0 for x, xp, ex in emitbins])
        return self.sliceProperties['slice_alpha_x']
    @property
    def slice_gamma_x(self):
        self.sliceProperties['slice_gamma_x'] = self.covariance(self.xp,self.xp) / self.horizontal_emittance
        return self.sliceProperties['slice_gamma_x']
    @property
    def slice_beta_y(self):
        ybins = self.slice_data(self['y'])
        eybins =  self.slice_vertical_emittance
        emitbins = list(zip(ybins, eybins))
        self.sliceProperties['slice_beta_y'] = np.array([self.covariance(y, y)/ey if ey > 0 else 0 for y, ey in emitbins])
        return self.sliceProperties['slice_beta_y']
    @property
    def slice_alpha_y(self):
        ybins = self.slice_data(self.y)
        ypbins = self.slice_data(self.yp)
        eybins =  self.slice_vertical_emittance
        emitbins = list(zip(ybins, ypbins, eybins))
        self.sliceProperties['slice_alpha_y'] = np.array([-1*self.covariance(y,yp)/ey if ey > 0 else 0 for y, yp, ey in emitbins])
        return self.sliceProperties['slice_alpha_y']
    @property
    def slice_gamma_y(self):
        self.sliceProperties['slice_gamma_y'] = self.covariance(self.yp,self.yp) / self.vertical_emittance
        return self.sliceProperties['slice_gamma_y']

    def sliceAnalysis(self, density=False):
        self.slice = {}
        self.bin_time()
        peakIPosition = self.slice_max_peak_current_slice
        slice_density = self.slice_density[peakIPosition] if density else 0
        return self.slice_peak_current[peakIPosition], \
            np.std(self.slice_peak_current), \
            self.slice_relative_momentum_spread[peakIPosition], \
            self.slice_normalized_horizontal_emittance[peakIPosition], \
            self.slice_normalized_vertical_emittance[peakIPosition], \
            self.slice_momentum[peakIPosition], \
            self.slice_density[peakIPosition],

    def mvesliceAnalysis(self):
        self.slice = {}
        self.bin_time()
        peakIPosition = self.slice_max_peak_current_slice
        return self.slice_peak_current[peakIPosition], \
            np.std(self.slice_peak_current), \
            self.slice_relative_momentum_spread[peakIPosition], \
            self.slice_normalized_mve_horizontal_emittance[peakIPosition], \
            self.slice_normalized_mve_vertical_emittance[peakIPosition], \
            self.slice_momentum[peakIPosition], \
            self.slice_density[peakIPosition],

    @property
    def chirp(self):
        self.bin_time()
        slice_current_centroid_indices = []
        slice_momentum_centroid = []
        peakIPosition = self.slice_max_peak_current_slice
        peakI = self.slice_peak_current[peakIPosition]
        slicemomentum = self.slice_momentum
        for index, slice_current in enumerate(self.slice_peak_current):
            if abs(peakI - slice_current) < (peakI * 0.75):
                slice_current_centroid_indices.append(index)
        for index in slice_current_centroid_indices:
            slice_momentum_centroid.append(slicemomentum[index])
        chirp = (1e-18 * (slice_momentum_centroid[-1] - slice_momentum_centroid[0]) / (len(slice_momentum_centroid) * self.slice_length))
        return chirp

    @property
    def x(self):
        return self['x']
    @property
    def y(self):
        return self['y']
    @property
    def z(self):
        return self['z']
    @property
    def zn(self):
        return self['z']-np.mean(self['z'])
    @property
    def px(self):
        return self['px']
    @property
    def py(self):
        return self['py']
    @property
    def pz(self):
        return self['pz']
    @property
    def cpx(self):
        return self['px'] / self.q_over_c
    @property
    def cpy(self):
        return self['py'] / self.q_over_c
    @property
    def cpz(self):
        return self['pz'] / self.q_over_c
    @property
    def xp(self):
        return np.arctan(self.px/self.pz)
    @property
    def yp(self):
        return np.arctan(self.py/self.pz)
    @property
    def t(self):
        return self['t']
    @property
    def p(self):
        return self.cp * self.q_over_c
    @property
    def cp(self):
        return np.sqrt(self.cpx**2 + self.cpy**2 + self.cpz**2)
    @property
    def Brho(self):
        return np.mean(self.p) / constants.elementary_charge
    @property
    def gamma(self):
        return np.sqrt(1+(self.cp/self.E0_eV)**2)
    @property
    def BetaGamma(self):
        return self.cp/self.E0_eV
    @property
    def vx(self):
        velocity_conversion = 1 / (constants.m_e * self.gamma)
        return velocity_conversion * self.px
    @property
    def vy(self):
        velocity_conversion = 1 / (constants.m_e * self.gamma)
        return velocity_conversion * self.py
    @property
    def vz(self):
        velocity_conversion = 1 / (constants.m_e * self.gamma)
        return velocity_conversion * self.pz
    @property
    def Bx(self):
        return self.vx / constants.speed_of_light
    @property
    def By(self):
        return self.vy / constants.speed_of_light
    @property
    def Bz(self):
        return self.vz / constants.speed_of_light
    @property
    def Q(self):
        return self['total_charge']
    @property
    def sigma_z(self):
        return self.rms(self.Bz*constants.speed_of_light*(self['t'] - np.mean(self['t'])))
    @property
    def momentum_spread(self):
        return self.cp.std()/np.mean(self.cp)
    @property
    def linear_chirp_z(self):
        return -1*self.rms(self.Bz*constants.speed_of_light*self.t)/self.momentum_spread/100

    @property
    def kinetic_energy(self):
        return np.array((np.sqrt(self.E0**2 + self.cp**2) - self.E0**2))

    @property
    def mean_energy(self):
        return np.mean(self.kinetic_energy)

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

    @property
    def eta_x(self):
        # print('etax = ', self.calculate_etax()[0])
        return self.calculate_etax()[0]

    @property
    def eta_xp(self):
        return self.calculate_etax()[1]

    def calculate_etax(self):
        p = self.cpz
        pAve = np.mean(p)
        p = [(a / pAve) - 1 for a in p]
        S16, S66 = self.covariance(self.x, p), self.covariance(p, p)
        eta1 = S16/S66 if S66 else 0
        S26 = self.covariance(self.xp, p)
        etap1 = S26/S66 if S66 else 0
        return eta1, etap1, np.mean(self.t)

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


    @property
    def Sx(self):
        return np.sqrt(self.covariance(self.x,self.x))
    @property
    def Sy(self):
        return np.sqrt(self.covariance(self.y,self.y))
