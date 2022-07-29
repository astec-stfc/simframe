from SimulationFramework.Modules.minimumVolumeEllipse import getMinVolEllipse
import numpy as np
from scipy.stats import gaussian_kde
from functools import partial

class MVE():

    def __init__(self, beam):
        self.beam = beam

    def kde_bw_func(self, bandwidth, x, *args, **kwargs):
        return (bandwidth / x.std(ddof=1))

    def kde_function(self, x, bandwidth=0.2, **kwargs):
        """Kernel Density Estimation with Scipy"""
        # Note that scipy weights its bandwidth by the covariance of the
        # input data.  To make the results comparable to the other methods,
        # we divide the bandwidth by the sample standard deviation here.
        # Taken from https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
        if not hasattr(self, '_kde_x') or not len(x) == len(self._kde_x) or not np.allclose(x, self._kde_x) or not bandwidth == self._kde_bandwidth:
            self._kde_x = x
            self._kde_bandwidth = bandwidth
            bw = partial(self.kde_bw_func, bandwidth, x)
            self._kde_function = gaussian_kde(x, bw_method=bw, **kwargs)
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
        (center, radii, rotation, hullP) = getMinVolEllipse(list(zip(x,xp)), .01)
        emittance = radii[0] * radii[1]
        if p is None:
            return emittance
        else:
            gamma = np.mean(p)/self.E0_eV
            return gamma*emittance

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
    @property
    def slice_density(self):
        if not hasattr(self,'_tbins') or not hasattr(self,'_cpbins'):
            self.bin_time()
        xbins = self.slice_data(self.x)
        volume = self.slice_6D_Volume
        self.sliceProperties['Density'] = np.array([len(x)/v for x, v in zip(xbins, volume)])

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
