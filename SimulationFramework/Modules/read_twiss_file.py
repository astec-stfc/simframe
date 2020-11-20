import os, math, h5py, sys
import numpy as np
from scipy import interpolate
import scipy.integrate as integrate
import scipy.constants as constants
import sdds
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(__file__+'/../../'))
import SimulationFramework.Modules.read_gdf_file as rgf
import munch
import glob

class twissData(np.ndarray):

    def __new__(cls=np.ndarray, input_array=[], units=''):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.units = units
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.units = getattr(obj, 'units', '')

class twiss(munch.Munch):

    properties = {
    'z': 'm',
    'x': 'm',
    'y': 'm',
    'z': 'm',
    't': 's',
    'kinetic_energy': 'J',
    'gamma': '',
    'cp': 'J',
    'cp_eV': 'eV',
    'p': 'kg*m/s',
    'enx': 'm-radians',
    'ex': 'm-radians',
    'eny': 'm-radians',
    'ey': 'm-radians',
    'enz': 'eV*s',
    'ez': 'eV*s',
    'beta_x': 'm',
    'gamma_x': '',
    'alpha_x': '',
    'beta_y': 'm',
    'gamma_y': '',
    'alpha_y': '',
    'beta_z': 'm',
    'gamma_z': '',
    'alpha_z': '',
    'sigma_x': 'm',
    'sigma_y': 'm',
    'sigma_z': 'm',
    'sigma_t': 's',
    'sigma_p': 'kg * m/s',
    'sigma_cp': 'J',
    'sigma_cp_eV': 'eV',
    'mux': '2 pi',
    'muy': '2 pi',
    'eta_x': 'm',
    'eta_xp': 'mrad',
    'element_name': '',
    'x': 'm',
    'y': 'm',
    'ecnx': 'm-mrad',
    'ecny': 'm-mrad',
    'eta_x_beam': 'm',
    'eta_xp_beam': 'radians',
    'eta_y_beam': 'm',
    'eta_yp_beam': 'radians',
    'beta_x_beam': 'm',
    'beta_y_beam': 'm',
    'alpha_x_beam': '',
    'alpha_y_beam': '',
    }

    E0 = constants.m_e * constants.speed_of_light**2
    E0_eV = E0 / constants.elementary_charge
    q_over_c = (constants.e / constants.speed_of_light)

    def __init__(self):
        super(twiss, self).__init__()
        self.reset_dicts()
        self.sddsindex = 0
        self.codes = {
            'elegant': self.read_elegant_twiss_files,
            'gpt': self.read_gdf_twiss_files,
            'astra': self.read_astra_twiss_files
        }

    def __getitem__(self, key):
        if key in super(twiss, self).__getitem__('data') and super(twiss, self).__getitem__('data') is not None:
            return self.data.get(key)
        else:
            return super(twiss, self).__getitem__(key)

    def __repr__(self):
        return repr([k for k,v in self.data.items() if len(v.data) > 0])

    def stat(self, key):
        if key in self.properties:
            return self.data[key]

    def units(self, key):
        if key in self.properties:
            return self.properties[key]

    def find_nearest(self, array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return idx-1
        else:
            return idx

    def reset_dicts(self):
        self.data = {}
        self.sddsindex = 0
        for k, v in self.properties.items():
            self.data[k] = twissData(units=v)
        self.elegant = {}

    def sort(self, array='z', reverse=False):
        index = self.data[array].argsort()
        for k in self.properties:
            if len(self.data[k]) > 0:
                if reverse:
                    self.data[k] = self.data[k][index[::-1]]
                else:
                    self.data[k] = self.data[k][index[::1]]

    def append(self, array, data):
        self.data[array] = twissData(np.concatenate([self.data[array], data]), units=self.data[array].units)

    def _which_code(self, name):
        if name.lower() in self.codes.keys():
            return self.codes[name.lower()]
        return None

    def read_twiss_files(self, directory='.', types={'elegant':'.twi', 'GPT': 'emit.gdf','ASTRA': 'Xemit.001'}):
        self.reset_dicts()
        print('Directory:',directory)
        for code, string in types.items():
            twiss_files = glob.glob(directory+'/*'+string)
            print(code, [os.path.basename(t) for t in twiss_files])
            if self._which_code(code) is not None and len(twiss_files) > 0:
                self._which_code(code)(twiss_files, reset=False)


    def read_sdds_file(self, fileName, ascii=False):
        self.sddsindex += 1
        sddsref = sdds.SDDS(self.sddsindex%20)
        sddsref.load(fileName)
        for col in range(len(sddsref.columnName)):
            # print 'col = ', sddsref.columnName[col]
            if len(sddsref.columnData[col]) == 1:
                self.elegant[sddsref.columnName[col]] = np.array(sddsref.columnData[col][0])
            else:
                self.elegant[sddsref.columnName[col]] = np.array(sddsref.columnData[col])
        self.SDDSparameterNames = list()
        for i, param in enumerate(sddsref.parameterName):
            # print 'param = ', param
            self.elegant[param] = sddsref.parameterData[i]
            # if isinstance(self[param][0], (float, long)):
            #     self.SDDSparameterNames.append(param)

    def read_elegant_floor_file(self, filename, offset=[0,0,0], rotation=[0,0,0], reset=True):
        if reset:
            self.reset_dicts()
        self.read_sdds_file(filename)
        self['x'] = twissData([np.round(x+offset[0], decimals = 6) for x in self.elegant['X']], units='m')
        self['y'] = twissData([np.round(y+offset[1], decimals = 6) for y in self.elegant['Y']], units='m')
        self['z'] = twissData([np.round(z+offset[2], decimals = 6) for z in self.elegant['Z']], units='m')
        self['theta'] = twissData([np.round(theta+rotation[0], decimals = 6) for theta in self.elegant['theta']], units='radians')
        self['phi'] = twissData([np.round(phi+rotation[1], decimals = 6) for phi in self.elegant['phi']], units='radians')
        self['psi'] = twissData([np.round(psi+rotation[2], decimals = 6) for psi in self.elegant['psi']], units='radians')
        xyz = list(zip(self['x'], self['y'], self['z']))
        thetaphipsi = list(zip(self['phi'], self['psi'], self['theta']))
        return list(zip(self.elegant['ElementName'], xyz[-1:] + xyz[:-1], xyz, thetaphipsi))[1:]

    def read_elegant_twiss_files(self, filename, startS=0, reset=True):
        if reset:
            self.reset_dicts()
        if isinstance(filename, (list, tuple)):
            for f in filename:
                self.read_elegant_twiss_files(f, reset=False)
        elif os.path.isfile(filename):
            pre, ext = os.path.splitext(filename)
            self.read_sdds_file(pre+'.flr')
            self.read_sdds_file(pre+'.sig')
            self.read_sdds_file(pre+'.twi')
            z = self.elegant['Z']
            self.append('z', z)
            cp = self.elegant['pCentral0'] * self.E0
            self.append('cp', cp)
            self.append('cp_eV', cp / constants.elementary_charge)
            ke = np.array((np.sqrt(self.E0**2 + cp**2) - self.E0**2) / constants.elementary_charge)
            self.append('kinetic_energy', ke)
            gamma = 1 + ke / self.E0_eV
            self.append('gamma', gamma)
            self.append('p', cp * self.q_over_c)
            self.append('enx', self.elegant['enx'])
            self.append('ex', self.elegant['ex'])
            self.append('eny', self.elegant['eny'])
            self.append('ey', self.elegant['ey'])
            self.append('enz', np.zeros(len(self.elegant['Z'])))
            self.append('ez', np.zeros(len(self.elegant['Z'])))
            self.append('beta_x', self.elegant['betax'])
            self.append('alpha_x', self.elegant['alphax'])
            self.append('gamma_x', (1 + self.elegant['alphax']**2) / self.elegant['betax'])
            self.append('beta_y', self.elegant['betay'])
            self.append('alpha_y', self.elegant['alphay'])
            self.append('gamma_y', (1 + self.elegant['alphay']**2) / self.elegant['betay'])
            self.append('beta_z', np.zeros(len(self.elegant['Z'])))
            self.append('gamma_z', np.zeros(len(self.elegant['Z'])))
            self.append('alpha_z', np.zeros(len(self.elegant['Z'])))
            self.append('sigma_x', self.elegant['Sx'])
            self.append('sigma_y', self.elegant['Sy'])
            self.append('sigma_t', self.elegant['St'])
            beta = np.sqrt(1-(gamma**-2))
            # print 'len(z) = ', len(z), '  len(beta) = ', len(beta)
            self.append('t', z / (beta * constants.speed_of_light))
            self.append('sigma_z', self.elegant['St'] * (beta * constants.speed_of_light))
            self.append('sigma_cp', self.elegant['Sdelta'] * cp )
            self.append('sigma_cp_eV', self.elegant['Sdelta'] * cp / constants.elementary_charge)
            # print('elegant = ', (self.elegant['Sdelta'] * cp / constants.elementary_charge)[-1)
            self.append('sigma_p', self.elegant['Sdelta'] )
            self.append('mux', self.elegant['psix'] / (2*constants.pi))
            self.append('muy', self.elegant['psiy'] / (2*constants.pi))
            self.append('eta_x', self.elegant['etax'])
            self.append('eta_xp', self.elegant['etaxp'])
            self.append('element_name', self.elegant['ElementName'])
            ### BEAM parameters
            self.append('ecnx', self.elegant['ecnx'])
            self.append('ecny', self.elegant['ecny'])
            self.append('eta_x_beam', self.elegant['s16']/(self.elegant['s6']**2))
            self.append('eta_xp_beam', self.elegant['s26']/(self.elegant['s6']**2))
            self.append('eta_y_beam', self.elegant['s36']/(self.elegant['s6']**2))
            self.append('eta_yp_beam', self.elegant['s46']/(self.elegant['s6']**2))
            self.append('beta_x_beam', self.elegant['betaxBeam'])
            self.append('beta_y_beam', self.elegant['betayBeam'])
            self.append('alpha_x_beam', self.elegant['alphaxBeam'])
            self.append('alpha_y_beam', self.elegant['alphayBeam'])

    def read_astra_twiss_files(self, filename, reset=True):
        if reset:
            self.reset_dicts()
        if isinstance(filename, (list, tuple)):
            for f in filename:
                self.read_astra_twiss_files(f, reset=False)
        elif os.path.isfile(filename):
            if 'xemit' not in filename.lower():
                filename = filename.replace('Yemit','Xemit').replace('Zemit','Xemit')
            xemit = np.loadtxt(filename, unpack=False) if os.path.isfile(filename) else False
            if 'yemit' not in filename.lower():
                filename = filename.replace('Xemit','Yemit').replace('Zemit','Yemit')
            yemit = np.loadtxt(filename, unpack=False) if os.path.isfile(filename) else False
            if 'zemit' not in filename.lower():
                filename = filename.replace('Xemit','Zemit').replace('Yemit','Zemit')
            zemit = np.loadtxt(filename, unpack=False) if os.path.isfile(filename) else False
            self.interpret_astra_data(xemit, yemit, zemit)

    def read_hdf_summary(self, filename, reset=True):
        if reset:
            self.reset_dicts()
        f = h5py.File(filename, "r")
        xemit = f.get('Xemit')
        yemit = f.get('Yemit')
        zemit = f.get('Zemit')
        for item, params in sorted(xemit.items()):
            self.interpret_astra_data(np.array(xemit.get(item)), np.array(yemit.get(item)), np.array(zemit.get(item)))

    def interpret_astra_data(self, xemit, yemit, zemit):
            z, t, mean_x, rms_x, rms_xp, exn, mean_xxp = np.transpose(xemit)
            z, t, mean_y, rms_y, rms_yp, eyn, mean_yyp = np.transpose(yemit)
            z, t, e_kin, rms_z, rms_e, ezn, mean_zep = np.transpose(zemit)
            e_kin = 1e6 * e_kin
            t = 1e-9 * t
            exn = 1e-6*exn
            eyn = 1e-6*eyn
            rms_x, rms_xp, rms_y, rms_yp, rms_z, rms_e = 1e-3*np.array([rms_x, rms_xp, rms_y, rms_yp, rms_z, rms_e])
            rms_e = 1e6 * rms_e
            self.append('z',z)
            self.append('t',t)
            self.append('kinetic_energy', e_kin)
            gamma = 1 + (e_kin / self.E0_eV)
            self.append('gamma', gamma)
            cp = np.sqrt(e_kin * (2 * self.E0_eV + e_kin)) * constants.elementary_charge
            self.append('cp', cp)
            self.append('cp_eV', cp / constants.elementary_charge)
            p = cp * self.q_over_c
            self.append('p', p)
            self.append('enx', exn)
            ex = exn / gamma
            self.append('ex', ex)
            self.append('eny', eyn)
            ey = eyn / gamma
            self.append('ey', ey)
            self.append('enz', ezn)
            ez = ezn / gamma
            self.append('ez', ez)
            self.append('beta_x', rms_x**2 / ex)
            self.append('gamma_x', rms_xp**2 / ex)
            self.append('alpha_x', (-1 * np.sign(mean_xxp) * rms_x * rms_xp) / ex)
            # self.append('alpha_x', (-1 * mean_xxp * rms_x) / ex)
            self.append('beta_y', rms_y**2 / ey)
            self.append('gamma_y', rms_yp**2 / ey)
            self.append('alpha_y', (-1 * np.sign(mean_yyp) * rms_y * rms_yp) / ey)
            self.append('beta_z', rms_z**2 / ez)
            self.append('gamma_z', rms_e**2 / ez)
            self.append('alpha_z', (-1 * np.sign(mean_zep) * rms_z * rms_e) / ez)
            self.append('sigma_x', rms_x)
            self.append('sigma_y', rms_y)
            self.append('sigma_z', rms_z)
            beta = np.sqrt(1-(gamma**-2))
            self.append('sigma_t', rms_z / (beta * constants.speed_of_light))
            self.append('sigma_p', (rms_e / e_kin))
            self.append('sigma_cp', (rms_e / e_kin) * p)
            self.append('sigma_cp_eV', (rms_e))
            # print('astra = ', (rms_e)[-1)
            self.append('mux', integrate.cumtrapz(x=self['z'], y=1/self['beta_x'], initial=0))
            self.append('muy', integrate.cumtrapz(x=self['z'], y=1/self['beta_y'], initial=0))
            self.append('eta_x', np.zeros(len(z)))
            self.append('eta_xp', np.zeros(len(z)))

            self.append('ecnx', exn)
            self.append('ecny', eyn)
            self.append('element_name', np.zeros(len(z)))
            self.append('eta_x_beam', np.zeros(len(z)))
            self.append('eta_xp_beam', np.zeros(len(z)))
            self.append('eta_y_beam', np.zeros(len(z)))
            self.append('eta_yp_beam', np.zeros(len(z)))
            self.append('beta_x_beam', rms_x**2 / ex)
            self.append('beta_y_beam', rms_y**2 / ey)
            self.append('alpha_x_beam', (-1 * np.sign(mean_xxp) * rms_x * rms_xp) / ex)
            self.append('alpha_y_beam', (-1 * np.sign(mean_yyp) * rms_y * rms_yp) / ey)

    def read_gdf_beam_file_object(self, file):
        if isinstance(file, (str)):
            gdfbeam = rgf.read_gdf_file(file)
        elif isinstance(file, (rgf.read_gdf_file)):
            gdfbeam = file
        else:
            raise Exception('file is not str or gdf object!')
        return gdfbeam

    def read_gdf_twiss_files(self, filename=None, gdfbeam=None, reset=True):
        if reset:
            self.reset_dicts()
        if isinstance(filename, (list, tuple)):
            for f in filename:
                self.read_gdf_twiss_files(file=f, reset=False)
        if os.path.isfile(filename):
            if gdfbeam is None and not filename is None:
                print('GDF filename = ', filename)
                gdfbeam = self.read_gdf_beam_file_object(filename)
                self.gdfbeam = gdfbeam
            elif gdfbeam is None and file is None:
                return None

            gdfbeamdata = gdfbeam.get_grab(0)
            # self.beam['code'] = "GPT"
            # 'avgx','avgy','avgz','stdx','stdBx','stdy','stdBy','stdz','stdt','nemixrms','nemiyrms','nemizrms','numpar','nemirrms','avgG','avgp','stdG','avgt','avgBx','avgBy','avgBz','CSalphax','CSalphay','CSbetax','CSbetay','avgfBx','avgfEx','avgfBy','avgfEy','avgfBz','avgfEz'
            # self.beam['x'] = gdfbeamdata.x
            if hasattr(gdfbeamdata, 'avgz'):
                self.append('z', gdfbeamdata.avgz)
            elif hasattr(gdfbeamdata, 'position'):
                self.append('z', gdfbeamdata.position)
            cp = self.E0 * np.sqrt(gdfbeamdata.avgG**2 - 1)
            self.append('cp', cp)
            self.append('cp_eV', cp / constants.elementary_charge)
            ke = np.array((np.sqrt(self.E0**2 + cp**2) - self.E0**2) / constants.elementary_charge)
            self.append('kinetic_energy', ke)
            gamma = 1 + ke / self.E0_eV
            self.append('gamma', gamma)
            self.append('p', cp * self.q_over_c)
            self.append('enx', gdfbeamdata.nemixrms)
            self.append('ex', gdfbeamdata.nemixrms / gdfbeamdata.avgG)
            self.append('eny', gdfbeamdata.nemiyrms)
            self.append('ey', gdfbeamdata.nemiyrms / gdfbeamdata.avgG)
            self.append('enz', gdfbeamdata.nemizrms)
            self.append('ez', gdfbeamdata.nemizrms / gdfbeamdata.avgG)
            self.append('beta_x', gdfbeamdata.CSbetax)
            self.append('alpha_x', gdfbeamdata.CSalphax)
            self.append('gamma_x', (1 + gdfbeamdata.CSalphax**2) / gdfbeamdata.CSbetax)
            self.append('beta_y', gdfbeamdata.CSbetay)
            self.append('alpha_y', gdfbeamdata.CSalphay)
            self.append('gamma_y', (1 + gdfbeamdata.CSalphax**2) / gdfbeamdata.CSbetay)
            self.append('beta_z', np.zeros(len(gdfbeamdata.stdx)))
            self.append('gamma_z', np.zeros(len(gdfbeamdata.stdx)))
            self.append('alpha_z', np.zeros(len(gdfbeamdata.stdx)))
            self.append('sigma_x', gdfbeamdata.stdx)
            self.append('sigma_y', gdfbeamdata.stdy)
            beta = np.sqrt(1-(gamma**-2))
            if hasattr(gdfbeamdata, 'stdt'):
                self.append('sigma_t', gdfbeamdata.stdt)
            else:
                self.append('sigma_t', gdfbeamdata.stdz / (beta * constants.speed_of_light))
            if hasattr(gdfbeamdata, 'avgt'):
                self.append('t', gdfbeamdata.avgt)
            else:
                self.append('t', gdfbeamdata.time)
            self.append('sigma_z', gdfbeamdata.stdz)
            self.append('sigma_cp', (gdfbeamdata.stdG / gdfbeamdata.avgG) * cp)
            self.append('sigma_cp_eV', (gdfbeamdata.stdG / gdfbeamdata.avgG) * cp / constants.elementary_charge)
            self.append('sigma_p', (gdfbeamdata.stdG / gdfbeamdata.avgG) * cp / constants.speed_of_light)
            self.append('mux', np.zeros(len(gdfbeamdata.stdx)))
            self.append('muy', np.zeros(len(gdfbeamdata.stdx)))
            self.append('eta_x', np.zeros(len(gdfbeamdata.stdx)))
            self.append('eta_xp', np.zeros(len(gdfbeamdata.stdx)))
            self.append('element_name', np.zeros(len(gdfbeamdata.stdx)))
            ### BEAM parameters
            self.append('ecnx', np.zeros(len(gdfbeamdata.stdx)))
            self.append('ecny', np.zeros(len(gdfbeamdata.stdx)))
            self.append('eta_x_beam', np.zeros(len(gdfbeamdata.stdx)))
            self.append('eta_xp_beam', np.zeros(len(gdfbeamdata.stdx)))
            self.append('eta_y_beam', np.zeros(len(gdfbeamdata.stdx)))
            self.append('eta_yp_beam', np.zeros(len(gdfbeamdata.stdx)))
            self.append('beta_x_beam', np.zeros(len(gdfbeamdata.stdx)))
            self.append('beta_y_beam', np.zeros(len(gdfbeamdata.stdx)))
            self.append('alpha_x_beam', np.zeros(len(gdfbeamdata.stdx)))
            self.append('alpha_y_beam', np.zeros(len(gdfbeamdata.stdx)))

    def interpolate(self, z=None, value='z', index='z'):
        f = interpolate.interp1d(self[index], self[value], kind='linear', fill_value="extrapolate")
        if z is None:
            return f
        else:
            if z > max(self[index]):
                return 10**6
            else:
                return float(f(z))

    def extract_values(self, array, start, end):
        startidx = self.find_nearest(self['z'], start)
        endidx = self.find_nearest(self['z'], end) + 1
        return self[array][startidx:endidx]

    def get_parameter_at_z(self, param, z):
        if z in self['z']:
            idx = list(self['z']).index(z)
            return self[param][idx]
        else:
            return self.interpolate(z=z, value=param, index='z')

    def covariance(self, u, up):
        u2 = u - np.mean(u)
        up2 = up - np.mean(up)
        return np.mean(u2*up2) - np.mean(u2)*np.mean(up2)

    # def write_HDF5_beam_file(self, filename, sourcefilename=None):
    #     with h5py.File(filename, "w") as f:
    #         inputgrp = f.create_group("Parameters")
    #         if sourcefilename is not None:
    #             inputgrp['Source'] = sourcefilename
    #         inputgrp['code'] = self.beam['code']
    #         twissgrp = f.create_group("twiss")
    #         array = np.array([self.s, self.t, self.sigma_x, self.sigma_y, self.sigma_z, self.sigma_p, self.sigma_t, self.beta_x, self.alpha_x, self.gamma_x,
    #                   self.beta_y, self.alpha_y, self.gamma_y, self.beta_z, self.alpha_z, self.gamma_z, self.mux, self.muy,
    #                   self.ex, self.enx, self.ey, self.eny]).transpose()
    #         beamgrp['columns'] = ("s","t","Sx","Sy","Sz","Sp","St","betax","alphax","gammax","betay","alphay","gammay","betaz","alphaz","gammaz","mux","muy")
    #         beamgrp['units'] = ("m","s","m","m","m","eV/c","s","m","","","m","","","m","","","","")
    #         beamgrp.create_dataset("twiss", data=array)
    #
    # def read_HDF5_beam_file(self, filename, local=False):
    #     self.reset_dicts()
    #     with h5py.File(filename, "r") as h5file:
    #         if h5file.get('beam/reference_particle') is not None:
    #             self.beam['reference_particle'] = np.array(h5file.get('beam/reference_particle'))
    #         if h5file.get('beam/longitudinal_reference') is not None:
    #             self.beam['longitudinal_reference'] = np.array(h5file.get('beam/longitudinal_reference'))
    #         else:
    #             self.beam['longitudinal_reference'] = 't'
    #         if h5file.get('beam/status') is not None:
    #             self.beam['status'] = np.array(h5file.get('beam/status'))
    #         x, y, z, cpx, cpy, cpz, t, charge = np.array(h5file.get('beam/beam')).transpose()
    #         cp = np.sqrt(cpx**2 + cpy**2 + cpz**2)
    #         self.beam['x'] = x
    #         self.beam['y'] = y
    #         self.beam['z'] = z
    #         # self.beam['cpx'] = cpx
    #         # self.beam['cpy'] = cpy
    #         # self.beam['cpz'] = cpz
    #         self.beam['px'] = cpx * self.q_over_c
    #         self.beam['py'] = cpy * self.q_over_c
    #         self.beam['pz'] = cpz * self.q_over_c
    #         # self.beam['cp'] = cp
    #         # self.beam['p'] = cp * self.q_over_c
    #         # self.beam['xp'] = np.arctan(self.px/self.pz)
    #         # self.beam['yp'] = np.arctan(self.py/self.pz)
    #         self.beam['clock'] = np.full(len(self.x), 0)
    #         # self.beam['gamma'] = np.sqrt(1+(self.cp/self.E0_eV)**2)
    #         # velocity_conversion = 1 / (constants.m_e * self.gamma)
    #         # self.beam['vx'] = velocity_conversion * self.px
    #         # self.beam['vy'] = velocity_conversion * self.py
    #         # self.beam['vz'] = velocity_conversion * self.pz
    #         # self.beam['Bx'] = self.vx / constants.speed_of_light
    #         # self.beam['By'] = self.vy / constants.speed_of_light
    #         # self.beam['Bz'] = self.vz / constants.speed_of_light
    #         self.beam['t'] = t
    #         self.beam['charge'] = charge
    #         self.beam['total_charge'] = np.sum(self.beam['charge'])
    #         startposition = np.array(h5file.get('/Parameters/Starting_Position'))
    #         startposition = startposition if startposition is not None else [0,0,0]
    #         self.beam['starting_position'] = startposition
    #         theta =  np.array(h5file.get('/Parameters/Rotation'))
    #         theta = theta if theta is not None else 0
    #         self.beam['rotation'] = theta
    #         if local == True:
    #             self.rotate_beamXZ(self.beam['rotation'], preOffset=self.beam['starting_position'])

    @property
    def cp_MeV(self):
        return self['cp_eV'] / 1e6
