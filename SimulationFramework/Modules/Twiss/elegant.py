import os
import sdds
import numpy as np
import scipy.constants as constants

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
            read_elegant_twiss_files(self, f, reset=False)
    elif os.path.isfile(filename):
        pre, ext = os.path.splitext(filename)
        read_sdds_file(self, pre+'.flr')
        read_sdds_file(self, pre+'.sig')
        read_sdds_file(self, pre+'.twi')
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
