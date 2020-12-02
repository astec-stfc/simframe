import os
import sdds
import numpy as np
from .. import constants

def read_sdds_file(self, fileName, ascii=False):
    self.sddsindex += 1
    sddsref = sdds.SDDS(self.sddsindex%20)
    sddsref.load(fileName)
    for col in range(len(sddsref.columnName)):
        # print 'col = ', sddsref.columnName[col]
        if len(sddsref.columnData[col]) == 1:
            self.elegantTwiss[sddsref.columnName[col]] = np.array(sddsref.columnData[col][0])
        else:
            self.elegantTwiss[sddsref.columnName[col]] = np.array(sddsref.columnData[col])
    self.SDDSparameterNames = list()
    for i, param in enumerate(sddsref.parameterName):
        # print 'param = ', param
        self.elegantTwiss[param] = sddsref.parameterData[i]
        # if isinstance(self[param][0], (float, long)):
        #     self.SDDSparameterNames.append(param)

def read_elegant_floor_file(self, filename, offset=[0,0,0], rotation=[0,0,0], reset=True):
    if reset:
        self.reset_dicts()
    self.read_sdds_file(filename)
    self['x'] = twissData([np.round(x+offset[0], decimals = 6) for x in self.elegantTwiss['X']], units='m')
    self['y'] = twissData([np.round(y+offset[1], decimals = 6) for y in self.elegantTwiss['Y']], units='m')
    self['z'] = twissData([np.round(z+offset[2], decimals = 6) for z in self.elegantTwiss['Z']], units='m')
    self['theta'] = twissData([np.round(theta+rotation[0], decimals = 6) for theta in self.elegantTwiss['theta']], units='radians')
    self['phi'] = twissData([np.round(phi+rotation[1], decimals = 6) for phi in self.elegantTwiss['phi']], units='radians')
    self['psi'] = twissData([np.round(psi+rotation[2], decimals = 6) for psi in self.elegantTwiss['psi']], units='radians')
    xyz = list(zip(self['x'], self['y'], self['z']))
    thetaphipsi = list(zip(self['phi'], self['psi'], self['theta']))
    return list(zip(self.elegantTwiss['ElementName'], xyz[-1:] + xyz[:-1], xyz, thetaphipsi))[1:]

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
        z = self.elegantTwiss['Z']
        self.append('z', z)
        cp = self.elegantTwiss['pCentral0'] * self.E0
        # self.append('cp', cp)
        self.append('cp', cp / constants.elementary_charge)
        ke = np.array((np.sqrt(self.E0**2 + cp**2) - self.E0**2) / constants.elementary_charge)
        self.append('kinetic_energy', ke)
        gamma = 1 + ke / self.E0_eV
        self.append('gamma', gamma)
        self.append('p', cp * self.q_over_c)
        self.append('enx', self.elegantTwiss['enx'])
        self.append('ex', self.elegantTwiss['ex'])
        self.append('eny', self.elegantTwiss['eny'])
        self.append('ey', self.elegantTwiss['ey'])
        self.append('enz', np.zeros(len(self.elegantTwiss['Z'])))
        self.append('ez', np.zeros(len(self.elegantTwiss['Z'])))
        self.append('beta_x', self.elegantTwiss['betax'])
        self.append('alpha_x', self.elegantTwiss['alphax'])
        self.append('gamma_x', (1 + self.elegantTwiss['alphax']**2) / self.elegantTwiss['betax'])
        self.append('beta_y', self.elegantTwiss['betay'])
        self.append('alpha_y', self.elegantTwiss['alphay'])
        self.append('gamma_y', (1 + self.elegantTwiss['alphay']**2) / self.elegantTwiss['betay'])
        self.append('beta_z', np.zeros(len(self.elegantTwiss['Z'])))
        self.append('gamma_z', np.zeros(len(self.elegantTwiss['Z'])))
        self.append('alpha_z', np.zeros(len(self.elegantTwiss['Z'])))
        self.append('sigma_x', self.elegantTwiss['Sx'])
        self.append('sigma_y', self.elegantTwiss['Sy'])
        self.append('sigma_t', self.elegantTwiss['St'])
        beta = np.sqrt(1-(gamma**-2))
        # print 'len(z) = ', len(z), '  len(beta) = ', len(beta)
        self.append('t', z / (beta * constants.speed_of_light))
        self.append('sigma_z', self.elegantTwiss['St'] * (beta * constants.speed_of_light))
        # self.append('sigma_cp', self.elegantTwiss['Sdelta'] * cp )
        self.append('sigma_cp', self.elegantTwiss['Sdelta'] * cp / constants.elementary_charge)
        # print('elegant = ', (self.elegantTwiss['Sdelta'] * cp / constants.elementary_charge)[-1)
        self.append('sigma_p', self.elegantTwiss['Sdelta'] )
        self.append('mux', self.elegantTwiss['psix'] / (2*constants.pi))
        self.append('muy', self.elegantTwiss['psiy'] / (2*constants.pi))
        self.append('eta_x', self.elegantTwiss['etax'])
        self.append('eta_xp', self.elegantTwiss['etaxp'])
        self.append('element_name', self.elegantTwiss['ElementName'])
        ### BEAM parameters
        self.append('ecnx', self.elegantTwiss['ecnx'])
        self.append('ecny', self.elegantTwiss['ecny'])
        self.append('eta_x_beam', self.elegantTwiss['s16']/(self.elegantTwiss['s6']**2))
        self.append('eta_xp_beam', self.elegantTwiss['s26']/(self.elegantTwiss['s6']**2))
        self.append('eta_y_beam', self.elegantTwiss['s36']/(self.elegantTwiss['s6']**2))
        self.append('eta_yp_beam', self.elegantTwiss['s46']/(self.elegantTwiss['s6']**2))
        self.append('beta_x_beam', self.elegantTwiss['betaxBeam'])
        self.append('beta_y_beam', self.elegantTwiss['betayBeam'])
        self.append('alpha_x_beam', self.elegantTwiss['alphaxBeam'])
        self.append('alpha_y_beam', self.elegantTwiss['alphayBeam'])
