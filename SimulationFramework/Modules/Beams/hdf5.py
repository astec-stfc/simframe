import h5py
import numpy as np
from .. import constants

def rotate_beamXZ(self, theta, preOffset=[0,0,0], postOffset=[0,0,0]):
    preOffset=np.array(preOffset)
    postOffset=np.array(postOffset)

    rotation_matrix = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-1*np.sin(theta), 0, np.cos(theta)]])
    beam = np.array([self.x,self.y,self.z]).transpose()
    self._beam['x'],self._beam['y'],self._beam['z'] = (np.dot(beam-preOffset, rotation_matrix)-postOffset).transpose()

    beam = np.array([self.px, self.py, self.pz]).transpose()
    self._beam['px'], self._beam['py'], self._beam['pz'] = np.dot(beam, rotation_matrix).transpose()

    if 'reference_particle' in self._beam:
        beam = np.array([self._beam['reference_particle'][0], self._beam['reference_particle'][1], self._beam['reference_particle'][2]])
        self._beam['reference_particle'][0], self._beam['reference_particle'][1], self._beam['reference_particle'][2] = (np.dot([beam-preOffset], rotation_matrix)[0]-postOffset)
        # print 'rotated ref part = ', np.dot([beam-preOffset], rotation_matrix)[0]
        beam = np.array([self._beam['reference_particle'][3], self._beam['reference_particle'][4], self._beam['reference_particle'][5]])
        self._beam['reference_particle'][3], self._beam['reference_particle'][4], self._beam['reference_particle'][5] = np.dot([beam], rotation_matrix)[0]

    self['rotation'] = theta
    self._beam['offset'] = preOffset

def unrotate_beamXZ(self):
    offset = self._beam['offset'] if 'offset' in self._beam else np.array([0,0,0])
    if 'rotation' in self._beam or abs(self['rotation']) > 0:
        self.rotate_beamXZ(-1*self['rotation'], -1*offset)

def write_HDF5_beam_file(self, filename, centered=False, mass=constants.m_e, sourcefilename=None, pos=None, rotation=None, longitudinal_reference='t', xoffset=0, yoffset=0, zoffset=0):
    # print('zoffset = ', zoffset, type(zoffset))
    if isinstance(zoffset,(list, np.ndarray)) and len(zoffset) == 3:
        xoffset = zoffset[0]
        yoffset = zoffset[1]
        zoffset = zoffset[2]
    # else:
    #     xoffset = 0 if xoffset is None else xoffset
    #     yoffset = 0 if yoffset is None else yoffset
    # print('xoffset = ', xoffset)
    # print('yoffset = ', yoffset)
    # print('zoffset = ', zoffset)
    with h5py.File(filename, "w") as f:
        inputgrp = f.create_group("Parameters")
        if not 'total_charge' in self._beam or self._beam['total_charge'] == 0:
            self._beam['total_charge'] = np.sum(self._beam['charge'])
        if sourcefilename is not None:
            inputgrp['Source'] = sourcefilename
        if pos is not None:
            inputgrp['Starting_Position'] = pos
        else:
            inputgrp['Starting_Position'] = [0, 0, 0]
        if rotation is not None:
            inputgrp['Rotation'] = rotation
        else:
            inputgrp['Rotation'] = 0
        inputgrp['total_charge'] = self._beam['total_charge']
        inputgrp['npart'] = len(self.x)
        inputgrp['centered'] = centered
        inputgrp['code'] = self['code']
        inputgrp['particle_mass'] = mass
        beamgrp = f.create_group("beam")
        if 'reference_particle' in self._beam:
            beamgrp['reference_particle'] = self._beam['reference_particle']
        if 'status' in self._beam:
            beamgrp['status'] = self._beam['status']
        beamgrp['longitudinal_reference'] = longitudinal_reference
        if len(self._beam['charge']) == len(self.x):
            chargevector = self._beam['charge']
        else:
            chargevector = np.full(len(self.x), self.charge/len(self.x))
        array = np.array([self.x + xoffset, self.y + yoffset, self.z + zoffset, self.cpx, self.cpy, self.cpz, self.t, chargevector]).transpose()
        beamgrp['columns'] = np.array(['x','y','z', 'cpx', 'cpy', 'cpz', 't', 'q'], dtype='S')
        beamgrp['units'] = np.array(['m','m','m','eV','eV','eV','s','e'], dtype='S')
        beamgrp.create_dataset("beam", data=array)

def read_HDF5_beam_file(self, filename, local=False):
    self.reset_dicts()
    self.filename = filename
    self['code'] = "SimFrame"
    with h5py.File(filename, "r") as h5file:
        if h5file.get('beam/reference_particle') is not None:
            self._beam['reference_particle'] = np.array(h5file.get('beam/reference_particle'))
        if h5file.get('beam/longitudinal_reference') is not None:
            self['longitudinal_reference'] = np.array(h5file.get('beam/longitudinal_reference'))
        else:
            self['longitudinal_reference'] = 't'
        if h5file.get('beam/status') is not None:
            self._beam['status'] = np.array(h5file.get('beam/status'))
        x, y, z, cpx, cpy, cpz, t, charge = np.array(h5file.get('beam/beam')).transpose()
        cp = np.sqrt(cpx**2 + cpy**2 + cpz**2)
        self._beam['x'] = x
        self._beam['y'] = y
        self._beam['z'] = z
        # self._beam['cpx'] = cpx
        # self._beam['cpy'] = cpy
        # self._beam['cpz'] = cpz
        self._beam['px'] = cpx * self.q_over_c
        self._beam['py'] = cpy * self.q_over_c
        self._beam['pz'] = cpz * self.q_over_c
        # self._beam['cp'] = cp
        # self._beam['p'] = cp * self.q_over_c
        # self._beam['xp'] = np.arctan(self.px/self.pz)
        # self._beam['yp'] = np.arctan(self.py/self.pz)
        self._beam['clock'] = np.full(len(self.x), 0)
        # self._beam['gamma'] = np.sqrt(1+(self.cp/self.E0_eV)**2)
        # velocity_conversion = 1 / (constants.m_e * self.gamma)
        # self._beam['vx'] = velocity_conversion * self.px
        # self._beam['vy'] = velocity_conversion * self.py
        # self._beam['vz'] = velocity_conversion * self.pz
        # self._beam['Bx'] = self.vx / constants.speed_of_light
        # self._beam['By'] = self.vy / constants.speed_of_light
        # self._beam['Bz'] = self.vz / constants.speed_of_light
        self._beam['t'] = t
        self._beam['charge'] = charge
        self._beam['total_charge'] = np.sum(self._beam['charge'])
        startposition = np.array(h5file.get('/Parameters/Starting_Position'))
        startposition = startposition if startposition is not None else [0,0,0]
        self['starting_position'] = startposition
        theta =  np.array(h5file.get('/Parameters/Rotation'))
        theta = theta if theta is not None else 0
        self['rotation'] = theta
        if local == True:
            self.rotate_beamXZ(self['rotation'], preOffset=self['starting_position'])
