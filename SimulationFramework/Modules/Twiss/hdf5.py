import h5py

def read_hdf_summary(self, filename, reset=True):
    if reset:
        self.reset_dicts()
    f = h5py.File(filename, "r")
    xemit = f.get('Xemit')
    yemit = f.get('Yemit')
    zemit = f.get('Zemit')
    for item, params in sorted(xemit.items()):
        self.interpret_astra_data(np.array(xemit.get(item)), np.array(yemit.get(item)), np.array(zemit.get(item)))


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
