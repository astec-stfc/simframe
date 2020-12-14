import os
import subprocess
from collections import OrderedDict
from munch import Munch
from ...FrameworkHelperFunctions import *
from ...Modules.merge_two_dicts import merge_two_dicts
from ...Modules import constants
from ...Modules import Beams as rbf

astra_generator_keywords = {
    'keywords':{
        'filename': 'FName',
        'combine_distributions': 'add',
        'number_of_particles': 'Ipart',
        'probe_particle': 'probe',
        'noise_reduction': 'noise_reduc',
        'high_resolution': 'high_res',
        'charge': ['q_total', 1e9],
        'reference_position': 'ref_zpos',
        'reference_time': 'ref_clock',
        'distribution_type_z': 'dist_z',
        'inital_energy': 'ref_ekin',
        'plateau_bunch_length': ['lt', 1e9],
        'plateau_rise_time': ['rt', 1e9],
        'sigma_t': ['sig_clock', 1e9],
        'sigma_z': ['sig_z', 1e3],
        'bunch_length': ['lz', 1e3],
        'plateau_rise_distance': ['rz', 1e3],
        'distribution_type_pz': 'dist_pz',
        'thermal_emittance': 'le',
        'distribution_type_x': 'dist_x',
        'sigma_x': ['sig_x', 1e3],
        'distribution_type_y': 'dist_y',
        'sigma_y': ['sig_y', 1e3],
        'distribution_type_px': 'dist_px',
        'distribution_type_py': 'dist_py',
        'normalized_horizontal_emittance': ['Nemit_x', 1e6],
        'normalized_vertical_emittance': ['Nemit_y', 1e6],
        'guassian_cutoff_x': 'C_sig_x',
        'guassian_cutoff_y': 'C_sig_y',
        'guassian_cutoff_z': 'C_sig_z',
        'offset_x': ['x_off', 1e3],
        'offset_y': ['y_off', 1e3],
    },
}
gpt_generator_keywords = {
    'keywords': [
        'image_filename', 'image_calibration_x', 'image_calibration_y'
    ],
}
generator_keywords = {
    'defaults': {
        'clara_400_3ps':{
            'combine_distributions': False,
            'species': 'electrons',
            'probe_particle': True,
            'noise_reduction': False,
            'high_resolution': True,
            'cathode': True,
            'reference_position': 0,
            'reference_time': 0,
            'distribution_type_z': 'p',
            'inital_energy': 0,
            'plateau_bunch_length': 3e-12,
            'plateau_rise_time': 0.2e-12,
            'distribution_type_pz': 'i',
            'thermal_emittance': 0.9e-3,
            'distribution_type_x': 'radial',
            'sigma_x': 0.25e-3,
            'distribution_type_y': 'r',
            'sigma_y': 0.25e-3,
            'offset_x': 0,
            'offset_y': 0,
        },
        'clara_400_1ps':{
            'combine_distributions': False,'species': 'electrons', 'probe_particle': True,'noise_reduction': False, 'high_resolution': True, 'cathode': True,
            'reference_position': 0, 'reference_time': 0, 'distribution_type_z': 'p',
            'inital_energy': 0, 'plateau_bunch_length': 1e-12, 'plateau_rise_time': 0.2e-12, 'distribution_type_pz': 'i', 'thermal_emittance': 0.9e-3,
            'distribution_type_x': 'radial', 'sigma_x': 0.25e-3, 'distribution_type_y': 'r', 'sigma_y': 0.25e-3,
            'offset_x': 0, 'offset_y': 0,
        },
        'clara_400_2ps_Gaussian':{
            'combine_distributions': False,'species': 'electrons', 'probe_particle': True,'noise_reduction': False, 'high_resolution': True, 'cathode': True,
            'reference_position': 0, 'reference_time': 0,
            'distribution_type_z': 'g', 'sigma_t': 0.85e-12,
            'inital_energy': 0, 'distribution_type_pz': 'i', 'thermal_emittance': 0.9e-3,
            'distribution_type_x': '2DGaussian', 'sigma_x': 0.25e-3,
            'distribution_type_y': '2DGaussian', 'sigma_y': 0.25e-3,
            'guassian_cutoff_x': 3, 'guassian_cutoff_y': 3, 'guassian_cutoff_z': 3,
            'offset_x': 0, 'offset_y': 0,
        },
        'clara_400_2ps_flattop':{
            'combine_distributions': False,'species': 'electrons', 'probe_particle': True,'noise_reduction': False, 'high_resolution': True, 'cathode': True,
            'reference_position': 0, 'reference_time': 0,
            'distribution_type_z': 'p', 'plateau_bunch_length': 2e-12, 'plateau_rise_time': 0.2e-12,
            'distribution_type_pz': 'i', 'inital_energy': 0, 'distribution_type_pz': 'i', 'thermal_emittance': 0.9e-3,
            'distribution_type_x': '2DGaussian', 'sigma_x': 0.25e-3,
            'distribution_type_y': '2DGaussian', 'sigma_y': 0.25e-3,
            'guassian_cutoff_x': 3, 'guassian_cutoff_y': 3, 'guassian_cutoff_z': 3,
            'offset_x': 0, 'offset_y': 0,
        },
        'clara_400_2ps_flattop_radial':{
            'combine_distributions': False,'species': 'electrons', 'probe_particle': True,'noise_reduction': False, 'high_resolution': True, 'cathode': True,
            'reference_position': 0, 'reference_time': 0,
            'distribution_type_z': 'p', 'plateau_bunch_length': 2e-12, 'plateau_rise_time': 0.2e-12,
            'distribution_type_pz': 'i', 'inital_energy': 0, 'distribution_type_pz': 'i', 'thermal_emittance': 0.9e-3,
            'distribution_type_x': 'radial', 'sigma_x': 0.25e-3, 'distribution_type_y': 'r', 'sigma_y': 0.25e-3,
            'offset_x': 0, 'offset_y': 0,
        },
    },
    'keywords': [
        'number_of_particles', 'filename',
        'probe_particle', 'noise_reduction', 'high_resolution', 'combine_distributions',
        'cathode', 'cathode_radius',
        'charge', 'species',
        'emission_time', 'thermal_emittance', 'bunch_length', 'inital_energy',
        'sigma_x', 'sigma_y', 'sigma_z', 'sigma_t',
        'distribution_type_z', 'distribution_type_pz', 'distribution_type_x', 'distribution_type_px', 'distribution_type_y', 'distribution_type_py',
        'guassian_cutoff_x', 'guassian_cutoff_y', 'guassian_cutoff_z',
        'plateau_bunch_length', 'plateau_rise_time', 'plateau_rise_distance',
        'offset_x', 'offset_y',
        'reference_position', 'reference_time',
        'normalized_horizontal_emittance', 'normalized_vertical_emittance',
        'image_file'
    ]
}

elegant_generator_keywords = {
    'keywords':[
        'bunch','n_particles_per_bunch', 'time_start', 'matched_to_cell', 'emit_x', 'emit_nx', 'beta_x', 'alpha_x', 'eta_x', 'etap_x', 'emit_y',
        'emit_ny', 'beta_y', 'alpha_y', 'eta_y', 'etap_y', 'use_twiss_command_values', 'use_moments_output_values', 'Po', 'sigma_dp','sigma_s',
        'dp_s_coupling', 'emit_z', 'beta_z', 'alpha_z', 'momentum_chirp', 'one_random_bunch', 'symmetrize', 'optimized_halton', 'limit_invariants',
        'limit_in_4d', 'first_is_fiducial', 'save_initial_coordinates', 'halton_sequence', 'halton_radix', 'randomize_order', 'enforce_rms_values',
        'distribution_cutoff', 'distribution_type', 'centroid'
    ],
    'defaults': {
    },
}

class frameworkGenerator(Munch):

    electron_mass = constants.m_e
    elementary_charge = constants.elementary_charge
    speed_of_light = constants.speed_of_light

    def __init__(self, executables, global_parameters, **kwargs):
        super(frameworkGenerator, self).__init__()
        self.global_parameters = global_parameters
        self.executables = executables
        self.kwargs = kwargs
        self.objectdefaults = {}

    def run(self):
        pass

    def load_defaults(self, defaults):
        if isinstance(defaults, str) and defaults in generator_keywords['defaults']:
            self.__init__(self.executables, self.global_parameters, **generator_keywords['defaults'][defaults])
        elif isinstance(defaults, dict):
            self.__init__(self.executables, self.global_parameters, **defaults)

    @property
    def particles(self):
        return self.number_of_particles if self.number_of_particles is not None else 512

    @particles.setter
    def particles(self, npart):
        self.add_property('number_of_particles', npart)

    @property
    def charge(self):
        return float(self['charge']) if 'charge' in self and self['charge'] is not None else 250e-12
    @charge.setter
    def charge(self, q):
        self['charge'] = q

    @property
    def thermal_kinetic_energy(self):
        thermal_emittance = float(self['thermal_emittance']) if 'thermal_emittance' in self.keys() and self['thermal_emittance'] is not None else 0.9e-3
        return float((3 * thermal_emittance**2 * self.speed_of_light**2 * self.electron_mass) / 2 / self.elementary_charge)

    @property
    def objectname(self):
        return self['name'] if 'name' in self.keys() and self['name'] is not None else 'generator'

    def write(self):
        pass

    @property
    def parameters(self):
        """ This returns a dictionary of parameter keys and values"""
        return self.toDict()

    def __getattr__(self, a):
        """ If key does not exist return None """
        if a in self.keys():
            return self[a]
        return None

    def postProcess(self):
        pass

class ASTRAGenerator(frameworkGenerator):
    def __init__(self, executables, global_parameters, **kwargs):
        super(ASTRAGenerator, self).__init__(executables, global_parameters, **kwargs)
        astra_keywords = list(astra_generator_keywords['keywords'].values())
        keywords = generator_keywords['keywords']
        self.allowedKeyWords = [*astra_keywords, *keywords]
        self.allowedKeyWords = [x.lower() if not isinstance(x, list) else x[0].lower() for x in self.allowedKeyWords]
        for key, value in list(kwargs.items()):
            key = key.lower()
            if key in self.allowedKeyWords:
                try:
                    # print 'key = ', key
                    self[key] = value
                    setattr(self, key, value)
                except:
                    pass
                    # print 'WARNING: Unknown keyword: ', key, value
                    # exit()

    def run(self):
        command = self.executables['ASTRAgenerator'] + [self.objectname+'.in']
        with open(os.devnull, "w") as f:
            subprocess.call(command, stdout=f, cwd=self.global_parameters['master_subdir'])

    def _write_ASTRA(self, d):
        output = ''
        for k, v in list(d.items()):
            val = v['value'] if v['value'] is not None else v['default'] if 'default' in v else None
            if isinstance(val,str):
                param_string = k+' = \''+str(val)+'\',\n'
            else:
                param_string = k+' = '+str(val)+',\n'
            if len((output + param_string).splitlines()[-1]) > 70:
                output += '\n'
            output += param_string
        return output[:-2]

    def write(self):
        output = '&INPUT\n'
        try:
            npart = eval(self.number_of_particles)
        except:
            npart = self.number_of_particles
        if self.filename is None:
            self.filename = 'generator.txt'
        framework_dict = OrderedDict([
            ['q_total', {'value': self.charge*1e9, 'default': 0.25}],
            ['Lprompt', {'value': False}],
            ['le', {'value': 1e-3*self.thermal_kinetic_energy, 'default': 0.62e-3}],
        ])
        keyword_dict = OrderedDict()
        for k in self.allowedKeyWords:
            m = None
            klower = k.lower()
            if klower in astra_generator_keywords['keywords'].keys():
                k = astra_generator_keywords['keywords'][klower]
                if isinstance(k, list):
                    k, m = k
            if klower not in [fk.lower() for fk in framework_dict.keys()] and k not in [fk.lower() for fk in framework_dict.keys()]:
                if getattr(self, klower) is not None:
                    try:
                        val = eval(getattr(self, klower))
                    except:
                        val = getattr(self, klower)
                    if m is not None:
                        val = m * val
                    keyword_dict[k] = {'value': val}
                    # print(k, val)
        output += self._write_ASTRA(merge_two_dicts(framework_dict, keyword_dict))
        output += '\n/\n'
        saveFile(self.global_parameters['master_subdir']+'/'+self.objectname+'.in', output)

    def postProcess(self):
        astrabeamfilename = 'generator.txt'
        rbf.astra.read_astra_beam_file(self.global_parameters['beam'], self.global_parameters['master_subdir'] + '/' + astrabeamfilename, normaliseZ=False)
        HDF5filename = 'laser.hdf5'
        rbf.hdf5.write_HDF5_beam_file(self.global_parameters['beam'], self.global_parameters['master_subdir'] + '/' + HDF5filename, centered=False, sourcefilename=astrabeamfilename)

class GPTGenerator(frameworkGenerator):
    def __init__(self, executables, global_parameters, **kwargs):
        super(GPTGenerator, self).__init__(executables, global_parameters, **kwargs)
        gpt_keywords = list(gpt_generator_keywords['keywords'])
        keywords = generator_keywords['keywords']
        self.code = "gpt"
        self.allowedKeyWords = [*gpt_keywords, *keywords]
        self.allowedKeyWords = [x.lower() for x in self.allowedKeyWords]
        for key, value in list(kwargs.items()):
            key = key.lower()
            if key in self.allowedKeyWords:
                try:
                    # print 'key = ', key
                    self[key] = value
                    setattr(self, key, value)
                except:
                    pass
                    # print 'WARNING: Unknown keyword: ', key, value
                    # exit()

    def run(self):
        """Run the code with input 'filename'"""
        command = self.executables[self.code] + ['-o', 'generator.gdf'] + ['GPTLICENSE='+self.global_parameters['GPTLICENSE']] + [self.objectname+'.in']
        my_env = os.environ.copy()
        my_env["LD_LIBRARY_PATH"] = my_env["LD_LIBRARY_PATH"] + ":/opt/GPT3.3.6/lib/" if "LD_LIBRARY_PATH" in my_env else "/opt/GPT3.3.6/lib/"
        my_env["OMP_WAIT_POLICY"] = "PASSIVE"
        # post_command_t = [self.executables[self.code][0].replace('gpt.exe','gdfa.exe')] + ['-o', self.objectname+'_emit.gdf'] + [self.objectname+'_out.gdf'] + ['time','avgx','avgy','stdx','stdBx','stdy','stdBy','stdz','stdt','nemixrms','nemiyrms','nemizrms','numpar','nemirrms','avgG','avgp','stdG','avgt','avgBx','avgBy','avgBz','CSalphax','CSalphay','CSbetax','CSbetay']
        # post_command = [self.executables[self.code][0].replace('gpt','gdfa')] + ['-o', self.objectname+'_emit.gdf'] + [self.objectname+'_out.gdf'] + ['position','avgx','avgy','stdx','stdBx','stdy','stdBy','stdz','stdt','nemixrms','nemiyrms','nemizrms','numpar','nemirrms','avgG','avgp','stdG','avgt','avgBx','avgBy','avgBz','CSalphax','CSalphay','CSbetax','CSbetay']
        # post_command_t = [self.executables[self.code][0].replace('gpt','gdfa')] + ['-o', self.objectname+'_emitt.gdf'] + [self.objectname+'_out.gdf'] + ['time','avgx','avgy','stdx','stdBx','stdy','stdBy','stdz','nemixrms','nemiyrms','nemizrms','numpar','nemirrms','avgG','avgp','stdG','avgBx','avgBy','avgBz','CSalphax','CSalphay','CSbetax','CSbetay']
        # post_command_traj = [self.executables[self.code][0].replace('gpt','gdfa')] + ['-o', self.objectname+'traj.gdf'] + [self.objectname+'_out.gdf'] + ['time','avgx','avgy','avgz']
        with open(os.path.relpath(self.global_parameters['master_subdir']+'/'+self.objectname+'.log', '.'), "w") as f:
            # print('gpt command = ', command)
            subprocess.call(command, stdout=f, cwd=self.global_parameters['master_subdir'], env=my_env)
            # subprocess.call(post_command, stdout=f, cwd=self.global_parameters['master_subdir'])
            # subprocess.call(post_command_t, stdout=f, cwd=self.global_parameters['master_subdir'])
            # subprocess.call(post_command_traj, stdout=f, cwd=self.global_parameters['master_subdir'])

    def generate_particles(self):
        return """#--Basic beam parameters--
E0 = """ + str(self.thermal_kinetic_energy) + """;
G = 1-qe*E0/(me * c * c);
GB = sqrt(G^2 - 1);
Qtot = """ + str(-1e12*self.charge) + """e-12;
npart = """ + str(self.particles) + """;
setparticles( "beam", npart, me, qe, Qtot ) ;
"""

    def check_xy_parameters(self, x: str, y: str, default: str):
        if getattr(self, x) is None and getattr(self, y) is not None:
            setattr(self, x, getattr(self, y))
        elif getattr(self, x) is not None and getattr(self, y) is None:
            setattr(self, y, getattr(self, x))
        elif getattr(self, x) is None and getattr(self, y) is None:
            setattr(self, x, default)
            setattr(self, y, default)

    def _uniform_distribution(self, distname: str, variable: str, left_multiplier=1, right_multiplier=2, **kwargs):
        return distname + '( "beam", "u", ' + str(left_multiplier) + '*' + variable + ', ' + str(right_multiplier) + '*' + variable + ') ;'

    def _gaussian_distribution(self, distname: str, variable: str, left_cutoff=3, right_cutoff=3, **kwargs):
        return distname + '( "beam", "g", 0, ' + variable + ', '+ str(left_cutoff) +', '+ str(right_cutoff) +') ;'

    def _distribution(self, param, distname, variable, **kwargs):
        if getattr(self,param).lower() in ["g","gaussian","2dgaussian"]:
            return self._gaussian_distribution(distname, variable, **kwargs)
        else:# self.distribution_type_x.lower() in ["u","uniform"]:
            return self._uniform_distribution(distname, variable, **kwargs)

    def generate_radial_distribution(self):
        # self.check_xy_parameters("sigma_x", "sigma_y", 1)
        # self.check_xy_parameters("distribution_type_x", "distribution_type_y", "g")
        if self.distribution_type_x == 'image' or self.distribution_type_y == 'image':
            image_filename = os.path.relpath('./'+self.image_filename, start=self.global_parameters['master_subdir']+'/')
            image_calibration_x = self.image_calibration_x if isinstance(self.image_calibration_x, int) and self.image_calibration_x > 0 else 1000 * 1e3
            image_calibration_y = self.image_calibration_y if isinstance(self.image_calibration_y, int) and self.image_calibration_y > 0 else 1000 * 1e3
            output =  'setxydistbmp("beam", \"' + str(image_filename) + '\", ' + str(image_calibration_x) + ', ' + str(image_calibration_y) + ') ;\n'
            return output
        elif (self.sigma_x != self.sigma_y) and (self.distribution_type_x == self.distribution_type_y):
            output =   "radius_x = " + str(self.sigma_x) + ";\n"
            output +=  "radius_y = " + str(self.sigma_y) + ";\n"
            # output += self._distribution('distribution_type_x', 'setxdist', 'radius_x', left_cutoff=self.guassian_cutoff_x, right_cutoff=self.guassian_cutoff_x) + "\n"
            # output += self._distribution('distribution_type_y', 'setydist', 'radius_y', left_cutoff=self.guassian_cutoff_y, right_cutoff=self.guassian_cutoff_y) + "\n"
            output += 'setellipse("beam", 2.0*radius_x, 2.0*radius_y, 1e-12);\n'
            return output
        elif (self.sigma_x == self.sigma_y) and (self.distribution_type_x == self.distribution_type_y):
            output =  "radius = " + str(self.sigma_x) + ";\n"
            output += self._distribution('distribution_type_x', 'setxdist', 'radius', left_cutoff=0, right_cutoff=self.guassian_cutoff_x) + "\n"
            output += 'setphidist("beam", "u", 0, 2*pi) ;\n'
            return output
        else:
            return ''

    def generate_phase_space_distribution(self):
        return '''#--Initial Phase-Space--
setGBzdist( "beam", "u", GB, 0 ) ;
setGBthetadist("beam","u", pi/4, pi/2);
setGBphidist("beam","u", 0, 2*pi);
'''

    def generate_thermal_emittance(self):
        if self.distribution_type_x == 'image' or self.distribution_type_x == 'image':
            return '\n'
        elif (self.sigma_x != self.sigma_y):
            thermal_emittance = float(self['thermal_emittance']) if 'thermal_emittance' in self.keys() and self['thermal_emittance'] is not None else 0.9e-3
            output = '''setGBxemittance("beam", (''' + str(thermal_emittance) + '''*radius_x)) ;'''
            output += '''setGByemittance("beam", (''' + str(thermal_emittance) + '''*radius_y)) ;'''
            return output
        else:
            thermal_emittance = float(self['thermal_emittance']) if 'thermal_emittance' in self.keys() and self['thermal_emittance'] is not None else 0.9e-3
            return '''setGBxemittance("beam", (''' + str(thermal_emittance) + '''*radius)) ;
setGByemittance("beam", (''' + str(thermal_emittance) + '''*radius)) ;
'''

    def generate_longitudinal_distribution(self):
        if self.distribution_type_z.lower() in ['g','gaussian']:
            output = '''tlen = ''' + str(1e12*self.sigma_t) + '''e-12;\n'''
        else:
            output = '''tlen = ''' + str(1e12*self.plateau_bunch_length) + '''e-12;\n'''
        output += self._distribution('distribution_type_z', 'settdist', 'tlen', left_cutoff=self.guassian_cutoff_z, right_cutoff=self.guassian_cutoff_z, left_multiplier=0, right_multiplier=1) + "\n"
        return output

    def generate_output(self):
        return '''screen( "wcs", "I", 0) ;
'''

    def generate_offset_transform(self):
        return 'settransform("wcs", ' + str(self.offset_x) + ', ' + str(self.offset_y) + ', ' + '0, 1, 0, 0, 0, 1, 0, "beam");\n'

    def write(self):
        # try:
        #     npart = eval(self.number_of_particles)
        # except:
        #     npart = self.number_of_particles
        if self.filename is None:
            self.filename = 'laser.in'
        output = ''
        output += self.generate_particles()
        output += self.generate_radial_distribution()
        output += self.generate_phase_space_distribution()
        output += self.generate_thermal_emittance()
        output += self.generate_longitudinal_distribution()
        output += self.generate_offset_transform()
        output += self.generate_output()
        # print('output = ', output)
        saveFile(self.global_parameters['master_subdir']+'/'+self.objectname+'.in', output)

    def postProcess(self):
        gptbeamfilename = 'generator.gdf'
        self.global_parameters['beam'].read_gdf_beam_file(self.global_parameters['master_subdir'] + '/' + gptbeamfilename, position=0, longitudinal_reference='t')
        # Set the Z component to be zero
        self.global_parameters['beam']['z'] = 0 * self.global_parameters['beam']['z']
        HDF5filename = 'laser.hdf5'
        self.global_parameters['beam']['status'] = np.full(len(self.global_parameters['beam']['x']), -1)
        self.global_parameters['beam'].write_HDF5_beam_file(self.global_parameters['master_subdir'] + '/' + HDF5filename, centered=False, sourcefilename=gptbeamfilename)
