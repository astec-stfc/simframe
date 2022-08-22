import os
from ...Framework_objects import *
from ...Framework_elements import *
from ...Modules import Beams as rbf

class elegantLattice(frameworkLattice):
    def __init__(self, *args, **kwargs):
        super(elegantLattice, self).__init__(*args, **kwargs)
        self.code = 'elegant'
        self.particle_definition = self.allElementObjects[self.start].objectname
        self.bunch_charge = None
        self.q = charge(name='START', type='charge', global_parameters=self.global_parameters,**{'total': 250e-12})
        self.trackBeam = True
        self.betax = None
        self.betay = None
        self.alphax = None
        self.alphay = None

    def endScreen(self, **kwargs):
        return screen(name='end', type='screen', position_start=self.endObject.position_start, position_end=self.endObject.position_start, global_rotation=self.endObject.global_rotation, global_parameters=self.global_parameters, **kwargs)

    def writeElements(self):
        self.w = self.endScreen(output_filename=self.end+'.SDDS')
        elements = self.createDrifts()
        fulltext = ''
        fulltext += self.q.write_Elegant()
        for element in list(elements.values()):
            # print(element.write_Elegant())
            if not element.subelement:
                fulltext += element.write_Elegant()
        fulltext += self.w.write_Elegant()
        fulltext += self.objectname+': Line=(START, '
        for e, element in list(elements.items()):
            if not element.subelement:
                if len((fulltext + e).splitlines()[-1]) > 60:
                    fulltext += '&\n'
                fulltext += e+', '
        return fulltext[:-2] + ', END )\n'

    def checkErrorElements(self):
        '''prepare the dictionary of element error definitions'''
        default_err = {'amplitude': 1E-6,
                       'fractional': 0,
                       'type': '"gaussian"'
                       }

        # check all element names in the errorElements object are correct
        output = {}
        for ele in self.errorElements:
            # check if the element exists in the lattice at all
            if ele not in self.allElements:
                raise KeyError('Specified element %s does not exist in the lattice' % str(ele))

            if ele in self.elements:
                # fetch object type
                eleType = str(self.allElementObjects[ele].objecttype)
                output[ele] = {}

                # iterate through parameters
                for item in self.errorElements[ele]:
                    # check that parameter is valid for the specified element type
                    if item not in elementkeywords[eleType]['keywords']:
                        raise KeyError('Element type %s has no keyword %s' % (str(eleType), item))

                    # check for keyword conversions
                    conversions = keyword_conversion_rules_elegant[eleType]
                    keyword = conversions[item] if (item in conversions) else item
                    output[ele][keyword] = copy.copy(default_err)

                    # fill in defined error parameters
                    for key in default_err:
                        if key in self.errorElements[ele][item]:
                            output[ele][keyword][key] = self.errorElements[ele][item][key]
        self.errorElements = output

    def write(self):
        self.lattice_file = self.global_parameters['master_subdir']+'/'+self.objectname+'.lte'
        saveFile(self.lattice_file, self.writeElements())
        try:
            self.command_file = self.global_parameters['master_subdir']+'/'+self.objectname+'.ele'
            saveFile(self.command_file, self.commandFile.write())
        except:
            pass

    def preProcess(self):
        prefix = self.file_block['input']['prefix'] if 'input' in self.file_block and 'prefix' in self.file_block['input'] else ''
        if self.trackBeam:
            self.hdf5_to_sdds(prefix)
        self.checkErrorElements()
        self.commandFile = elegantTrackFile(lattice=self, trackBeam=self.trackBeam, elegantbeamfilename=self.objectname+'.sdds', sample_interval=self.sample_interval,
        betax=self.betax,
        betay=self.betay,
        alphax=self.alphax,
        alphay=self.alphay,
        global_parameters=self.global_parameters,
        error_elements=self.errorElements)

    def postProcess(self):
        if self.trackBeam:
            for s in self.screens:
                s.sdds_to_hdf5()
            self.w.sdds_to_hdf5()

    def hdf5_to_sdds(self, prefix=''):
        HDF5filename = prefix+self.particle_definition+'.hdf5'
        rbf.hdf5.read_HDF5_beam_file(self.global_parameters['beam'], self.global_parameters['master_subdir'] + '/' + HDF5filename)
        if self.bunch_charge is not None:
            self.q = charge(name='START', type='charge', global_parameters=self.global_parameters,**{'total': abs(self.bunch_charge)})
        else:
            self.q = charge(name='START', type='charge', global_parameters=self.global_parameters,**{'total': abs(self.global_parameters['beam'].Q)})
        # print('mean cpz = ', np.mean(self.global_parameters['beam'].cpz), ' prefix = ', prefix)
        sddsbeamfilename = self.objectname+'.sdds'
        rbf.sdds.write_SDDS_file(self.global_parameters['beam'], self.global_parameters['master_subdir'] + '/' + sddsbeamfilename, xyzoffset=self.startObject.position_start)

    def run(self):
        """Run the code with input 'filename'"""
        if not os.name == 'nt':
            command = self.executables[self.code] + [self.objectname+'.ele']
            if self.global_parameters['simcodes_location'] is None:
                my_env = {**os.environ}
            else:
                my_env = {**os.environ, 'RPN_DEFNS': os.path.abspath(self.global_parameters['simcodes_location'])+'/Elegant/defns_linux.rpn'}
            with open(os.path.abspath(self.global_parameters['master_subdir']+'/'+self.objectname+'.log'), "w") as f:
                subprocess.call(command, stdout=f, cwd=self.global_parameters['master_subdir'], env=my_env)
        else:
            code_string = ' '.join(self.executables[self.code]).lower()
            command = self.executables[self.code] + [self.objectname+'.ele']
            if 'pelegant' in code_string:
                command = [command[0]] + ['-env','RPN_DEFNS',(os.path.abspath(self.global_parameters['simcodes_location'])+'/Elegant/defns.rpn').replace('/','\\')] + command[1:]
                command = [c.replace('/','\\') for c in command]
                with open(os.path.abspath(self.global_parameters['master_subdir']+'/'+self.objectname+'.log'), "w") as f:
                    subprocess.call(command, stdout=f, cwd=self.global_parameters['master_subdir'])
            else:
                command = [c.replace('/','\\') for c in command]
                with open(os.path.abspath(self.global_parameters['master_subdir']+'/'+self.objectname+'.log'), "w") as f:
                    subprocess.call(command, stdout=f, cwd=self.global_parameters['master_subdir'], env={'RPN_DEFNS': (os.path.abspath(self.global_parameters['simcodes_location'])+'/Elegant/defns.rpn').replace('/','\\')})

class elegantCommandFile(object):
    def __init__(self, lattice='', *args, **kwargs):
        super(elegantCommandFile, self).__init__()
        self.global_parameters = kwargs['global_parameters']
        self.commandObjects = OrderedDict()
        self.lattice_filename = lattice.objectname+'.lte'
        self.ncommands = 0

    def addCommand(self, objectname=None, **kwargs):
        if objectname == None:
            if not 'objectname' in kwargs:
                if not 'objecttype' in kwargs:
                    raise NameError('Command does not have a name')
                else:
                    objectname = kwargs['objecttype']
            else:
                objectname = kwargs['objectname']
        command = frameworkCommand(objectname, global_parameters=self.global_parameters, **kwargs)
        self.commandObjects[self.ncommands] = command
        self.ncommands += 1
        return command

    def write(self):
        output = ''
        for c in list(self.commandObjects.values()):
            output += c.write_Elegant()
        return output

class elegantTrackFile(elegantCommandFile):
    def __init__(self, error_elements={}, lattice='', trackBeam=True, elegantbeamfilename='', betax=None, betay=None, alphax=None, alphay=None, etax=None, etaxp=None, *args, **kwargs):
        super(elegantTrackFile, self).__init__(lattice, *args, **kwargs)
        self.elegantbeamfilename = elegantbeamfilename
        self.sample_interval = kwargs['sample_interval'] if 'sample_interval' in kwargs else 1
        self.trackBeam = trackBeam
        self.betax = betax if betax is not None else self.global_parameters['beam'].twiss.beta_x_corrected
        self.betay = betay if betay is not None else self.global_parameters['beam'].twiss.beta_y_corrected
        self.alphax = alphax if alphax is not None else self.global_parameters['beam'].twiss.alpha_x_corrected
        self.alphay = alphay if alphay is not None else self.global_parameters['beam'].twiss.alpha_y_corrected
        self.etax = etax if etax is not None else self.global_parameters['beam'].twiss.eta_x
        self.etaxp = etaxp if etaxp is not None else self.global_parameters['beam'].twiss.eta_xp
        if not os.name == 'nt':
            self.addCommand(objecttype='global_settings', mpi_io_read_buffer_size=16777216, mpi_io_write_buffer_size=16777216, inhibit_fsync=1)
        else:
            self.addCommand(objecttype='global_settings', inhibit_fsync=1)
        self.addCommand(objecttype='run_setup',lattice=self.lattice_filename, \
            use_beamline=lattice.objectname,p_central=np.mean(self.global_parameters['beam'].BetaGamma), \
            centroid='%s.cen',always_change_p0 = 1, \
            sigma='%s.sig', default_order=3)

        enable_errors = True if (len(error_elements) > 0) else False
        if enable_errors:
            self.addCommand(objecttype='run_control', n_steps=2, n_passes=1, reset_rf_for_each_step=0, first_is_fiducial=1)
            self.addCommand(objecttype='error_control', no_errors_for_first_step=1, error_log='%s.erl')
            for e in error_elements.keys():
                for item in error_elements[e]:
                    self.addCommand(objecttype='error_element', name=e, item=item, allow_missing_elements=1, **error_elements[e][item])
        else:
            self.addCommand(objecttype='run_control', n_steps=1, n_passes=1)
        self.addCommand(objecttype='twiss_output',matched = 0,output_at_each_step=0,radiation_integrals=1,statistics=1,filename="%s.twi",
        beta_x  = self.betax,
        alpha_x = self.alphax,
        beta_y  = self.betay,
        alpha_y = self.alphay,
        eta_x = self.etax,
        etap_x = self.etaxp)
        flr = self.addCommand(objecttype='floor_coordinates', filename="%s.flr",
        X0  = lattice.startObject['position_start'][0],
        Z0 = lattice.startObject['position_start'][2],
        theta0 = 0)
        mat = self.addCommand(objecttype='matrix_output', SDDS_output="%s.mat",
        full_matrix_only=0, SDDS_output_order=2)
        if self.trackBeam:
            if enable_errors:
                self.addCommand(objecttype='sdds_beam', input=self.elegantbeamfilename, sample_interval=self.sample_interval, reuse_bunch=1)
            else:
                self.addCommand(objecttype='sdds_beam', input=self.elegantbeamfilename, sample_interval=self.sample_interval)
            self.addCommand(objecttype='track')

class elegantOptimisation(elegantCommandFile):

    def __init__(self, lattice='', variables={}, constraints={}, terms={}, settings={}, *args, **kwargs):
        super(elegantOptimisation, self).__init__(lattice, *args, **kwargs)
        for k, v in list(variables.items()):
            self.add_optimisation_variable(k, **v)

    def add_optimisation_variable(self, name, item=None, lower=None, upper=None, step=None, restrict_range=None):
        self.addCommand(name=name, type='optimization_variable', item=item, lower_limit=lower, upper_limit=upper, step_size=step, force_inside=restrict_range)

    def add_optimisation_constraint(self, name, item=None, lower=None, upper=None):
        self.addCommand(name=name, type='optimization_constraint', quantity=item, lower=lower, upper=upper)

    def add_optimisation_term(self, name, item=None, **kwargs):
        self.addCommand(name=name, type='optimization_term', term=item, **kwargs)
