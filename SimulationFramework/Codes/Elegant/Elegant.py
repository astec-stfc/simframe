import os, sdds, shutil
import lox
from ...Framework_objects import *
from ...Framework_elements import *
from ...Modules import Beams as rbf
from ...Modules.merge_two_dicts import merge_two_dicts

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
        self.commandFiles = {}

    def endScreen(self, **kwargs):
        return screen(name='end', type='screen', centre=self.endObject.centre, global_rotation=self.endObject.global_rotation, global_parameters=self.global_parameters, **kwargs)

    def writeElements(self):
        self.w = None
        if not self.endObject in self.screens_and_bpms:
            self.w = self.endScreen(output_filename=self.endObject.objectname+'.sdds')
        elements = self.createDrifts()
        fulltext = ''
        fulltext += self.q.write_Elegant()
        for element in list(elements.values()):
            # print(element.write_Elegant())
            if not element.subelement:
                fulltext += element.write_Elegant()
        fulltext += self.w.write_Elegant() if self.w is not None else ''
        fulltext += self.objectname+': Line=(START, '
        for e, element in list(elements.items()):
            if not element.subelement:
                if len((fulltext + e).splitlines()[-1]) > 60:
                    fulltext += '&\n'
                fulltext += e+', '
        fulltext = fulltext[:-2] + ', END )\n' if self.w is not None else fulltext[:-2] + ')\n'
        return fulltext

    def processRunSettings(self):
        '''process the runSettings object to extract the number of runs and the random number seed,
           and extract error definitions or a parameter scan definiton pertaining to this lattice section'''
        nruns = self.runSettings.nruns
        seed = self.runSettings.seed
        elementErrors = None if (self.runSettings.elementErrors is None) else self.processElementErrors(self.runSettings.elementErrors)
        elementScan   = None if (self.runSettings.elementScan is None) else self.processElementScan(self.runSettings.elementScan, nruns)
        return nruns, seed, elementErrors, elementScan

    def processElementErrors(self, elementErrors):
        '''process the elementErrors dictionary to prepare it for use with the current lattice section in ELEGANT'''
        output = {}
        default_err = {'amplitude':  1E-6,
                       'fractional': 0,
                       'type': '"gaussian"',
                        }

        for ele, error_defn in elementErrors.items():
            # identify element names with wildcard characters
            wildcard = ('*' in ele)

            # raise errors for non-wildcarded element names that don't exist in the global lattice
            if (ele not in self.allElements) and (not wildcard):
                raise KeyError('Lattice element %s does not exist in the current lattice' % str(ele))

            # check if the lattice element (or a wildcard match) exist in the local lattice section
            # fetch the element type (or the types of matching elements, if using a wildcard name)
            element_exists = False
            if (ele in self.elements) and not wildcard:
                element_exists = True
                element_types = [self.allElementObjects[ele].objecttype]
            elif wildcard:
                element_matches = [x for x in self.elements if (ele.replace('*', '') in x)]
                if len(element_matches) != 0:
                    element_exists = True
                    element_types = [self.allElementObjects[x].objecttype for x in element_matches]

            # if the element exists in the local lattice, do processing
            if element_exists:
                output[ele] = {}

                # check that all matching elements have the same type
                ele_type = str(element_types[0])
                has_expected_type = [(x == ele_type) for x in element_types]
                if not all(has_expected_type):
                    raise TypeError('All lattice elements matching a wilcarded element name must have the same type')

                # check error definition associated with each of the element parameters
                # for example, an element corresponding to an RF cavity might have parameters 'amplitude' and 'phase'
                for param in error_defn:
                    # check that the current element type has this parameter
                    if param not in elementkeywords[ele_type]['keywords']:
                        raise KeyError('Element type %s has no associated keyword %s' % (str(ele_type), str(param)))

                    # check for keyword conversions between simframe and elegant
                    # for example, in simframe the elegant parameter 'voltage' for RF cavities is called 'amplitude'
                    conversions = keyword_conversion_rules_elegant[ele_type]
                    keyword = conversions[param] if (param in conversions) else param
                    output[ele][keyword] = copy.copy(default_err)

                    # fill in the define error parameters
                    for k in default_err:
                        if k in error_defn[param]:
                            output[ele][keyword][k] = error_defn[param][k]

                    # bind errors across wildcarded elements
                    if wildcard:
                        output[ele][keyword]['bind'] = 1
                        output[ele][keyword]['bind_across_names'] = 1
        return output

    def processElementScan(self, elementScan, nsteps):
        '''process the elementScan dictionary to prepare it for use with the current lattice section in ELEGANT'''
        # extract the name of the beamline element, and the parameter to scan
        ele, param = elementScan['name'], elementScan['item']

        # raise errors for element names that don't exist anywhere in the global lattice
        if (ele not in self.allElements):
            raise KeyError('Lattice element %s does not exist in the current lattice' % str(ele))

        # check if the lattice element exists in the local lattice section and fetch the element type
        element_exists = (ele in self.elements)
        if element_exists:
            ele_type = self.allElementObjects[ele].objecttype

            # check that the element type has the parameter corresponding to the scan variable
            if param not in elementkeywords[ele_type]['keywords']:
                raise KeyError('Element type %s has no associated parameter %s' % (str(ele_type), str(param)))

            # check for keyword conversions between simframe and elegant
            conversions = keyword_conversion_rules_elegant[ele_type]
            keyword = conversions[param] if (param in conversions) else param

            # build the scan value array
            scan_values = np.linspace(elementScan['min'], elementScan['max'], int(nsteps)-1)

            # the first scan step is always the baseline simulation, for fiducialization
            multiplicative = elementScan['multiplicative']
            if multiplicative:
                scan_values = [1.] + list(scan_values)
            else:
                scan_values = [0.] + list(scan_values)

            # build the SDDS file with the scan values
            scan_fname = '%s-%s.sdds' % (ele, param)
            scanSDDS = sddsFile()
            scanSDDS.add_column('values', scan_values)
            scanSDDS.save(self.global_parameters['master_subdir']+'/'+scan_fname)

            output = {'name': ele,
                      'item': keyword,
                      'differential': int(not multiplicative),
                      'multiplicative': int(multiplicative),
                      'enumeration_file': scan_fname,
                      'enumeration_column': 'values'
                      }
            return output

        else:
            return None

    def write(self):
        self.lattice_file = self.global_parameters['master_subdir']+'/'+self.objectname+'.lte'
        saveFile(self.lattice_file, self.writeElements())
        try:
            self.command_file = self.global_parameters['master_subdir']+'/'+self.objectname+'.ele'
            saveFile(self.command_file, '', 'w')
            for cfileid in self.commandFilesOrder:
                if cfileid in self.commandFiles:
                    cfile = self.commandFiles[cfileid]
                    saveFile(self.command_file, cfile.write(), 'a')
        except:
            pass

    def createCommandFiles(self):
        if not isinstance(self.commandFiles, dict) or self.commandFiles == {}:
            # print('createCommandFiles is creating new command files!')
            nruns, seed, elementErrors, elementScan = self.processRunSettings()
            self.commandFiles['global_settings'] = elegant_global_settings_command(lattice=self, warning_limit=0)
            self.commandFiles['run_setup'] = elegant_run_setup_command(lattice=self, p_central=np.mean(self.global_parameters['beam'].BetaGamma),  seed=seed, losses="%s.loss")

            # generate commands for monte carlo jitter runs
            if (elementErrors is not None):
                self.commandFiles['run_control'] = elegant_run_control_command(lattice=self, n_steps=nruns, n_passes=1, reset_rf_for_each_step=0, first_is_fiducial=1)
                self.commandFiles['error_elements'] = elegant_error_elements_command(lattice=self, elementErrors=elementErrors, nruns=nruns)
                for e in elementErrors:
                    for item in elementErrors[e]:
                        self.commandFiles['error_element_'+e] = elegantCommandFile(objecttype='error_element', name=e, item=item, allow_missing_elements=1, **elementErrors[e][item])

            # generate commands for parameter scans without fiducialisation (i.e. jitter scans)
            elif (elementScan is not None):
                self.commandFiles['run_control'] = elegant_run_control_command(lattice=self, n_steps=nruns, n_passes=1, n_indices=1, reset_rf_for_each_step=0, first_is_fiducial=1)
                self.commandFiles['scan_elements'] = elegant_scan_elements_command(lattice=self, elementScan=elementScan, nruns=nruns)

            # run_control for standard runs with no jitter
            else:
                self.commandFiles['run_control'] = elegant_run_control_command(lattice=self, n_steps=1, n_passes=1)
                        
            self.commandFiles['twiss_output'] = elegant_twiss_output_command(lattice=self, beam=self.global_parameters['beam'],
                betax=self.betax,
                betay=self.betay,
                alphax=self.alphax,
                alphay=self.alphay)
            self.commandFiles['floor_coordinates'] = elegant_floor_coordinates_command(lattice=self)
            self.commandFiles['matrix_output'] = elegant_matrix_output_command(lattice=self)
            self.commandFiles['sdds_beam'] = elegant_sdds_beam_command(
                lattice=self, elegantbeamfilename=self.objectname+'.sdds', sample_interval=self.sample_interval,
                reuse_bunch=1, fiducialization_bunch=0, center_arrival_time=0
                )
            self.commandFiles['track'] = elegant_track_command(lattice=self, trackBeam=self.trackBeam)
            self.commandFilesOrder = list(self.commandFiles.keys())#['global_settings', 'run_setup', 'error_elements', 'scan_elements', 'run_control', 'twiss', 'sdds_beam', 'track']

    def preProcess(self):
        prefix = self.file_block['input']['prefix'] if 'input' in self.file_block and 'prefix' in self.file_block['input'] else ''
        if self.trackBeam:
            self.hdf5_to_sdds(prefix)
        else:
            HDF5filename = prefix+self.particle_definition+'.hdf5'
            rbf.hdf5.read_HDF5_beam_file(self.global_parameters['beam'], os.path.abspath(self.global_parameters['master_subdir'] + '/' + HDF5filename))
        self.createCommandFiles()

    @lox.thread
    def screen_threaded_function(self, screen, sddsindex):
        return screen.sdds_to_hdf5(sddsindex)

    def postProcess(self):
        if self.trackBeam:
            for i,s in enumerate(self.screens_and_bpms):
                self.screen_threaded_function.scatter(s, i)
            if self.w is not None and not self.w['output_filename'].lower() in [s['output_filename'].lower() for s in self.screens_and_bpms]:
                self.screen_threaded_function.scatter(self.w, len(self.screens_and_bpms))
        results = self.screen_threaded_function.gather()
        self.commandFiles = {}

    def hdf5_to_sdds(self, prefix='', write=True):
        HDF5filename = prefix+self.particle_definition+'.hdf5'
        rbf.hdf5.read_HDF5_beam_file(self.global_parameters['beam'], os.path.abspath(self.global_parameters['master_subdir'] + '/' + HDF5filename))
        # print('HDF5 Total charge', self.global_parameters['beam']['total_charge'])
        if self.bunch_charge is not None:
            self.q = charge(name='START', type='charge', global_parameters=self.global_parameters,**{'total': abs(self.bunch_charge)})
        else:
            self.q = charge(name='START', type='charge', global_parameters=self.global_parameters,**{'total': abs(self.global_parameters['beam'].Q)})
        # print('mean cpz = ', np.mean(self.global_parameters['beam'].cpz), ' prefix = ', prefix)
        # shutil.copyfile(self.global_parameters['master_subdir'] + '/' + HDF5filename, self.global_parameters['master_subdir'] + '/' + self.objectname+'.hdf5') #copy src to dst
        sddsbeamfilename = self.objectname+'.sdds'
        if write:
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

    def elegantCommandFile(self, *args, **kwargs):
        return elegantCommandFile(*args, **kwargs)

class elegantCommandFile(frameworkCommand):
    def __init__(self, objectname=None, lattice='', *args, **kwargs):
        super().__init__(objectname=objectname, *args, **kwargs)
        self.lattice = lattice
        self.write = self.write_Elegant

class elegant_global_settings_command(elegantCommandFile):
    def __init__(self, lattice='', **kwargs):
        super().__init__(objecttype='global_settings', lattice=lattice, **kwargs)
        self.add_properties(inhibit_fsync=0, mpi_io_force_file_sync=0, mpi_io_read_buffer_size=16777216, \
                            mpi_io_write_buffer_size=16777216, usleep_mpi_io_kludge=0, **kwargs)

class elegant_run_setup_command(elegantCommandFile):
    def __init__(self, lattice='', p_central=0, seed=0, always_change_p0 = 1, default_order=3, **kwargs):
        super().__init__(objecttype='run_setup', lattice=lattice, **kwargs)
        self.lattice_filename = lattice.objectname+'.lte'
        self.add_properties(lattice=self.lattice_filename, \
            use_beamline=lattice.objectname, p_central=p_central, \
            random_number_seed=seed, \
            centroid='%s.cen',always_change_p0 = always_change_p0, \
            sigma='%s.sig', default_order=default_order, **kwargs)

class elegant_error_elements_command(elegantCommandFile):
    # build commands for randomised errors on specified elements
    def __init__(self, lattice='', elementErrors=None, nruns=1, **kwargs):
        super().__init__(objecttype='error_control', lattice=lattice, **kwargs)
        self.add_properties(objecttype='error_control', no_errors_for_first_step=1, error_log='%s.erl')

class elegant_scan_elements_command(elegantCommandFile):
    # build command for a systematic parameter scan
    def __init__(self, lattice='', elementScan=None, nruns=1, index_number=0, **kwargs):
        super().__init__(objecttype='vary_element', lattice=lattice, **kwargs)
        self.add_properties(objecttype='vary_element', index_number=0, **elementScan)

class elegant_run_control_command(elegantCommandFile):
    def __init__(self, lattice='', **kwargs):
        super().__init__(objecttype='run_control', lattice=lattice, **kwargs)
        self.add_properties(**kwargs)

class elegant_twiss_output_command(elegantCommandFile):
    # build command for a systematic parameter scan
    def __init__(self, lattice='', beam=None, betax=None, betay=None, alphax=None, alphay=None, etax=None, etaxp=None, **kwargs):
        super().__init__(objecttype='twiss_output', lattice=lattice, **kwargs)
        self.betax = betax if betax is not None else beam.twiss.beta_x_corrected
        self.betay = betay if betay is not None else beam.twiss.beta_y_corrected
        self.alphax = alphax if alphax is not None else beam.twiss.alpha_x_corrected
        self.alphay = alphay if alphay is not None else beam.twiss.alpha_y_corrected
        self.etax = etax if etax is not None else beam.twiss.eta_x
        self.etaxp = etaxp if etaxp is not None else beam.twiss.eta_xp

        self.add_properties(matched = 0,output_at_each_step=0,radiation_integrals=1,statistics=1,filename="%s.twi",
                        beta_x  = self.betax,
                        alpha_x = self.alphax,
                        beta_y  = self.betay,
                        alpha_y = self.alphay,
                        eta_x = self.etax,
                        etap_x = self.etaxp, **kwargs)

class elegant_floor_coordinates_command(elegantCommandFile):
    def __init__(self, lattice='', **kwargs):
        super().__init__(objecttype='floor_coordinates', lattice=lattice, **kwargs)
        flr = self.add_properties(filename="%s.flr",
                        X0  = lattice.startObject['position_start'][0],
                        Z0 = lattice.startObject['position_start'][2],
                        theta0 = 0,
                        magnet_centers = 0, **kwargs)

class elegant_matrix_output_command(elegantCommandFile):
    def __init__(self, lattice='', **kwargs):
        super().__init__(objecttype='matrix_output', lattice=lattice, **kwargs)
        mat = self.add_properties(SDDS_output="%s.mat",
                        full_matrix_only=0, SDDS_output_order=2, **kwargs)

class elegant_sdds_beam_command(elegantCommandFile):
    def __init__(self, lattice='', elegantbeamfilename='', **kwargs):
        super().__init__(objecttype='sdds_beam', lattice=lattice, **kwargs)
        self.elegantbeamfilename = elegantbeamfilename
        self.add_properties(input=self.elegantbeamfilename, **kwargs)

class elegant_track_command(elegantCommandFile):
    def __init__(self, lattice='', trackBeam=True, **kwargs):
        super().__init__(objecttype='track', lattice=lattice, **kwargs)
        if trackBeam:
            self.add_properties(**kwargs)

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

class sddsFile(object):
    '''simple class for writing generic column data to a new SDDS file'''

    def __init__(self):
        '''initialise an SDDS instance, prepare for writing to file'''
        self.sdds = sdds.SDDS(0)

    def add_column(self, name, data, **kwargs):
        '''add a column of floating point numbers to the file'''
        if not isinstance(name, str):
            raise TypeError('Column names must be string types')
        self.sdds.defineColumn(name,
                               symbol=kwargs['symbol'] if ('symbol' in kwargs) else '',
                               units=kwargs['units'] if ('units' in kwargs) else '',
                               description=kwargs['description'] if ('description' in kwargs) else '',
                               formatString='', type=self.sdds.SDDS_DOUBLE, fieldLength=0)

        if isinstance(data, (tuple, list, np.ndarray)):
            self.sdds.setColumnValueList(name, list(data), page=1)
        else:
            raise TypeError('Column data must be a list, tuple or array-like type')

    def save(self, fname):
        '''save the sdds data structure to file'''
        if not isinstance(fname, str):
            raise TypeError('SDDS file name must be a string!')
        self.sdds.save(fname)
