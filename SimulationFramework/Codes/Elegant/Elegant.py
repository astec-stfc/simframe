import os, sdds, shutil
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
            saveFile(self.command_file, self.commandFile.write())
        except:
            pass

    def preProcess(self):
        prefix = self.file_block['input']['prefix'] if 'input' in self.file_block and 'prefix' in self.file_block['input'] else ''
        if self.trackBeam:
            self.hdf5_to_sdds(prefix)
        nruns, seed, elementErrors, elementScan = self.processRunSettings()
        self.commandFile = elegantTrackFile(lattice=self, trackBeam=self.trackBeam, elegantbeamfilename=self.objectname+'.sdds', sample_interval=self.sample_interval,
        betax=self.betax,
        betay=self.betay,
        alphax=self.alphax,
        alphay=self.alphay,
        global_parameters=self.global_parameters,
        nruns=nruns,
        seed=seed,
        elementErrors=elementErrors,
        elementScan=elementScan
        )

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
        # shutil.copyfile(self.global_parameters['master_subdir'] + '/' + HDF5filename, self.global_parameters['master_subdir'] + '/' + self.objectname+'.hdf5') #copy src to dst
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
    def __init__(self, lattice='', trackBeam=True, elegantbeamfilename='', betax=None, betay=None, alphax=None, alphay=None, etax=None, etaxp=None, \
                 nruns=1, seed=0, elementErrors=None, elementScan=None, *args, **kwargs):
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
            self.addCommand(objecttype='global_settings', inhibit_fsync=0, mpi_io_force_file_sync=0, mpi_io_read_buffer_size=16777216, \
                            mpi_io_write_buffer_size=16777216, usleep_mpi_io_kludge=0)
        self.addCommand(objecttype='run_setup',lattice=self.lattice_filename, \
            use_beamline=lattice.objectname,p_central=np.mean(self.global_parameters['beam'].BetaGamma), \
            random_number_seed=seed, \
            centroid='%s.cen',always_change_p0 = 1, \
            sigma='%s.sig', default_order=3)

        # build commands for randomised errors on specified elements
        if (elementErrors is not None):
            self.addCommand(objecttype='run_control', n_steps=nruns, n_passes=1, reset_rf_for_each_step=0, first_is_fiducial=1)
            self.addCommand(objecttype='error_control', no_errors_for_first_step=1, error_log='%s.erl')
            for e in elementErrors:
                for item in elementErrors[e]:
                    self.addCommand(objecttype='error_element', name=e, item=item, allow_missing_elements=1, **elementErrors[e][item])
        # build command for a systematic parameter scan
        elif (elementScan is not None):
            self.addCommand(objecttype='run_control', n_steps=nruns, n_passes=1, n_indices=1, reset_rf_for_each_step=0, first_is_fiducial=1)
            self.addCommand(objecttype='vary_element', index_number=0, **elementScan)
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
        theta0 = 0,
        magnet_centers = 0)
        mat = self.addCommand(objecttype='matrix_output', SDDS_output="%s.mat",
        full_matrix_only=0, SDDS_output_order=2)
        if self.trackBeam:
            if (elementErrors is not None) or (elementScan is not None):
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
