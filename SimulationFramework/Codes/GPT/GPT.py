from ...Framework_objects import *
from ...Framework_elements import *
from ...FrameworkHelperFunctions import _rotation_matrix
from ...Modules import Beams as rbf

gpt_defaults = {}

class gptLattice(frameworkLattice):
    def __init__(self, *args, **kwargs):
        super(gptLattice, self).__init__(*args, **kwargs)
        self.code = 'gpt'
        self.bunch_charge = None
        # self.particle_definition = self.allElementObjects[self.start].objectname
        self.headers = OrderedDict()
        if 'particle_definition' in self.file_block['input'] and self.file_block['input']['particle_definition'] == 'initial_distribution':
            self.particle_definition = 'laser'
        else:
            self.particle_definition = self.allElementObjects[self.start].objectname
        self.headers['setfile'] = gpt_setfile(set="\"beam\"", filename="\"" + self.particle_definition + ".gdf\"")
        self.headers['floorplan'] = gpt_writefloorplan(filename="\"" + self.objectname + "_floor.gdf\"")
        # self.headers['settotalcharge'] = gpt_charge(set="\"beam\"", charge=250e-12)
        start = self.allElementObjects[self.start].start[2]
        end = self.allElementObjects[self.end].end[2]
        self.ignore_start_screen = None
        self.screen_step_size = 0.1
        self.time_step_size = '0.1/c'
        # self.headers['tout'] = gpt_tout(startpos=0, endpos=self.allElementObjects[self.end].end[2], step=0.1)

    def endScreen(self, **kwargs):
        return screen(name=self.endObject.objectname, type='screen', position_start=self.endObject.position_start, position_end=self.endObject.position_start, global_rotation=self.endObject.global_rotation, global_parameters=self.global_parameters, **kwargs)

    def writeElements(self):
        ccs = gpt_ccs("wcs", [0,0,0], [0,0,0])
        fulltext = ''
        if self.particle_definition == 'laser' and self.file_block['charge']['space_charge_mode'] is not None:
            self.file_block['charge']['space_charge_mode'] = 'cathode'
            self.headers['spacecharge'] = gpt_spacecharge(**merge_two_dicts(self.global_parameters, self.file_block['charge']))
            self.headers['spacecharge'].npart = len(self.global_parameters['beam'].x)
            self.headers['spacecharge'].sample_interval = self.sample_interval
            # self.headers['dtmaxt'] = gpt_dtmaxt(tstart=0, tend="0.25/c", dtmax="0.25/1000/c")
        else:
            self.headers['spacecharge'] = gpt_spacecharge(**merge_two_dicts(self.global_parameters, self.file_block['charge']))
        if self.csr_enable and not os.name == 'nt':
            self.headers['csr1d'] = gpt_csr1d()
        # self.headers['forwardscatter'] = gpt_forwardscatter(ECS='"wcs", "I"', name='cathode', probability=0)
        # self.headers['scatterplate'] = gpt_scatterplate(ECS='"wcs", "z", -1e-6', model='cathode', a=1, b=1)
        for header in self.headers:
            fulltext += self.headers[header].write_GPT()
        for i, element in enumerate(list(self.elements.values())):
            if i ==0:
                screen0pos = element.start[2]
                ccs = element.gpt_ccs(ccs)
            if i == 0 and isinstance(element, screen):
                self.ignore_start_screen = element
            else:
                fulltext += element.write_GPT(self.Brho, ccs=ccs)
                new_ccs = element.gpt_ccs(ccs)
                if not new_ccs == ccs:
                    # print('ccs = ', ccs, '  new_ccs = ', new_ccs)
                    relpos, relrot = ccs.relative_position(element.end, element.global_rotation)
                    if self.particle_definition == 'laser':
                        fulltext += 'screen( ' + ccs.name + ', "I", '+ str(screen0pos+self.screen_step_size) + ', ' + str(relpos[2]) + ', ' + str(self.screen_step_size) + ');\n'
                    else:
                        fulltext += 'screen( ' + ccs.name + ', "I", '+ str(screen0pos+self.screen_step_size) + ', ' + str(relpos[2]) + ', ' + str(self.screen_step_size) + ');\n'
                    screen0pos = 0
                ccs = new_ccs
        if not isinstance(element, screen):
            element = self.endScreenObject = self.endScreen()
            fulltext += self.endScreenObject.write_GPT(self.Brho, ccs=ccs)
        else:
            # print('End screen', element.objectname)
            self.endScreenObject = None
        relpos, relrot = ccs.relative_position(element.position_end, element.global_rotation)
        if self.particle_definition == 'laser':
            fulltext += 'screen( ' + ccs.name + ', "I", '+ str(screen0pos+self.screen_step_size) + ', ' + str(relpos[2]) + ', ' + str(self.screen_step_size) + ');\n'
        else:
            fulltext += 'screen( ' + ccs.name + ', "I", '+ str(screen0pos+self.screen_step_size) + ', ' + str(relpos[2]) + ', ' + str(self.screen_step_size) + ');\n'
        return fulltext

    def write(self):
        self.code_file = self.global_parameters['master_subdir']+'/'+self.objectname+'.in'
        saveFile(self.code_file, self.writeElements())
        return self.writeElements()

    def preProcess(self):
        self.headers['setfile'].particle_definition = self.particle_definition
        prefix = self.file_block['input']['prefix'] if 'input' in self.file_block and 'prefix' in self.file_block['input'] else ''
        self.hdf5_to_gdf(prefix)

    def run(self):
        """Run the code with input 'filename'"""
        command = self.executables[self.code] + ['-o', self.objectname+'_out.gdf'] + ['GPTLICENSE='+self.global_parameters['GPTLICENSE']] + [self.objectname+'.in']
        my_env = os.environ.copy()
        my_env["LD_LIBRARY_PATH"] = my_env["LD_LIBRARY_PATH"] + ":/opt/GPT3.3.6/lib/" if "LD_LIBRARY_PATH" in my_env else "/opt/GPT3.3.6/lib/"
        my_env["OMP_WAIT_POLICY"] = "PASSIVE"
        # post_command_t = [self.executables[self.code][0].replace('gpt.exe','gdfa.exe')] + ['-o', self.objectname+'_emit.gdf'] + [self.objectname+'_out.gdf'] + ['time','avgx','avgy','stdx','stdBx','stdy','stdBy','stdz','stdt','nemixrms','nemiyrms','nemizrms','numpar','nemirrms','avgG','avgp','stdG','avgt','avgBx','avgBy','avgBz','CSalphax','CSalphay','CSbetax','CSbetay']
        post_command = [self.executables[self.code][0].replace('gpt','gdfa')] + ['-o', self.objectname+'_emit.gdf'] + [self.objectname+'_out.gdf'] + ['position','avgx','avgy','avgz','stdx','stdBx','stdy','stdBy','stdz','stdt','nemixrms','nemiyrms','nemizrms','numpar','nemirrms','avgG','avgp','stdG','avgt','avgBx','avgBy','avgBz','CSalphax','CSalphay','CSbetax','CSbetay']
        post_command_t = [self.executables[self.code][0].replace('gpt','gdfa')] + ['-o', self.objectname+'_emitt.gdf'] + [self.objectname+'_out.gdf'] + ['time', 'avgx','avgy','avgz','stdx','stdBx','stdy','stdBy','stdz','nemixrms','nemiyrms','nemizrms','numpar','nemirrms','avgG','avgp','stdG','avgBx','avgBy','avgBz','CSalphax','CSalphay','CSbetax','CSbetay','avgfBx','avgfEx','avgfBy','avgfEy','avgfBz','avgfEz']
        post_command_traj = [self.executables[self.code][0].replace('gpt','gdfa')] + ['-o', self.objectname+'traj.gdf'] + [self.objectname+'_out.gdf'] + ['time','avgx','avgy','avgz']
        with open(os.path.relpath(self.global_parameters['master_subdir']+'/'+self.objectname+'.log', '.'), "w") as f:
            # print('gpt command = ', command)
            subprocess.call(command, stdout=f, cwd=self.global_parameters['master_subdir'], env=my_env)
            subprocess.call(post_command, stdout=f, cwd=self.global_parameters['master_subdir'])
            subprocess.call(post_command_t, stdout=f, cwd=self.global_parameters['master_subdir'])
            subprocess.call(post_command_traj, stdout=f, cwd=self.global_parameters['master_subdir'])

    def postProcess(self):
        for e in self.screens_and_bpms:
            if not e == self.ignore_start_screen:
                e.gdf_to_hdf5(self.objectname + '_out.gdf')
            # else:
                # print('Ignoring', self.ignore_start_screen.objectname)
        if self.endScreenObject is not None:
            self.endScreenObject.gdf_to_hdf5(self.objectname + '_out.gdf')

    def hdf5_to_gdf(self, prefix=''):
        HDF5filename = prefix+self.particle_definition+'.hdf5'
        rbf.hdf5.read_HDF5_beam_file(self.global_parameters['beam'], self.global_parameters['master_subdir'] + '/' + HDF5filename)
        # print('beam charge = ', self.global_parameters['beam'].charge)
        if self.sample_interval > 1:
            self.headers['setreduce'] = gpt_setreduce(set="\"beam\"", setreduce=len(self.global_parameters['beam'].x)/self.sample_interval)
        # self.headers['settotalcharge'] = gpt_charge(set="\"beam\"", charge=self.global_parameters['beam'].charge)
        meanBz = np.mean(self.global_parameters['beam'].Bz)
        if meanBz < 0.5:
            meanBz = 0.75
        # print(self.findS(self.end), self.findS(self.start))
        self.headers['tout'] = gpt_tout(starttime=0, endpos=(self.findS(self.end)[0][1]-self.findS(self.start)[0][1])/meanBz/2.998e8, step=str(self.time_step_size))
        gdfbeamfilename = self.particle_definition+'.gdf'
        cathode = self.particle_definition == 'laser'
        rbf.gdf.write_gdf_beam_file(self.global_parameters['beam'], self.global_parameters['master_subdir'] + '/' + self.particle_definition+'.txt', normaliseX=self.allElementObjects[self.start].start[0], cathode=cathode)
        subprocess.call([self.executables[self.code][0].replace('gpt','asci2gdf'), '-o', gdfbeamfilename, self.particle_definition+'.txt'], cwd=self.global_parameters['master_subdir'])
        self.Brho = self.global_parameters['beam'].Brho

class gpt_element(frameworkElement):

    def __init__(self, elementName=None, elementType=None, **kwargs):
        super(gpt_element, self).__init__(elementName, elementType, **kwargs)
        # if elementName in gpt_defaults:
        #     for k, v in list(gpt_defaults[elementName].items()):
        #         self.add_default(k, v)

    def write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + '('
        for k in elementkeywords[self.objecttype]['keywords']:
            k = k.lower()
            if getattr(self,k) is not None:
                output += str(getattr(self, k))+', '
            elif k in self.objectdefaults :
                output += self.objectdefaults[k]+', '
        output = output[:-2]
        output+=');\n'
        return output

class gpt_setfile(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_setfile, self).__init__(elementName='setfile', elementType='gpt_setfile', **kwargs)

    def hdf5_to_gpt(self, prefix=''):
        HDF5filename = prefix+self.particle_definition.replace('.gdf','')+'.hdf5'
        rbf.hdf5.read_HDF5_beam_file(self.global_parameters['beam'], self.global_parameters['master_subdir'] + '/' + HDF5filename)
        # self.global_parameters['beam'].rotate_beamXZ(self.theta, preOffset=self.starting_offset)
        gptbeamfilename = self.particle_definition
        rbf.gdf.write_gdf_beam_file(self.global_parameters['beam'], self.global_parameters['master_subdir'] + '/' + gptbeamfilename, normaliseZ=False)

class gpt_charge(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_charge, self).__init__(elementName='settotalcharge', elementType='gpt_charge', **kwargs)

    def write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + '('
        output += str(self.set) + ','
        output += str(-1*abs(self.charge)) + ');\n'
        return output

class gpt_setreduce(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_setreduce, self).__init__(elementName='setreduce', elementType='gpt_setreduce', **kwargs)

    def write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + '('
        output += str(self.set) + ','
        output += str(self.setreduce) + ');\n'
        return output

class gpt_spacecharge(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_spacecharge, self).__init__(elementName='spacecharge', elementType='gpt_spacecharge', **kwargs)
        self.grids = getGrids()
        self.add_default('accuracy', 6)
        self.add_default('ngrids', None)

    def write_GPT(self, *args, **kwargs):
        output = 'accuracy(' + str(self.accuracy) + ');\n'#'setrmacrodist(\"beam\","u",1e-9,0) ;\n'
        if isinstance(self.space_charge_mode,str) and self.space_charge_mode.lower() == 'cathode':
            if self.ngrids is None:
                self.ngrids = self.grids.getGridSizes((self.npart/self.sample_interval))
            output += 'spacecharge3Dmesh("Cathode","RestMaxGamma",1000);\n'
        elif isinstance(self.space_charge_mode,str) and self.space_charge_mode.lower() == '3d':
            output += 'Spacecharge3Dmesh();\n'
        elif isinstance(self.space_charge_mode,str) and self.space_charge_mode.lower() == '2d':
            output += 'sc3dmesh();\n'
        else:
            output = ''
        return output

class gpt_tout(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_tout, self).__init__(elementName='tout', elementType='gpt_tout', **kwargs)

    def write_GPT(self, *args, **kwargs):
        self.starttime = 0 if self.starttime < 0 else self.starttime
        output = str(self.objectname) + '('
        if self.starttime is not None:
            output += str(self.starttime) + ','
        else:
            output += str(self.startpos) + '/c,'
        output += str(self.endpos) + ','
        output += str(self.step) + ');\n'
        return output

class gpt_csr1d(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_csr1d, self).__init__(elementName='csr1d', elementType='gpt_csr1d', **kwargs)

    def write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + '();\n'
        return output

class gpt_writefloorplan(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_writefloorplan, self).__init__(elementName='writefloorplan', elementType='gpt_writefloorplan', **kwargs)

    def write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + '(' + self.filename + ');\n'
        return output

class gpt_Zminmax(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_Zminmax, self).__init__(elementName='Zminmax', elementType='gpt_Zminmax', **kwargs)

    def write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + '(' + self.ECS + ', ' + str(self.zmin) + ', ' + str(self.zmax) + ');\n'
        return output

class gpt_forwardscatter(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_forwardscatter, self).__init__(elementName='forwardscatter', elementType='gpt_forwardscatter', **kwargs)

    def write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + '(' + self.ECS + ', \"' + str(self.name) + '\", ' + str(self.probability) + ');\n'
        return output

class gpt_scatterplate(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_scatterplate, self).__init__(elementName='scatterplate', elementType='gpt_scatterplate', **kwargs)

    def write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + '(' + self.ECS + ', ' + str(self.a) + ', ' + str(self.b) + ') scatter=\"' + str(self.model) + '\";\n'
        return output

class gpt_dtmaxt(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_dtmaxt, self).__init__(elementName='dtmaxt', elementType='gpt_dtmaxt', **kwargs)

    def write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + '(' + str(self.tstart) + ', ' + str(self.tend) + ', ' + str(self.dtmax) + ');\n'
        return output
