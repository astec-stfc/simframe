import time, os, subprocess, re
import yaml
import copy
from pprint import pprint
from collections import OrderedDict
from .Modules.merge_two_dicts import merge_two_dicts
from .Modules import Beams as rbf
from .Modules import Twiss as rtf
from .Codes import Executables as exes
from .Codes.ASTRA.ASTRA import *
from .Codes.CSRTrack.CSRTrack import *
from .Codes.Elegant.Elegant import *
from .Codes.Generators.Generators import *
from .Codes.GPT.GPT import *
try:
    import MasterLattice
    MasterLatticeLocation = os.path.dirname(MasterLattice.__file__)+'/'
except:
    MasterLatticeLocation = None
try:
    import SimulationFramework.Modules.plotting as groupplot
    use_matplotlib = True
except ImportError as e:
    print('Import error - plotting disabled. Missing package:', e)
    use_matplotlib = False
import progressbar
from munch import Munch, unmunchify

_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

def dict_representer(dumper, data):
    return dumper.represent_dict(iter(list(data.items())))

def dict_constructor(loader, node):
    return OrderedDict(loader.construct_pairs(node))

yaml.add_representer(OrderedDict, dict_representer)
yaml.add_constructor(_mapping_tag, dict_constructor)

class Framework(Munch):

    def __init__(self, directory='test', master_lattice=None, overwrite=None, runname='CLARA_240', clean=False, verbose=True, sddsindex=0, delete_output_files=False):
        super(Framework, self).__init__()
        gptlicense = os.environ['GPTLICENSE'] if 'GPTLICENSE' in os.environ else ''
        astra_use_wsl = os.environ['WSL_ASTRA'] if 'WSL_ASTRA' in os.environ else 1
        self.global_parameters = {'beam': rbf.beam(sddsindex=sddsindex), 'GPTLICENSE': gptlicense, 'delete_tracking_files': delete_output_files, 'astra_use_wsl': astra_use_wsl}
        self.verbose = verbose
        self.subdir = directory
        self.clean = clean
        self.elementObjects = OrderedDict()
        self.latticeObjects = OrderedDict()
        self.commandObjects = OrderedDict()
        self.groupObjects = OrderedDict()
        self.progress = 0
        self.basedirectory = os.getcwd()
        self.filedirectory = os.path.dirname(os.path.abspath(__file__))
        self.overwrite = overwrite
        self.runname = runname
        if self.subdir is not None:
            self.setSubDirectory(self.subdir)
        self.setMasterLatticeLocation(master_lattice)

        self.executables = exes.Executables(self.global_parameters)
        self.defineASTRACommand = self.executables.define_astra_command
        self.defineElegantCommand = self.executables.define_elegant_command
        self.defineASTRAGeneratorCommand = self.executables.define_ASTRAgenerator_command
        self.defineCSRTrackCommand = self.executables.define_csrtrack_command
        self.define_gpt_command = self.executables.define_gpt_command

    def __repr__(self):
        return repr({'master_lattice_location': self.global_parameters['master_lattice_location'], 'subdirectory': self.subdirectory, 'settingsFilename': self.settingsFilename})

    def setSubDirectory(self, dir):
        # global self.global_parameters['master_subdir'], self.global_parameters['master_lattice_location']
        self.subdirectory = os.path.abspath(dir)
        # print('self.subdirectory = ', self.subdirectory)
        self.global_parameters['master_subdir'] = self.subdirectory
        if not os.path.exists(self.subdirectory):
            os.makedirs(self.subdirectory, exist_ok=True)
        else:
            if self.clean == True:
                clean_directory(self.subdirectory)
        if self.overwrite == None:
            self.overwrite = True
        self.setMasterLatticeLocation()

    def setMasterLatticeLocation(self, master_lattice=None):
        # global master_lattice_location
        if master_lattice is None:
            if MasterLatticeLocation is not None:
                # print('Found MasterLattice Package =', MasterLatticeLocation)
                self.global_parameters['master_lattice_location'] = MasterLatticeLocation.replace('\\','/')
            elif os.path.isdir(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../../MasterLattice/MasterLattice/')+'/'):
                # print('Found MasterLattice Directory 2-up =', MasterLatticeLocation)
                self.global_parameters['master_lattice_location'] = (os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../../MasterLattice/MasterLattice/')+'/').replace('\\','/')
            elif os.path.isdir(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../MasterLattice/')+'/'):
                # print('Found MasterLattice Directory 1-up =', MasterLatticeLocation)
                self.global_parameters['master_lattice_location'] = (os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../MasterLattice/')+'/').replace('\\','/')
            else:
                raise Exception("Master Lattice not available - specify using master_lattice=<location>")
        else:
            self.global_parameters['master_lattice_location'] = master_lattice
        self.master_lattice_location = self.global_parameters['master_lattice_location']

    def load_Elements_File(self, input):
        if isinstance(input,(list,tuple)):
            filename = input
        else:
            filename = [input]
        for f in filename:
            if os.path.isfile(f):
                with open(f, 'r') as stream:
                    elements = yaml.safe_load(stream)['elements']
            elif os.path.isfile(self.subdirectory+'/'+f):
                with open(self.subdirectory+'/'+f, 'r') as stream:
                    elements = yaml.safe_load(stream)['elements']
            else:
                with open(self.global_parameters['master_lattice_location'] + f, 'r') as stream:
                    elements = yaml.safe_load(stream)['elements']
            for name, elem in list(elements.items()):
                self.read_Element(name, elem)

    def loadSettings(self, filename='short_240.settings'):
        """Load Lattice Settings from file"""
        self.settingsFilename = filename
        # print 'self.settingsFilename = ', self.settingsFilename
        if os.path.exists(filename):
            stream = open(filename, 'r')
        else:
            stream = open(self.global_parameters['master_lattice_location']+filename, 'r')
        self.settings = yaml.safe_load(stream)
        self.globalSettings = self.settings['global'] if 'global' in self.settings else {}
        master_run_no = self.globalSettings['run_no'] if 'run_no' in self.globalSettings else 1
        if 'generator' in self.settings:
            self.generatorSettings = self.settings['generator']
            self.add_Generator(**self.generatorSettings)
        self.fileSettings = self.settings['files'] if 'files' in self.settings else {}
        elements = self.settings['elements']
        self.groups = self.settings['groups'] if 'groups' in self.settings and self.settings['groups'] is not None else {}
        changes = self.settings['changes'] if 'changes' in self.settings and self.settings['changes'] is not None else {}
        stream.close()

        for name, elem in list(self.groups.items()):
            group = globals()[elem['type']](name, self.elementObjects, global_parameters=self.global_parameters, **elem)
            self.groupObjects[name] = group

        for name, elem in list(elements.items()):
            self.read_Element(name, elem)

        for name, lattice in list(self.fileSettings.items()):
            self.read_Lattice(name, lattice)

        self.apply_changes(changes)

        self.original_elementObjects = {}
        for e in self.elementObjects:
            self.original_elementObjects[e] = unmunchify(self.elementObjects[e])

    def save_settings(self, filename=None, directory='.', elements=None):
        if filename is None:
            pre, ext = os.path.splitext(os.path.basename(self.settingsFilename))
        else:
            pre, ext = os.path.splitext(os.path.basename(filename))
        if filename is None:
            filename =  'settings.def'
        settings = self.settings.copy()
        if elements is not None:
            settings = self.settings.copy()
            settings['elements'] = elements
        with open(directory + '/' + filename,"w") as yaml_file:
            yaml.default_flow_style=True
            yaml.dump(settings, yaml_file)

    def read_Lattice(self, name, lattice):
        code = lattice['code'] if 'code' in lattice else 'astra'
        self.latticeObjects[name] = globals()[lattice['code'].lower()+'Lattice'](name, lattice, self.elementObjects, self.groupObjects, self.settings, self.executables, self.global_parameters)

    def convert_numpy_types(self, v):
        if isinstance(v, (np.ndarray, list, tuple)):
            return [self.convert_numpy_types(l) for l in v]
        elif isinstance(v, (np.float64, np.float32, np.float16, np.float_ )):
            return float(v)
        elif isinstance(v, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(v)
        else:
            return v

    def detect_changes(self, elementtype=None, elements=None, function=None):
        disallowed = ['allowedkeywords', 'keyword_conversion_rules_elegant', 'objectdefaults', 'global_parameters']
        start = time.time()
        changedict = {}
        if elementtype is not None:
            changeelements = self.getElementType(elementtype, 'objectname')
        elif elements is not None:
            changeelements = elements
        else:
            changeelements = list(self.elementObjects.keys())
        # print('changeelements = ', changeelements)
        # print(changeelements[0])
        if len(changeelements) > 0 and isinstance(changeelements[0], (list, tuple, dict)) and len(changeelements[0]) > 1:
                for ek in changeelements:
                    new = None
                    e, k = ek[:2]
                    if e in self.elementObjects:
                        new = unmunchify(self.elementObjects[e])
                    elif e in self.groupObjects:
                        new = self.groupObjects[e]
                        # print 'detecting group = ', e, new, new[k]
                    if new is not None:
                        # print (new)
                        if e not in changedict:
                            changedict[e] = {}
                        changedict[e][k] = self.convert_numpy_types(new[k])
        else:
            for e in changeelements:
                if not self.original_elementObjects[e] == unmunchify(self.elementObjects[e]):
                    orig = self.original_elementObjects[e]
                    new = unmunchify(self.elementObjects[e])
                    try:
                        changedict[e] = {k: self.convert_numpy_types(new[k]) for k in new if k in orig and not new[k] == orig[k] and not k in disallowed}
                        changedict[e].update({k: self.convert_numpy_types(new[k]) for k in new if k not in orig and not k in disallowed})
                        if changedict[e] == {}:
                            del changedict[e]
                    except:
                        print ('##### ERROR IN CHANGE ELEMS: ', e, new)
                        pass
        return changedict

    def save_changes_file(self, filename=None, type=None, elements=None, function=None):
        if filename is None:
            pre, ext = os.path.splitext(os.path.basename(self.settingsFilename))
            filename =  pre     + '_changes.yaml'
        changedict = self.detect_changes(elementtype=type, elements=elements, function=function)
        with open(filename,"w") as yaml_file:
            yaml.default_flow_style=True
            yaml.dump(changedict, yaml_file)

    def save_lattice(self, lattice=None, filename=None, directory='.'):
        if filename is None:
            pre, ext = os.path.splitext(os.path.basename(self.settingsFilename))
        else:
            pre, ext = os.path.splitext(os.path.basename(filename))
        dic = OrderedDict({'elements': OrderedDict()})
        latticedict = dic['elements']
        if lattice is None:
            elements = list(self.elementObjects.keys())
            filename =  pre     + '.yaml'
        else:
            if self.latticeObjects[lattice].elements is None:
                return
            elements = list(self.latticeObjects[lattice].elements.keys())
            filename =  pre + '_' + lattice + '_lattice.yaml'
        disallowed = ['allowedkeywords', 'keyword_conversion_rules_elegant', 'objectdefaults', 'global_parameters', 'objectname', 'subelement']
        for e in elements:
            new = unmunchify(self.elementObjects[e])
            try:
                if ('subelement' in new and not new['subelement']) or not 'subelement' in new:
                    latticedict[e] = {k.replace('object',''): self.convert_numpy_types(new[k]) for k in new if not k in disallowed}
                    if 'sub_elements' in new:
                        for subelem in new['sub_elements']:
                            newsub = self.elementObjects[subelem]
                            latticedict[e]['sub_elements'][subelem] = {k.replace('object',''): self.convert_numpy_types(newsub[k]) for k in newsub if not k in disallowed}
            except:
                print ('##### ERROR IN CHANGE ELEMS: ', e, new)
                pass
        # print('#### Saving Lattice - ', filename)
        with open(directory + '/' + filename,"w") as yaml_file:
            yaml.default_flow_style = True
            yaml.dump(dic, yaml_file)

    def load_changes_file(self, filename=None, apply=True):
        if isinstance(filename, (tuple, list)):
            for c in filename:
                self.load_changes_file(c)
        else:
            if filename is None:
                pre, ext = os.path.splitext(os.path.basename(self.settingsFilename))
                filename =  pre     + '_changes.yaml'
            with open(filename,"r") as infile:
                changes = dict(yaml.safe_load(infile))
            if apply:
                self.apply_changes(changes)
            else:
                return changes

    def apply_changes(self, changes):
        for e, d in list(changes.items()):
            # print 'found change element = ', e
            if e in self.elementObjects:
                # print 'change element exists!'
                for k, v in list(d.items()):
                    self.modifyElement(e, k, v)
                    # print ('modifying ',e,'[',k,']', ' = ', v)
            if e in self.groupObjects:
                # print ('change group exists!')
                for k, v in list(d.items()):
                    self.groupObjects[e].change_Parameter(k, v)
                    # print ('modifying ',e,'[',k,']', ' = ', v)

    def check_lattice(self, decimals=4):
        for elem in self.elementObjects.values():
            start = elem.position_start
            end = elem.position_end
            length = elem.length
            theta = elem.global_rotation[2]
            if elem.objecttype == 'dipole':
                angle = elem.angle
                rho = length / angle
                clength = np.array([rho * (np.cos(angle) - 1), 0, rho * np.sin(angle)])
            else:
                clength = np.array([0, 0, length])
            cend = start + np.dot(clength, _rotation_matrix(theta))
            if not np.round(cend - end, decimals=decimals).any() == 0:
                print (elem.objectname, cend - end)

    def change_Lattice_Code(self, name, code, exclude=None):
        if name == 'All':
            [self.change_Lattice_Code(l, code, exclude) for l in self.latticeObjects]
        elif isinstance(name, (tuple, list)):
            [self.change_Lattice_Code(l, code, exclude) for l in name]
        else:
            if not name == 'generator' and not (name == exclude or (isinstance(exclude, (list, tuple)) and name in exclude)):
                # print('Changing lattice ', name, ' to ', code.lower())
                currentLattice = self.latticeObjects[name]
                self.latticeObjects[name] = globals()[code.lower()+'Lattice'](currentLattice.objectname, currentLattice.file_block, self.elementObjects, self.groupObjects, self.settings, self.executables, self.global_parameters)

    def read_Element(self, name, element, subelement=False):
        if name == 'filename':
            self.load_Elements_File(element)
        else:
            if subelement:
                if 'subelement' in element:
                    del element['subelement']
                self.add_Element(name, subelement=True, **element)
            else:
                self.add_Element(name, **element)
            if 'sub_elements' in element:
                for name, elem in list(element['sub_elements'].items()):
                    self.read_Element(name, elem, subelement=True)

    def add_Element(self, name=None, type=None, **kwargs):
        if name == None:
            if not 'name' in kwargs:
                raise NameError('Element does not have a name')
            else:
                name = kwargs['name']
        # try:
        element = globals()[type](name, type, global_parameters=self.global_parameters, **kwargs)
        # print element
        self.elementObjects[name] = element
        return element
        # except Exception as e:
        #     raise NameError('Element \'%s\' does not exist' % type)

    def getElement(self, element, param=None):
        if self.__getitem__(element) is not None:
            if param is not None:
                param = param.lower()
                return getattr(self.__getitem__(element), param)
            else:
                return self.__getitem__(element)
        else:
            print(( 'WARNING: Element ', element,' does not exist'))
            return {}

    def getElementType(self, type, setting=None):
        if isinstance(type, (list, tuple)):
            all_elements = [self.getElementType(t) for t in type]
            return [item for sublist in all_elements for item in sublist]
        return [self.elementObjects[element] if setting is None else self.elementObjects[element][setting] for element in list(self.elementObjects.keys()) if self.elementObjects[element].objecttype.lower() == type.lower()]

    def setElementType(self, type, setting, values):
        elems = self.getElementType(type)
        if len(elems) == len(values):
            for e, v  in zip(elems, values):
                e[setting] = v
        else:
            # print(( len(elems), len(values)))
            raise ValueError

    def modifyElement(self, elementName, parameter, value):
        if elementName in self.groupObjects:
            self.groupObjects[elementName].change_Parameter(parameter,value)
        elif elementName in self.elementObjects:
            setattr(self.elementObjects[elementName], parameter, value)
        # set(getattr(self.elementObjects[elementName], parameter), value)

    def add_Generator(self, default=None, **kwargs):
        if 'code' in generator_keywords:
            if generator_keywords['code'].lower() == "gpt":
                code = GPTGenerator
            else:
                code = ASTRAGenerator
        else:
            code = ASTRAGenerator
        if default in generator_keywords['defaults']:
            self.generator = code(self.executables, self.global_parameters, **merge_two_dicts(kwargs, generator_keywords['defaults'][default]))
        else:
            self.generator = code(self.executables, self.global_parameters, **kwargs)
        self.latticeObjects['generator'] = self.generator

    def change_generator(self, generator):
        old_kwargs = self.generator.kwargs
        if generator.lower() == "gpt":
            generator = GPTGenerator(self.executables, self.global_parameters, **old_kwargs)
        else:
            generator = ASTRAGenerator(self.executables, self.global_parameters, **old_kwargs)
        self.latticeObjects['generator'] = generator

    def loadParametersFile(self, file):
        pass

    def saveParametersFile(self, file, parameters):
        output = {}
        if isinstance(parameters, dict):
            for k,v in list(parameters.items()):
                output[k] = {}
                if isinstance(v, (list, tuple)):
                    for p in v:
                        output[k][p] = getattr(self[k],p)
                else:
                    output[k][v] = getattr(self[k],v)
        elif isinstance(parameters, (list,tuple)):
            for k, v in parameters:
                output[k] = {}
                if isinstance(v, (list, tuple)):
                    for p in v:
                        output[k][p] = getattr(self[k],p)
                else:
                    output[k][v] = getattr(self[k],v)
        with open(file,"w") as yaml_file:
            yaml.default_flow_style = True
            yaml.dump(output, yaml_file)
        # try:
        # elem = self.getelement(k, v)
        # outputfile.write(k+' '+v+' ')

    def set_lattice_prefix(self, lattice, prefix):
        if lattice in self.latticeObjects:
            self.latticeObjects[lattice].prefix = prefix

    def __getitem__(self,key):
        if key in super(Framework, self).__getitem__('elementObjects'):
            return self.elementObjects.get(key)
        elif key in super(Framework, self).__getitem__('latticeObjects'):
            return self.latticeObjects.get(key)
        elif key in super(Framework, self).__getitem__('groupObjects'):
            return self.groupObjects.get(key)
        else:
            try:
                return super(Framework, self).__getitem__(key)
            except:
                return None

    @property
    def elements(self):
        return list(self.elementObjects.keys())

    @property
    def lines(self):
        return list(self.latticeObjects.keys())

    @property
    def commands(self):
        return list(self.commandObjects.keys())

    def track(self, files=None, startfile=None, endfile=None, preprocess=True, write=True, track=True, postprocess=True, save_summary=True):
        self.save_lattice(directory=self.subdirectory, filename='lattice.yaml')
        self.save_settings(directory=self.subdirectory, filename='settings.def', elements={'filename': 'lattice.yaml'})
        self.progress = 0
        if files is None:
            files = ['generator'] + self.lines if not hasattr(self, 'generator') else self.lines
        if startfile is not None and startfile in files:
            index = files.index(startfile)
            files = files[index:]
        if endfile is not None and endfile in files:
            index = files.index(endfile)
            files = files[:index+1]
        if self.verbose:
            format_custom_text = progressbar.FormatCustomText(
                'File: %(running)s', {'running': ''}
            )
            bar = progressbar.ProgressBar(widgets=[format_custom_text, progressbar.Percentage(), progressbar.Bar(), progressbar.Percentage(),], max_value=len(files))
            format_custom_text.update_mapping(running=files[0]+'  ')
            for i in bar(list(range(len(files)))):
                l = files[i]
                self.progress = 100. * (i+1)/len(files)
                if l == 'generator' and hasattr(self, 'generator'):
                    format_custom_text.update_mapping(running='Generator  ')
                    if write:
                        self.generator.write()
                    if track:
                        self.generator.run()
                    if postprocess:
                        self.generator.postProcess()
                    if save_summary:
                        self.save_summary_files()
                else:
                    if i == (len(files) - 1):
                        format_custom_text.update_mapping(running='Finished')
                    else:
                        format_custom_text.update_mapping(running=files[i+1]+'  ')
                    if preprocess:
                        self.latticeObjects[l].preProcess()
                    if write:
                        self.latticeObjects[l].write()
                    if track:
                        self.latticeObjects[l].run()
                    if postprocess:
                        self.latticeObjects[l].postProcess()
                    if save_summary:
                        self.save_summary_files()
        else:
            for i in range(len(files)):
                l = files[i]
                self.progress = 100. * (i)/len(files)
                if l == 'generator' and hasattr(self, 'generator'):
                    if write:
                        self.generator.write()
                    self.progress = 100. * (i+0.33)/len(files)
                    if track:
                        self.generator.run()
                    self.progress = 100. * (i+0.66)/len(files)
                    if postprocess:
                        self.generator.postProcess()
                    if save_summary:
                        self.save_summary_files()
                else:
                    if preprocess:
                        self.latticeObjects[l].preProcess()
                    self.progress = 100. * (i+0.25)/len(files)
                    if write:
                        self.latticeObjects[l].write()
                    self.progress = 100. * (i+0.5)/len(files)
                    if track:
                        self.latticeObjects[l].run()
                    self.progress = 100. * (i+0.75)/len(files)
                    if postprocess:
                        self.latticeObjects[l].postProcess()
                    if save_summary:
                        self.save_summary_files()
            self.progress = 100

    def save_summary_files(self, twiss=True, beams=True):
        t = rtf.load_directory(self.subdirectory)
        t.save_HDF5_twiss_file(self.subdirectory+'/'+'Twiss_Summary.hdf5')
        rbf.save_HDF5_summary_file(self.subdirectory, self.subdirectory+'/'+'Beam_Summary.hdf5')

class frameworkDirectory(Munch):

    def __init__(self, directory='.', twiss=True, beams=False, verbose=False, settings='settings.def', changes='changes.yaml'):
        super(frameworkDirectory, self).__init__()
        directory = os.path.abspath(directory)
        self.framework = Framework(directory, clean=False, verbose=verbose)
        self.framework.loadSettings(directory+'/'+settings)
        if os.path.exists(directory+'/'+changes):
            self.framework.load_changes_file(directory+'/'+changes)
        if twiss:
            self.twiss = rtf.load_directory(directory)
        else:
            self.twiss = None
        if beams:
            self.beams = rbf.load_directory(directory)
        else:
            self.beams = None

    if use_matplotlib:
        def plot(self, *args, **kwargs):
            return groupplot.plot(self, *args, **kwargs)

    def __repr__(self):
        return repr({'framework': self.framework, 'twiss': self.twiss, 'beams': self.beams})

    def getScreen(self, screen):
        if self.beams:
            return self.beams.getScreen(screen)

    def element(self, element, field=None):
        elem = self.framework.getElement(element)
        if field:
            return elem[field]
        else:
            disallowed = ['allowedkeywords', 'keyword_conversion_rules_elegant', 'objectdefaults', 'global_parameters', 'objectname', 'subelement']
            return pprint({k.replace('object',''):v for k,v in elem.items() if k not in disallowed})
        return elem


def load_directory(directory='.', twiss=True, beams=False, **kwargs):
    fw = frameworkDirectory(directory=directory, twiss=twiss, beams=beams, verbose=True, **kwargs)
    return fw
