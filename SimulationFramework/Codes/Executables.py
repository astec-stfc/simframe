import socket
import os
import yaml

def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

class executable:

    def __init__(self, name, settings={}, location=None, ncpu=1, default=''):
        self.name = name
        self.settings = settings
        self.location = location
        self.ncpu = ncpu
        if location is not None:
            if isinstance(location,str):
                self.executable = [location]
            elif isinstance(location,list):
                self.executable = location
        elif socket.gethostname() in self.settings:
            self.executable = self._subsitute_variables(self.settings[socket.gethostname()][name])
        elif os.name in self.settings:
            self.executable = self._subsitute_variables(self.settings[os.name][name])
        else:
            self.executable =  self._subsitute_variables(default)

    def _subsitute_variables(self, param):
        if isinstance(param, list):
            return [self._subsitute_variables(s) for s in param]
        else:
            return self._subsitute_ncpu(self._subsitute_simcodes(param))

    def _subsitute_simcodes(self, param):
        if isinstance(param, list):
            return [self._subsitute_simcodes(s) for s in param]
        else:
            return param.replace('$simcodes$', self.settings['sim_codes_location'])

    def _subsitute_ncpu(self, param):
        if isinstance(param, list):
            return [self._subsitute_ncpu(s) for s in param]
        else:
            return param.replace('$ncpu$', str(self.ncpu))

class Executables(object):

    def __init__(self, global_parameters):
        super(Executables, self).__init__()
        self.global_parameters = global_parameters
        sim_codes = self.global_parameters['simcodes_location'] if 'simcodes_location' in self.global_parameters else None
        if sim_codes is None:
            self.sim_codes_location = (os.path.relpath(os.path.dirname(os.path.abspath(__file__)) + '/../SimCodes/SimCodes')+'/').replace('\\','/')
            # print('Using SimCodes at ', os.path.abspath(self.sim_codes_location))
        else:
            self.sim_codes_location = sim_codes
        # try:
        with open(os.path.join(os.path.dirname(__file__), '../Executables.yaml'), 'r') as file:
            self.settings = yaml.load(file, Loader=yaml.Loader)
        # except:
        #     self.settings = {}
        self.settings['sim_codes_location'] = self.sim_codes_location
        self.define_ASTRAgenerator_command()
        self.define_astra_command()
        self.define_elegant_command()
        self.define_csrtrack_command()
        self.define_gpt_command()

    def __getitem__(self, item):
        return getattr(self, item)

    def getNCPU(self, ncpu, scaling):
        if scaling is not None and ncpu == 1:
            return 3*scaling
        else:
            return ncpu

    def define_ASTRAgenerator_command(self, location=None):
        self.ASTRAgeneratorExecutable = executable('astragenerator', settings=self.settings, location=location, default=[self.sim_codes_location+'ASTRA/generator'])
        self.ASTRAgenerator =  self.ASTRAgeneratorExecutable.executable

    def define_astra_command(self, location=None, ncpu=1, scaling=None):
        ncpu = self.getNCPU(ncpu, scaling)
        self.astraExecutable = executable('astra', settings=self.settings, location=location, ncpu=ncpu, default=[self.sim_codes_location+'ASTRA/astra'])
        self.astra = self.astraExecutable.executable

    def define_elegant_command(self, location=None, ncpu=1, scaling=None):
        ncpu = self.getNCPU(ncpu, scaling)
        if ncpu > 1:
            self.elegantExecutable = self.executable('Pelegant', settings=self.settings, location=location, ncpu=ncpu, default=[which('mpiexec.exe'),'-np',str(min([2,int(ncpu/3)])), which('Pelegant.exe')])
        else:
            self.elegantExecutable = executable('elegant', settings=self.settings, location=location, ncpu=ncpu, default=[self.sim_codes_location+'Elegant/elegant'])
        self.elegant = self.elegantExecutable.executable

    def define_csrtrack_command(self, location=None, ncpu=1, scaling=None):
        ncpu = self.getNCPU(ncpu, scaling)
        self.csrtrackExecutable = executable('csrtrack', settings=self.settings, location=location, ncpu=ncpu, default=[self.sim_codes_location+'CSRTrack/csrtrack'])
        self.csrtrack = self.csrtrackExecutable.executable

    def define_gpt_command(self, location=None, ncpu=1, scaling=None):
        ncpu = self.getNCPU(ncpu, scaling)
        self.gptExecutable = executable('gpt', settings=self.settings, location=location, ncpu=ncpu, default=[self.sim_codes_location+'GPT/gpt.exe','-j',str(ncpu)])
        self.gpt = self.gptExecutable.executable
