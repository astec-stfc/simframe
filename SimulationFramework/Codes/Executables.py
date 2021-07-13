import socket
import os

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

class Executables(object):

    def __init__(self, global_parameters):
        super(Executables, self).__init__()
        self.osname = os.name
        self.hostname = socket.gethostname()
        self.global_parameters = global_parameters
        sim_codes = self.global_parameters['simcodes_location'] if 'simcodes_location' in self.global_parameters else None
        if sim_codes is None:
            self.sim_codes_location = (os.path.relpath(os.path.dirname(os.path.abspath(__file__)) + '/../../../SimCodes/SimCodes')+'/').replace('\\','/')
            # print('Using SimCodes at ', os.path.abspath(self.sim_codes_location))
        else:
            self.sim_codes_location = sim_codes
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
        if location is not None:
            if isinstance(location,str):
                self.ASTRAgenerator = [location]
            elif isinstance(location,list):
                self.ASTRAgenerator = location
        elif not self.osname == 'nt':
            if 'apclara1' in self.hostname:
                self.ASTRAgenerator =  ['/opt/ASTRA/generator.sh']
            elif 'apclara2' in self.hostname:
                self.ASTRAgenerator =  ['/opt/ASTRA/generator.sh']
            elif 'apclara3' in self.hostname:
                self.ASTRAgenerator =  ['/opt/ASTRA/generator.sh']
        else:
            self.ASTRAgenerator =  [self.sim_codes_location+'ASTRA/generator']

    def define_astra_command(self, location=None, ncpu=1, scaling=None):
        ncpu = self.getNCPU(ncpu, scaling)
        if location is not None:
            if isinstance(location,str):
                self.astra = [location]
            elif isinstance(location,list):
                self.astra = location
        elif not self.osname == 'nt':
            if 'apclara1' in self.hostname:
                self.astra = ['mpiexec','-np',str(ncpu),'/opt/ASTRA/astra_MPICH2.sh']
            elif 'apclara2' in self.hostname:
                self.astra =  ['salloc','-n',str(ncpu),'/usr/lib64/compat-openmpi16/bin/mpirun','/opt/ASTRA/astra_r62_Linux_x86_64_OpenMPI_sld6']
            elif 'apclara3' in self.hostname:
                self.astra =  ['salloc','-n',str(ncpu),'/usr/lib64/compat-openmpi16/bin/mpirun','/opt/ASTRA/astra_r62_Linux_x86_64_OpenMPI_sld6']
        else:
            if int(self.global_parameters['astra_use_wsl']) > 1:
                # print('WSL ASTRA in use! ncpu = ', self.global_parameters['astra_use_wsl'])
                wsl_cpu = int(self.global_parameters['astra_use_wsl'])
                self.astra =  ['wsl','mpiexec','-np',str(wsl_cpu),'/opt/ASTRA/astra_MPICH2.sh']
            else:
                # print('Serial ASTRA in use!')
                self.astra =  [self.sim_codes_location+'ASTRA/astra']

    def define_elegant_command(self, location=None, ncpu=1, scaling=None):
        ncpu = self.getNCPU(ncpu, scaling)
        if location is not None:
            if isinstance(location,str):
                self.elegant = [location]
            elif isinstance(location,list):
                self.elegant = location
        elif ncpu > 1:
            if not self.osname == 'nt':
                if 'apclara1' in self.hostname:
                    self.elegant = ['/opt/MPICH2-3.2/bin/mpiexec','-np',str(ncpu),'Pelegant']
                elif 'apclara2' in self.hostname:
                    self.elegant = ['salloc','-n',str(ncpu),'/usr/lib64/openmpi3/bin/mpiexec','/usr/bin/Pelegant']
                elif 'apclara3' in self.hostname:
                    self.elegant = ['salloc','-w','apclara3','-n',str(ncpu),'/usr/lib64/openmpi3/bin/mpiexec','Pelegant']
            else:
                if which('mpiexec.exe') is not None and which('Pelegant.exe') is not None:
                    self.elegant = [which('mpiexec.exe'),'-np',str(min([2,int(ncpu/3)])), which('Pelegant.exe')]
                else:
                    self.elegant = [self.sim_codes_location+'Elegant/elegant']
        else:
            if not self.osname == 'nt':
                if 'apclara1' in self.hostname:
                    self.elegant = ['elegant']
                elif 'apclara2' in self.hostname:
                    self.elegant = ['srun','elegant']
                elif 'apclara3' in self.hostname:
                    self.elegant = ['srun','elegant']
            else:
                self.elegant = [self.sim_codes_location+'Elegant/elegant']

    def define_csrtrack_command(self, location=None, ncpu=1, scaling=None):
        ncpu = self.getNCPU(ncpu, scaling)
        if location is not None:
            if isinstance(location,str):
                self.csrtrack = [location]
            elif isinstance(location,list):
                self.csrtrack = location
        elif not self.osname == 'nt':
            if 'apclara1' in self.hostname:
                self.csrtrack = ['/opt/OpenMPI-1.4.3/bin/mpiexec','-n',str(ncpu),'/opt/CSRTrack/csrtrack_openmpi.sh']
            elif 'apclara2' in self.hostname:
                self.csrtrack = ['/opt/OpenMPI-1.4.5/bin/mpiexec','-n',str(ncpu),'/opt/CSRTrack/csrtrack_1.204_Linux_x86_64_OpenMPI_1.4.3']
            elif 'apclara3' in self.hostname:
                self.csrtrack = ['/opt/OpenMPI-1.4.5/bin/mpiexec','-n',str(ncpu),'/opt/CSRTrack/csrtrack_1.204_Linux_x86_64_OpenMPI_1.4.3']
        else:
            self.csrtrack = [self.sim_codes_location+'CSRTrack/csrtrack']

    def define_gpt_command(self, location=None, ncpu=1, scaling=None):
        ncpu = self.getNCPU(ncpu, scaling)
        if location is not None:
            if isinstance(location,str):
                self.gpt = [location]
            elif isinstance(location,list):
                self.gpt = location
        elif not self.osname == 'nt':
            # print('gpt on apclara3')
            self.gpt = ['/opt/GPT3.3.6/bin/gpt', '-j',str(ncpu)]
            # print('gpt on apclara3', self.gpt)
        else:
            self.gpt = ['C:/Program Files/General Particle Tracer/bin/gpt.exe','-j',str(ncpu)]
