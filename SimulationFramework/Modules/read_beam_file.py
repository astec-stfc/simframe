parent_name = '.'.join(__name__.split('.')[:-1])
print('parent = ', parent_name)
import os, time, csv, sys, subprocess
import copy
import h5py
import numpy as np
import munch
import re
from collections import OrderedDict
import scipy.constants as constants
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from itertools import compress
try:
    import sdds
except:
    try:
        import ASTeCsdds.sdds as sdds
    except:
        print('sdds failed to load')
        pass
sys.path.append(os.path.abspath(__file__+'/../../'))
import SimulationFramework.Modules.read_gdf_file as rgf
import SimulationFramework.Modules.minimumVolumeEllipse as mve
from .Beams import Particles
MVE = mve.EllipsoidTool()
import glob
import numpy as np

class beamGroup(munch.Munch):


    def __repr__(self):
        return repr(list(self.beams.keys()))

    def __len__(self):
        return len(super(beamGroup, self).__getitem__('beams'))

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.beams.values())[key]
        if key in super(beamGroup, self).__getitem__('beams') and super(beamGroup, self).__getitem__('beams') is not None:
            return super(beamGroup, self).__getitem__('beams').get(key)
        elif hasattr(super(beamGroup, self).__getitem__('beams'),key):
            return getattr(super(beamGroup, self).__getitem__('beams'),key)
        else:
            return super(beamGroup, self).__getitem__(key)

    def __init__(self, filenames=[]):
        self.sddsindex = 0
        self.beams = OrderedDict()
        if isinstance(filenames, (str)):
            filenames = [filenames]
        for f in filenames:
            self.add(f)

    def sort(self, key='t', function='mean', *args, **kwargs):
        if isinstance(function,str) and hasattr(np, function):
            func = getattr(np, function)
        else:
            func = function
        self.beams = OrderedDict(sorted(self.beams.items(), key=lambda item: func(item[1][key]), *args, **kwargs))
        return self

    def add(self, filename):
        if isinstance(filename, (str)):
            filename = [filename]
        for file in filename:
            if os.path.isdir(file):
                self.add_directory(file)
            elif os.path.isfile(file):
                file = file.replace('\\','/')
                self.beams[file] = beam(file)

    def add_directory(self, directory='.', types={'SimFrame':'.hdf5'}, verbose=False):
        if verbose:
            print('Directory:',directory)
        for code, string in types.items():
            beam_files = glob.glob(directory+'/*'+string)
            if verbose:
                print(code, [os.path.basename(t) for t in beam_files])
            self.add(beam_files)

    def param(self, param):
        return [getattr(b._beam, param) for b in self.beams.values()]

class beam(munch.Munch):

    particle_mass = constants.m_e
    E0 = particle_mass * constants.speed_of_light**2
    E0_eV = E0 / constants.elementary_charge
    q_over_c = (constants.elementary_charge / constants.speed_of_light)
    speed_of_light = constants.speed_of_light

    def __init__(self, filename=None, sddsindex=0):
        self._beam = Particles()
        self.sddsindex = sddsindex
        self.filename = ''
        self.code = None
        if filename is not None:
            self.read_beam_file(filename)

    @property
    def beam(self):
        return self._beam

    def __len__(self):
        return len(self._beam.x)

    def __getitem__(self, key):
        if key in super(beam, self).__getitem__('_beam') and super(beam, self).__getitem__('_beam') is not None:
            return super(beam, self).__getitem__('_beam').get(key)
        elif hasattr(super(beam, self).__getitem__('_beam'),key):
            return getattr(super(beam, self).__getitem__('_beam'),key)
        else:
            return super(beam, self).__getitem__(key)

    def __repr__(self):
        return repr({'filename': self.filename, 'code': self.code, 'beam': [k for k,v in self._beam.items() if isinstance(v, np.ndarray) and v.size > 0]})

    def set_particle_mass(self, mass=constants.m_e):
        self.particle_mass = mass

    def normalise_to_ref_particle(self, array, index=0,subtractmean=False):
        array = copy.copy(array)
        array[1:] = array[0] + array[1:]
        if subtractmean:
            array = array - array[0]#np.mean(array)
        return array

    def reset_dicts(self):
        self._beam = Particles()

    def read_beam_file(self, filename, run_extension='001'):
        pre, ext = os.path.splitext(os.path.basename(filename))
        if ext.lower()[:4] == '.hdf':
            self.read_HDF5_beam_file(filename)
        elif ext.lower() == '.sdds':
            self.read_SDDS_beam_file(filename)
        elif ext.lower() == '.gdf':
            self.read_gdf_beam_file(filename)
        elif ext.lower() == '.astra':
            self.read_astra_beam_file(filename)
        elif re.match('.*.\d\d\d\d.'+run_extension, filename):
            self.read_astra_beam_file(filename)
        else:
            try:
                with open(filename, 'r') as f:
                    firstline = f.readline()
                    if 'SDDS' in firstline:
                        self.read_SDDS_beam_file(filename)
            except UnicodeDecodeError:
                if rgf.is_gdf_file(filename):
                        self.read_gdf_beam_file(filename)
                else:
                    return None

    def read_SDDS_beam_file(self, fileName, charge=None, ascii=False):
        self.reset_dicts()
        self.sddsindex += 1
        sddsref = sdds.SDDS(self.sddsindex%20)
        sddsref.load(fileName)
        for col in range(len(sddsref.columnName)):
            if len(sddsref.columnData[col]) == 1:
                self._beam[sddsref.columnName[col]] = sddsref.columnData[col][0]
            else:
                self._beam[sddsref.columnName[col]] = sddsref.columnData[col]
        SDDSparameters = dict()
        for param in range(len(sddsref.parameterName)):
            SDDSparameters[sddsref.parameterName[param]] = sddsref.parameterData[param]
        self.filename = fileName
        self['code'] = "SDDS"
        cp = self._beam['p'] * self.E0_eV
        cpz = cp / np.sqrt(self._beam['xp']**2 + self._beam['yp']**2 + 1)
        cpx = self._beam['xp'] * cpz
        cpy = self._beam['yp'] * cpz
        self._beam['px'] = cpx * self.q_over_c
        self._beam['py'] = cpy * self.q_over_c
        self._beam['pz'] = cpz * self.q_over_c
        # self._beam['t'] = self._beam['t']
        self._beam['z'] = (-1*self._beam.Bz * constants.speed_of_light) * (self._beam.t-np.mean(self._beam.t)) #np.full(len(self.t), 0)
        if 'Charge' in SDDSparameters and len(SDDSparameters['Charge']) > 0:
            self._beam['total_charge'] = SDDSparameters['Charge'][0]
        elif charge is None:
            self._beam['total_charge'] = 0
        else:
            self._beam['total_charge'] = charge
        self._beam['charge'] = []

    def write_SDDS_file(self, filename, ascii=False, xyzoffset=[0,0,0]):
        """Save an SDDS file using the SDDS class."""
        xoffset = xyzoffset[0]
        yoffset = xyzoffset[1]
        zoffset = xyzoffset[2] # Don't think I need this because we are using t anyway...
        self.sddsindex += 1
        x = sdds.SDDS(self.sddsindex%20)
        if ascii:
            x.mode = x.SDDS_ASCII
        else:
            x.mode = x.SDDS_BINARY
        # {x, xp, y, yp, t, p, particleID}
        Cnames = ["x", "xp", "y", "yp", "t","p"]
        Ccolumns = ['x', 'xp', 'y', 'yp', 't', 'BetaGamma']
        Ctypes = [x.SDDS_DOUBLE, x.SDDS_DOUBLE, x.SDDS_DOUBLE, x.SDDS_DOUBLE, x.SDDS_DOUBLE, x.SDDS_DOUBLE]
        Csymbols = ["", "x'","","y'","",""]
        Cunits = ["m","","m","","s","m$be$nc"]
        Ccolumns = [np.array(self.x) - float(xoffset), self.xp, np.array(self.y) - float(yoffset), self.yp, self.t , self.cp/self.E0_eV]
        # {Step, pCentral, Charge, Particles, IDSlotsPerBunch}
        Pnames = ["pCentral", "Charge", "Particles"]
        Ptypes = [x.SDDS_DOUBLE, x.SDDS_DOUBLE, x.SDDS_LONG]
        Psymbols = ["p$bcen$n", "", ""]
        Punits = ["m$be$nc", "C", ""]
        parameterData = [[np.mean(self.BetaGamma)], [abs(self._beam['total_charge'])], [len(self.x)]]
        for i in range(len(Ptypes)):
            x.defineParameter(Pnames[i], Psymbols[i], Punits[i],"","", Ptypes[i], "")
            x.setParameterValueList(Pnames[i], parameterData[i])
        for i in range(len(Ctypes)):
            # name, symbol, units, description, formatString, type, fieldLength
            x.defineColumn(Cnames[i], Csymbols[i], Cunits[i],"","", Ctypes[i], 0)
            x.setColumnValueLists(Cnames[i], [list(Ccolumns[i])])
        x.save(filename)

    def set_beam_charge(self, charge):
        self._beam['total_charge'] = charge

    def read_csv_file(self, file, delimiter=' '):
        with open(file, 'r') as f:
            data = np.array([l for l in csv.reader(f, delimiter=delimiter,  quoting=csv.QUOTE_NONNUMERIC, skipinitialspace=True)])
        return data

    def write_csv_file(self, file, data):
        if sys.version_info[0] > 2:
            with open(file, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=' ',  quoting=csv.QUOTE_NONNUMERIC, skipinitialspace=True)
                [writer.writerow(l) for l in data]
        else:
            with open(file, 'wb') as f:
                writer = csv.writer(f, delimiter=' ',  quoting=csv.QUOTE_NONNUMERIC, skipinitialspace=True)
                [writer.writerow(l) for l in data]

    def read_astra_beam_file(self, file, normaliseZ=False):
        starttime = time.time()
        self.reset_dicts()
        data = self.read_csv_file(file)
        self.filename = fileName
        self.interpret_astra_data(data, normaliseZ=normaliseZ)

    def interpret_astra_data(self, data, normaliseZ=False):
        x, y, z, cpx, cpy, cpz, clock, charge, index, status = np.transpose(data)
        zref = z[0]
        self['code'] = "ASTRA"
        self._beam['reference_particle'] = data[0]
        # if normaliseZ:
        #     self._beam['reference_particle'][2] = 0
        self['longitudinal_reference'] = 'z'
        znorm = self.normalise_to_ref_particle(z, subtractmean=True)
        z = self.normalise_to_ref_particle(z, subtractmean=False)
        cpz = self.normalise_to_ref_particle(cpz, subtractmean=False)
        clock = self.normalise_to_ref_particle(clock, subtractmean=True)
        cp = np.sqrt(cpx**2 + cpy**2 + cpz**2)
        self._beam['x'] = x
        self._beam['y'] = y
        self._beam['z'] = z
        self._beam['px'] = cpx * self.q_over_c
        self._beam['py'] = cpy * self.q_over_c
        self._beam['pz'] = cpz * self.q_over_c
        self._beam['clock'] = 1.0e-9*clock
        self._beam['charge'] = 1.0e-9*charge
        self._beam['index'] = index
        self._beam['status'] = status
        # print self.Bz
        self._beam['t'] = [clock if status == -1 else ((z-zref) / (-1 * Bz * constants.speed_of_light)) for status, z, Bz, clock in zip(self._beam['status'], z, self.Bz, self._beam['clock'])]
        # self._beam['t'] = self.z / (1 * self.Bz * constants.speed_of_light)#[time if status is -1 else 0 for time, status in zip(clock, status)]#
        self._beam['total_charge'] = np.sum(self._beam['charge'])

    def read_csrtrack_beam_file(self, file):
        self.reset_dicts()
        data = self.read_csv_file(file)
        self['code'] = "CSRTrack"
        self._beam['reference_particle'] = data[0]
        self['longitudinal_reference'] = 'z'
        z, x, y, cpz, cpx, cpy, charge = np.transpose(data[1:])
        z = self.normalise_to_ref_particle(z, subtractmean=False)
        cpz = self.normalise_to_ref_particle(cpz, subtractmean=False)
        cp = np.sqrt(cpx**2 + cpy**2 + cpz**2)
        self._beam['x'] = x
        self._beam['y'] = y
        self._beam['z'] = z
        self._beam['px'] = cpx * self.q_over_c
        self._beam['py'] = cpy * self.q_over_c
        self._beam['pz'] = cpz * self.q_over_c
        self._beam['clock'] = np.full(len(self.x), 0)
        self._beam['clock'][0] = data[0, 0] * 1e-9
        self._beam['index'] = np.full(len(self.x), 5)
        self._beam['status'] = np.full(len(self.x), 1)
        self._beam['t'] = self.z / (-1 * self.Bz * constants.speed_of_light)# [time if status is -1 else 0 for time, status in zip(clock, self._beam['status'])]
        self._beam['charge'] = charge
        self._beam['total_charge'] = np.sum(self._beam['charge'])

    def read_vsim_h5_beam_file(self, filename, charge=70e-12, interval=1):
        self.reset_dicts()
        with h5py.File(filename, "r") as h5file:
            data = np.array(h5file.get('/BeamElectrons'))[1:-1:interval]
            z, y, x, cpz, cpy, cpx = data.transpose()
        self.filename = fileName
        self['code'] = "VSIM"
        self['longitudinal_reference'] = 'z'
        cp = np.sqrt(cpx**2 + cpy**2 + cpz**2)
        self._beam['x'] = x
        self._beam['y'] = y
        self._beam['z'] = z
        self._beam['px'] = cpx * self.particle_mass
        self._beam['py'] = cpy * self.particle_mass
        self._beam['pz'] = cpz * self.particle_mass
        self._beam['t'] = [(z / (-1 * Bz * constants.speed_of_light)) for z, Bz in zip(self.z, self.Bz)]
        # self._beam['t'] = self.z / (1 * self.Bz * constants.speed_of_light)#[time if status is -1 else 0 for time, status in zip(clock, status)]#
        self._beam['total_charge'] = charge
        self._beam['charge'] = []

    def read_pacey_beam_file(self, fileName, charge=250e-12):
        self.reset_dicts()
        data = self.read_csv_file(fileName, delimiter='\t')
        self.filename = fileName
        self['code'] = "TPaceyASTRA"
        self['longitudinal_reference'] = 'z'
        x, y, z, cpx, cpy, cpz = np.transpose(data)
        cp = np.sqrt(cpx**2 + cpy**2 + cpz**2)
        self._beam['x'] = x
        self._beam['y'] = y
        self._beam['z'] = z
        self._beam['px'] = cpx * self.q_over_c
        self._beam['py'] = cpy * self.q_over_c
        self._beam['pz'] = cpz * self.q_over_c
        self._beam['t'] = [(z / (-1 * Bz * constants.speed_of_light)) for z, Bz in zip(self.z, self.Bz)]
        # self._beam['t'] = self.z / (1 * self.Bz * constants.speed_of_light)#[time if status is -1 else 0 for time, status in zip(clock, status)]#
        self._beam['total_charge'] = charge
        self._beam['charge'] = []

    def convert_csrtrackfile_to_astrafile(self, csrtrackfile, astrafile):
        data = self.read_csv_file(csrtrackfile)
        z, x, y, cpz, cpx, cpy, charge = np.transpose(data[1:])
        charge = -charge*1e9
        clock0 = (data[0, 0] / constants.speed_of_light) * 1e9
        clock = np.full(len(x), 0)
        clock[0] = clock0
        index = np.full(len(x), 1)
        status = np.full(len(x), 5)
        array = np.array([x, y, z, cpx, cpy, cpz, clock, charge, index, status]).transpose()
        self.write_csv_file(astrafile, array)

    def find_nearest_vector(self, nodes, node):
        return cdist([node], nodes).argmin()

    def rms(self, x, axis=None):
        return np.sqrt(np.mean(x**2, axis=axis))

    def create_ref_particle(self, array, index=0, subtractmean=False):
        array[1:] = array[0] + array[1:]
        if subtractmean:
            array = array - np.mean(array)
        return array

    def write_astra_beam_file(self, file, index=1, status=5, charge=None, normaliseZ=False):
        if not isinstance(index,(list, tuple, np.ndarray)):
            if len(self._beam['charge']) == len(self.x):
                chargevector = 1e9*self._beam['charge']
            else:
                chargevector = np.full(len(self.x), 1e9*self.charge/len(self.x))
        if not isinstance(index,(list, tuple, np.ndarray)):
            indexvector = np.full(len(self.x), index)
        statusvector = self._beam['status'] if 'status' in self._beam else status if isinstance(status,(list, tuple, np.ndarray)) else np.full(len(self.x), status)
        ''' if a particle is emitting from the cathode it's z value is 0 and it's clock value is finite, otherwise z is finite and clock is irrelevant (thus zero) '''
        if self['longitudinal_reference'] == 't':
            zvector = [0 if status == -1 and t == 0 else z for status, z, t in zip(statusvector, self.z, self.t)]
        else:
            zvector = self.z
        ''' if the clock value is finite, we calculate it from the z value, using Betaz '''
        # clockvector = [1e9*z / (1 * Bz * constants.speed_of_light) if status == -1 and t == 0 else 1.0e9*t for status, z, t, Bz in zip(statusvector, self.z, self.t, self.Bz)]
        clockvector = [1.0e9*t for status, z, t, Bz in zip(statusvector, self.z, self.t, self.Bz)]
        ''' this is the ASTRA array in all it's glory '''
        array = np.array([self.x, self.y, zvector, self.cpx, self.cpy, self.cpz, clockvector, chargevector, indexvector, statusvector]).transpose()
        if 'reference_particle' in self._beam:
            ref_particle = self._beam['reference_particle']
            # print 'we have a reference particle! ', ref_particle
            # np.insert(array, 0, ref_particle, axis=0)
        else:
            ''' take the rms - if the rms is 0 set it to 1, so we don't get a divide by error '''
            rms_vector = [a if abs(a) > 0 else 1 for a in self.rms(array, axis=0)]
            ''' normalise the array '''
            norm_array = array / rms_vector
            ''' take the meen of the normalised array '''
            mean_vector = np.mean(norm_array, axis=0)
            ''' find the index of the vector that is closest to the mean - if you read in an ASTRA file, this should actually return the reference particle! '''
            nearest_idx = self.find_nearest_vector(norm_array, mean_vector);
            ref_particle = array[nearest_idx]
            ''' set the closest mean vector to be in position 0 in the array '''
            array = np.roll(array, -1*nearest_idx, axis=0)

        ''' normalise Z to the reference particle '''
        array[1:,2] = array[1:,2] - ref_particle[2]
        ''' should we leave Z as the reference value, set it to 0, or set it to be some offset? '''
        if not normaliseZ is False:
            array[0,2] = 0
        if not isinstance(normaliseZ,(bool)):
            array[0,2] += normaliseZ
        ''' normalise pz and the clock '''
        # print('Mean pz = ', np.mean(array[:,5]))
        array[1:,5] = array[1:,5] - ref_particle[5]
        array[0,6] = array[0,6] + ref_particle[6]
        np.savetxt(file, array, fmt=('%.12e','%.12e','%.12e','%.12e','%.12e','%.12e','%.12e','%.12e','%d','%d'))

    def write_vsim_beam_file(self, file, normaliseT=False):
        if len(self._beam['charge']) == len(self.x):
            chargevector = self._beam['charge']
        else:
            chargevector = np.full(len(self.x), self._beam['total_charge']/len(self.x))
        if normaliseT:
            tvector = self.t - np.mean(self.t)
            zvector = self.z - np.mean(self.z)
        else:
            tvector = self.t
            zvector = self.z
        zvector = [t * (1 * Bz * constants.speed_of_light) if z == 0 else z for z, t, Bz in zip(zvector, tvector, self.Bz)]
        ''' this is the VSIM array in all it's glory '''
        array = np.array([zvector, self.y, self.x, self.Bz*self.gamma*constants.speed_of_light, self.By*self.gamma*constants.speed_of_light, self.Bx*self.gamma*constants.speed_of_light]).transpose()
        ''' take the rms - if the rms is 0 set it to 1, so we don't get a divide by error '''
        np.savetxt(file, array, fmt=('%.12e','%.12e','%.12e','%.12e','%.12e','%.12e'))

    def write_gdf_beam_file(self, filename, normaliseX=False, normaliseZ=False, cathode=False):
        q = np.full(len(self.x), -1 * constants.elementary_charge)
        m = np.full(len(self.x), constants.electron_mass)
        nmacro = np.full(len(self.x), abs(self._beam['total_charge'] / constants.elementary_charge / len(self.x)))
        toffset = np.mean(self.z / (self.Bz * constants.speed_of_light))
        x = self.x if not normaliseX else (self.x - normaliseX) if isinstance(normaliseX,(int, float)) else (self.x - np.mean(self.x))
        z = self.z if not normaliseZ else (self.z - normaliseZ) if isinstance(normaliseZ,(int, float)) else (self.z - np.mean(self.z))
        if cathode:
            dataarray = np.array([x, self.y, z, q, m, nmacro, self.gamma*self.Bx, self.gamma*self.By, self.gamma*self.Bz, self.t]).transpose()
            namearray = 'x y z q m nmacro GBx GBy GBz t'
            np.savetxt(filename, dataarray, fmt=('%.12e','%.12e','%.12e','%.12e','%.12e','%.12e','%.12e','%.12e','%.12e','%.12e'), header=namearray, comments='')
        else:
            dataarray = np.array([x, self.y, z, q, m, nmacro, self.gamma*self.Bx, self.gamma*self.By, self.gamma*self.Bz]).transpose()
            namearray = 'x y z q m nmacro GBx GBy GBz'
            np.savetxt(filename, dataarray, fmt=('%.12e','%.12e','%.12e','%.12e','%.12e','%.12e','%.12e','%.12e','%.12e'), header=namearray, comments='')

    def read_gdf_beam_file_object(self, file):
        if isinstance(file, (str)):
            gdfbeam = rgf.read_gdf_file(file)
        elif isinstance(file, (rgf.read_gdf_file)):
            gdfbeam = file
        else:
            raise Exception('file is not str or gdf object!')
        return gdfbeam

    def calculate_gdf_s(self, file):
        gdfbeam = self.read_gdf_beam_file_object(file)
        datagrab = gdfbeam.get_grab(0)
        avgt = [datagrab.avgt]
        position = [datagrab.position]
        sposition = list(zip(*list(sorted(zip(avgt[0], position[0])))))[1]
        ssposition = list(zip(sposition, list(sposition[1:])+[0]))
        offset = 0
        spos = []
        for p1,p2 in ssposition:
            spos += [p1 + offset]
            if p2 < p1:
                offset += p1
        return spos

    def calculate_gdf_eta(self, file):
        gdfbeam = self.read_gdf_beam_file_object(file)
        etax = []
        etaxp = []
        tp = []
        for p in gdfbeam.positions:
            self.read_gdf_beam_file(gdfbeam=gdfbeam, position=p)
            if len(self.x) > 0:
                e, ep, t = self.calculate_etax()
                etax += [e]
                etaxp += [ep]
                tp += [t]
        etax, etaxp = list(zip(*list(sorted(zip(tp, etax, etaxp)))))[1:]
        return etax, etaxp

    def read_gdf_beam_file_info(self, file):
        self.reset_dicts()
        gdfbeamdata = None
        gdfbeam = self.read_gdf_beam_file_object(file)
        print('grab_groups = ',  gdfbeam.grab_groups)
        print(( 'Positions = ', gdfbeam.positions))
        print(( 'Times = ', gdfbeam.times))

    def read_gdf_beam_file(self, file=None, position=None, time=None, block=None, charge=None, longitudinal_reference='t', gdfbeam=None):
        self.reset_dicts()
        if gdfbeam is None and not file is None:
            gdfbeam = self.read_gdf_beam_file_object(file)
            self.gdfbeam = gdfbeam
        elif gdfbeam is None and file is None:
            return None

        if position is not None:# and (time is not None or block is not None):
            # print 'Assuming position over time!'
            self['longitudinal_reference'] = 't'
            gdfbeamdata = gdfbeam.get_position(position)
            if gdfbeamdata is not None:
                # print('GDF found position ', position)
                time = None
                block = None
            else:
                print('GDF DID NOT find position ', position)
                position = None
        elif position is None and time is not None and block is None:
            # print('Assuming time over block!')
            self['longitudinal_reference'] = 'z'
            gdfbeamdata = gdfbeam.get_time(time)
            if gdfbeamdata is not None:
                block = None
            else:
                 time = None
        elif position is None and time is None and block is not None:
            gdfbeamdata = gdfbeam.get_grab(block)
            if gdfbeamdata is None:
                block = None
        elif position is None and time is None and block is None:
            gdfbeamdata = gdfbeam.get_grab(0)
        self.filename = file
        self['code'] = "GPT"
        self._beam['x'] = gdfbeamdata.x
        self._beam['y'] = gdfbeamdata.y
        if hasattr(gdfbeamdata,'z') and longitudinal_reference == 'z':
            # print( 'z!')
            # print(( gdfbeamdata.z))
            self._beam['z'] = gdfbeamdata.z
            self._beam['t'] = np.full(len(self.z), 0)# self.z / (-1 * self.Bz * constants.speed_of_light)
        elif hasattr(gdfbeamdata,'t') and longitudinal_reference == 't':
            # print( 't!')
            self._beam['t'] = gdfbeamdata.t
            self._beam['z'] = (-1 * gdfbeamdata.Bz * constants.speed_of_light) * (gdfbeamdata.t-np.mean(gdfbeamdata.t)) + gdfbeamdata.z
        else:
            pass
            # print('not z and not t !!')
            # print('z = ', hasattr(gdfbeamdata,'z'))
            # print('t = ', hasattr(gdfbeamdata,'t'))
            # print('longitudinal_reference = ', longitudinal_reference)
        self._beam['gamma'] = gdfbeamdata.G
        if hasattr(gdfbeamdata,'q') and  hasattr(gdfbeamdata,'nmacro'):
            self._beam['charge'] = gdfbeamdata.q * gdfbeamdata.nmacro
            self._beam['total_charge'] = np.sum(self._beam['charge'])
        else:
            if charge is None:
                self._beam['total_charge'] = 0
            else:
                self._beam['total_charge'] = charge
        # print(( self._beam['charge']))
        vx = gdfbeamdata.Bx * constants.speed_of_light
        vy = gdfbeamdata.By * constants.speed_of_light
        vz = gdfbeamdata.Bz * constants.speed_of_light
        velocity_conversion = 1 / (constants.m_e * self._beam['gamma'])
        self._beam['px'] = vx / velocity_conversion
        self._beam['py'] = vy / velocity_conversion
        self._beam['pz'] = vz / velocity_conversion
        return gdfbeam

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
