import numpy as np
from .. import read_gdf_file as rgf
from .. import constants
from math import copysign


def sign(x):
    return copysign(1, x)

def get_particle_index(m, q):
    if (0.9*constants.m_e) < m < (1.1*constants.m_e):
        if sign(q) < 0:
            return 1
        else:
            return 2
    elif (0.9*constants.m_p) < m < (1.1*constants.m_p):
        return 3

def write_gdf_beam_file(
    self,
    filename,
    normaliseX=False,
    normaliseZ=False,
    cathode=False,
    charge=None,
    mass=None,
):
    q = self._beam["particle_charge"]
    m = self._beam["particle_mass"]

    if "nmacro" in self._beam and len(self._beam["nmacro"]) == len(self.x):
        nmacro = abs(self._beam["nmacro"])
    elif len(self._beam["charge"]) == len(self.x):
        nmacro = abs(self._beam["charge"] / charge)
    else:
        nmacro = np.full(len(self.x), abs(self._beam['total_charge'] / constants.elementary_charge / len(self.x)))
    toffset = np.mean(self.z / (self.Bz * constants.speed_of_light))
    x = self.x if not normaliseX else (self.x - normaliseX) if isinstance(normaliseX,(int, float)) else (self.x - np.mean(self.x))
    z = self.z if not normaliseZ else (self.z - normaliseZ) if isinstance(normaliseZ,(int, float)) else (self.z - np.mean(self.z))
    # t = self.t - np.max(self.t)
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
    gdfbeam = read_gdf_beam_file_object(self, file)
    # print('grab_groups = ',  gdfbeam.grab_groups)
    # print(( 'Positions = ', gdfbeam.positions))
    # print(( 'Times = ', gdfbeam.times))
    return gdfbeam

def read_gdf_beam_file(self, file=None, position=None, time=None, block=None, charge=None, longitudinal_reference='t', gdfbeam=None):
    self.reset_dicts()
    if gdfbeam is None and not file is None:
        gdfbeam = read_gdf_beam_file_object(self, file)
        # self.gdfbeam = gdfbeam
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
    # print(gdfbeamdata.datasets)
    self.filename = file
    self["code"] = "GPT"
    if hasattr(gdfbeamdata, "m"):
        self._beam["particle_mass"] = gdfbeamdata.m
    else:
        self._beam["particle_mass"] = np.full(len(gdfbeamdata.x), constants.electron_mass)
    self._beam["particle_rest_energy"] = [m * constants.speed_of_light**2 for m in self._beam["particle_mass"]]
    self._beam["particle_rest_energy_eV"] = [E0 / constants.elementary_charge for E0 in self._beam["particle_rest_energy"]]
    self._beam["particle_charge"] = [sign(q) for q in gdfbeamdata.q]
    self._beam["particle_index"] = [get_particle_index(m, q) for m, q in zip(self._beam["particle_mass"], self._beam["particle_charge"]) ]

    self._beam["x"] = gdfbeamdata.x
    self._beam["y"] = gdfbeamdata.y

    if hasattr(gdfbeamdata, "G"):
        self._beam["gamma"] = gdfbeamdata.G
    if hasattr(gdfbeamdata, "Bx") and hasattr(gdfbeamdata, "G"):
        self._beam["px"] = gdfbeamdata.G * gdfbeamdata.Bx * self._beam["particle_rest_energy"] / constants.speed_of_light
        self._beam["py"] = gdfbeamdata.G * gdfbeamdata.By * self._beam["particle_rest_energy"] / constants.speed_of_light
        self._beam["pz"] = gdfbeamdata.G * gdfbeamdata.Bz * self._beam["particle_rest_energy"] / constants.speed_of_light
    elif hasattr(gdfbeamdata, "GBx"):
        self._beam["px"] = gdfbeamdata.GBx * self._beam["particle_rest_energy"] / constants.speed_of_light
        self._beam["py"] = gdfbeamdata.GBy * self._beam["particle_rest_energy"] / constants.speed_of_light
        self._beam["pz"] = gdfbeamdata.GBz * self._beam["particle_rest_energy"] / constants.speed_of_light
    else:
        raise Exception('GDF File does not have Bx or GBx!')

    if hasattr(gdfbeamdata, "z") and longitudinal_reference == "z":
        # print( 'z!')
        # print(( gdfbeamdata.z))
        self._beam['z'] = gdfbeamdata.z
        self._beam['t'] = np.full(len(self.z), 0)# self.z / (-1 * self.Bz * constants.speed_of_light)
    elif hasattr(gdfbeamdata,'t') and longitudinal_reference == 't':
        # print( 't!')
        self._beam['t'] = gdfbeamdata.t
        self._beam['z'] = (-1 * gdfbeamdata.Bz * constants.speed_of_light) * (gdfbeamdata.t-np.mean(gdfbeamdata.t)) + gdfbeamdata.z
    else:
        if hasattr(gdfbeamdata,'z') and self['longitudinal_reference'] == 'z':
                self._beam['z'] = gdfbeamdata.z
                self._beam['t'] = gdfbeamdata.z / (-1 * gdfbeamdata.Bz * constants.speed_of_light)
                self._beam['t'][self._beam['t'] == inf] = 0
        elif hasattr(gdfbeamdata,'t'):
            self._beam['t'] = gdfbeamdata.t
            self._beam['z'] = (-1 * gdfbeamdata.Bz * constants.speed_of_light) * (gdfbeamdata.t-np.mean(gdfbeamdata.t)) + gdfbeamdata.z

    if hasattr(gdfbeamdata,'q') and  hasattr(gdfbeamdata,'nmacro'):
        self._beam['charge'] = gdfbeamdata.q * gdfbeamdata.nmacro
        self._beam['total_charge'] = np.sum(self._beam['charge'])
        self._beam['nmacro'] = gdfbeamdata.nmacro
    elif hasattr(gdfbeamdata,'q'):
        self._beam['charge'] = gdfbeamdata.q
        self._beam['total_charge'] = np.sum(self._beam['charge'])
        self._beam['nmacro'] = int(self._beam['charge'] / constants.elementary_charge)
    else:
        if charge is None:
            self._beam['total_charge'] = 0
            self._beam['charge'] = np.full(len(self.z), 0)
        else:
            self._beam['total_charge'] = charge

    if hasattr(gdfbeamdata,'nmacro'):
        self._beam['nmacro'] = gdfbeamdata.nmacro
    else:
        self._beam['nmacro'] = np.full(len(self.z), 1)
    return gdfbeam
