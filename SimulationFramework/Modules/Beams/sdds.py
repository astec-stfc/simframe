import numpy as np
from .. import constants
from ..units import UnitValue

from ..SDDSFile import SDDSFile, SDDS_Types


def read_SDDS_beam_file(self, fileName, charge=None, ascii=False):
    self.reset_dicts()
    self.sddsindex += 1
    elegantObject = SDDSFile(index=(self.sddsindex), ascii=ascii)
    elegantObject.read_file(fileName)
    elegantData = elegantObject.data
    for k, v in elegantData.items():
        self._beam[k] = v
    self.filename = fileName
    self['code'] = "SDDS"
    cp = (self._beam['p']) * self.E0_eV
    cpz = cp / np.sqrt(self._beam['xp']**2 + self._beam['yp']**2 + 1)
    cpx = self._beam['xp'] * cpz
    cpy = self._beam['yp'] * cpz
    self._beam['px'] = cpx * self.q_over_c
    self._beam['py'] = cpy * self.q_over_c
    self._beam['pz'] = cpz * self.q_over_c
    # self._beam['t'] = self._beam['t']
    self._beam['z'] = (-1*self._beam.Bz * constants.speed_of_light) * (self._beam.t-np.mean(self._beam.t)) #np.full(len(self.t), 0)
    if 'Charge' in elegantData and len(elegantData['Charge']) > 0:
        self._beam['total_charge'] = elegantData['Charge'][0]
        self._beam['charge'] = np.full(len(self._beam['z']), self._beam['total_charge']/len(self._beam['x']))
    elif charge is None:
        self._beam['total_charge'] = 0
        self._beam['charge'] = np.full(len(self._beam['z']), self._beam['total_charge']/len(self._beam['x']))
    else:
        self._beam['total_charge'] = charge
        self._beam['charge'] = np.full(len(self._beam['z']), self._beam['total_charge']/len(self._beam['x']))
    self._beam['nmacro'] = np.full(len(self._beam['z']), 1)
    # self._beam['charge'] = []

def write_SDDS_file(self, filename, ascii=False, xyzoffset=[0,0,0]):
    """Save an SDDS file using the SDDS class."""
    xoffset = xyzoffset[0]
    yoffset = xyzoffset[1]
    zoffset = xyzoffset[2] # Don't think I need this because we are using t anyway...
    self.sddsindex += 1
    x = SDDSFile(index=(self.sddsindex), ascii=ascii)

    Cnames = ["x", "xp", "y", "yp", "t", "p"]
    Ctypes = [SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_DOUBLE]
    Csymbols = ["", "x'","","y'","",""]
    Cunits = ["m","","m","","s","m$be$nc"]
    Ccolumns = [np.array(self.x) - float(xoffset), self.xp, np.array(self.y) - float(yoffset), self.yp, self.t , self.cp/self.E0_eV]
    x.add_columns(Cnames, Ccolumns, Ctypes, Cunits, Csymbols)

    Pnames = ["pCentral", "Charge", "Particles"]
    Ptypes = [SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_DOUBLE]
    Psymbols = ["p$bcen$n", "", ""]
    Punits = ["m$be$nc", "C", ""]
    parameterData = [np.mean(self.BetaGamma), abs(self._beam['total_charge']), len(self.x)]
    x.add_parameters(Pnames, parameterData, Ptypes, Punits, Psymbols)

    x.write_file(filename)

def set_beam_charge(self, charge):
    self._beam['total_charge'] = charge
