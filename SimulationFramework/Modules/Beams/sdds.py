import numpy as np
import scipy.constants as constants
try:
    import sdds
except:
    print('sdds failed to load')
    pass

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
        self._beam['charge'] = np.full(len(self._beam['z']), self._beam['total_charge']/len(self._beam['x']))
    elif charge is None:
        self._beam['total_charge'] = 0
        self._beam['charge'] = np.full(len(self._beam['z']), self._beam['total_charge']/len(self._beam['x']))
    else:
        self._beam['total_charge'] = charge
        self._beam['charge'] = np.full(len(self._beam['z']), self._beam['total_charge']/len(self._beam['x']))
    # self._beam['charge'] = []

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
