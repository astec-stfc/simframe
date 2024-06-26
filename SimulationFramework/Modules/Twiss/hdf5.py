import h5py
import numpy as np
from ..units import UnitValue

def read_hdf_summary(self, filename, reset=True):
    if reset:
        self.reset_dicts()
    f = h5py.File(filename, "r")
    xemit = f.get('Xemit')
    yemit = f.get('Yemit')
    zemit = f.get('Zemit')
    for item, params in sorted(xemit.items()):
        self.interpret_astra_data(np.array(xemit.get(item)), np.array(yemit.get(item)), np.array(zemit.get(item)))


def write_HDF5_twiss_file(self, filename, sourcefilename=None):
    with h5py.File(filename, "w") as f:
        inputgrp = f.create_group("Parameters")
        if sourcefilename is not None:
            inputgrp['Source'] = sourcefilename
        twissgrp = f.create_group("twiss")
        twissgrp['columns'] = np.array(list(self.properties.keys()), dtype='S')
        twissgrp['units'] = np.array(list(self.properties.values()), dtype='S')
        array = np.array([self[k] if not k == 'element_name' else np.array(self[k], dtype='S') for k in self.properties.keys() if len(self[k]) > 0]).transpose()
        # print([[k, self[k].shape] for k in self.properties.keys()])
        twissgrp.create_dataset("twiss", data=array)

def read_HDF5_twiss_file(self, filename):
    with h5py.File(filename, "r") as h5file:
        cols = list(h5file.get('twiss/columns').asstr())
        twiss = np.array(h5file.get('twiss/twiss')).transpose()
        units = list(h5file.get('twiss/units').asstr())
        [setattr(self, c, UnitValue(v, units=u, dtype=self.properties[c].dtype)) for c,v,u in zip(cols,twiss,units)]
