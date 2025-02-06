import os
import munch
import numpy as np
import re
import copy
import glob
import h5py
from ..units import UnitValue
from .. import constants

# I can't think of a clever way of doing this, so...
def get_properties(obj):
    return [f for f in dir(obj) if type(getattr(obj, f)) is property]


class field(munch.Munch):

    properties = {
        "x": "m",
        "y": "m",
        "z": "m",
        "r": "m",
        "Ex": "V/m",
        "Ey": "V/m",
        "Ez": "V/m",
        "Er": "V/m",
        "Bx": "T",
        "By": "T",
        "Bz": "T",
        "Br": "T",
    }

    def __init__(self, filename=None):
        self.filename = ""
        self.code = None
        if filename is not None:
            self.read_field_file(filename)

    def read_field_file(self, filename, run_extension="001"):
        try:
            with h5py.File(filename, "r") as f:
                print(f"{filename} is a valid HDF5 file.")
        except OSError:
            print(f"{filename} is NOT an HDF5 file.")
