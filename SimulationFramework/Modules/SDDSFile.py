import os
import munch
try:
    import sdds
except:
    raise Exception('Needs PythonSDDS - see https://www.aps.anl.gov/Accelerator-Operations-Physics/Software#Python%20SDDS%20Binaries')
import numpy as np
import enum

def read_sdds_file(filename, ascii=False, object=None):
    sddsobject = SDDSFile(index=0, ascii=ascii)
    sddsobject.read_file(filename)
    if object is None:
        object = munch.Munch()
        for k, v in sddsobject.data.items():
            object[k] = v
        return object
    return sddsobject

class MyEnumMeta(enum.EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        else:
            return True

class SDDS_Types(enum.IntEnum, metaclass=MyEnumMeta):
    SDDS_LONGDOUBLE = 1
    SDDS_DOUBLE = 2
    SDDS_REAL64 = 2
    SDDS_FLOAT = 3
    SDDS_REAL32 = 3
    SDDS_LONG = 4
    SDDS_INT32 = 4
    SDDS_ULONG = 5
    SDDS_UINT32 = 5
    SDDS_SHORT = 6
    SDDS_INT16 = 6
    SDDS_USHORT = 7
    SDDS_UINT16 = 7
    SDDS_STRING = 8
    SDDS_CHARACTER = 9
    SDDS_NUM_TYPES = 9
    SDDS_BINARY = 1
    SDDS_ASCII = 2

class SDDSObject(munch.Munch):

    def __init__(self, index=1, name=None, data=[], unit="", type=2, symbol="", formatstring="", fieldlength=0, description=""):
        super().__init__()
        self.types = SDDS_Types
        self._name = name
        self._data = data
        self._unit = unit
        self._symbol = symbol
        self._type = type
        self._formatstring = formatstring
        self._fieldlength = fieldlength
        self._description = description

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, name):
        self._name = name
        return self._name

    @property
    def unit(self):
        return self._unit
    @unit.setter
    def unit(self, unit):
        if unit is not None:
            self._unit = unit
        return self._unit

    @property
    def symbol(self):
        return self._symbol
    @symbol.setter
    def symbol(self, symbol):
        if symbol is not None:
            self._symbol = symbol
        return self._symbol

    @property
    def type(self):
        return self._type
    @type.setter
    def type(self, type):
        if isinstance(type, str):
            if type in self.types:
                self._type = type
        elif isinstance(type, int):
            self._type = type
        return self._type

    @property
    def fieldlength(self):
        return self._fieldlength
    @fieldlength.setter
    def fieldlength(self, length):
        if isinstance(length, (int, float)):
            self._fieldlength = int(fieldlength)
        return self._fieldlength

    @property
    def formatstring(self):
        return self._formatstring
    @formatstring.setter
    def formatstring(self, string):
        if isinstance(string, str):
            self._formatstring = string
        return self._formatstring

    @property
    def description(self):
        return self._description
    @description.setter
    def description(self, string):
        if isinstance(string, str):
            self._description = string
        return self._description


    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, data):
        self._data = data
        return self._data

class SDDSColumn(SDDSObject):

    def __init__(self, name=None, data=[], unit="", type=2, symbol="", formatstring="", fieldlength=0, description=""):
        super().__init__(name=name, data=None, unit=unit, type=type, symbol=symbol)
        self.objectType = 'Column'
        self.data = data

    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, data):
        if not isinstance(data, (dict, tuple, list, np.ndarray)):
            if isinstance(data, (float, int, str)):
                data = [data]
            else:
                raise Exception('Wrong data type for SDDS Column!', type(data))
        self._data = data
        return self._data

    def length(self):
        return len(self._data)

class SDDSParameter(SDDSObject):

    def __init__(self, name=None, data=[], unit="", type=2, symbol="", formatstring="", fieldlength=0, description=""):
        super().__init__(name=name, data=None, unit=unit, type=type, symbol=symbol)
        self.objectType = 'Parameter'
        self.data = data

    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, data):
        if not isinstance(data, (dict, tuple, list)):
            self._data = [data]
        else:
            raise Exception('Wrong data type for SDDS Parameter!', type(data))
        return self._data

class SDDSFile(object):

    def __init__(self, index=1, ascii=False):
        super().__init__()
        self.types = SDDS_Types
        self._columns = {}
        self._parameters = {}
        self.sddsindex = index
        self.sddsObject = sdds.SDDS(self.sddsindex%20)
        if ascii:
            self.sddsObject.mode = self.sddsObject.SDDS_ASCII
        else:
            self.sddsObject.mode = self.sddsObject.SDDS_BINARY

    def add_column(self, name, data, type=2, unit="", symbol="", formatstring="", fieldlength=0, description=""):
        self._columns[name] = SDDSColumn(name=name, data=data, unit=unit, type=type, symbol=symbol, formatstring=formatstring, fieldlength=fieldlength, description=description)

    def add_columns(self, name, data, type, unit, symbol, formatstring=None, fieldlength=None, description=None):
        combined_data = {k:v for k,v in {'name': name, 'data': data, 'type': type, 'unit': unit, 'symbol': symbol, 'formatstring': formatstring, 'fieldlength': fieldlength, 'description': description}.items() if v is not None}
        for i in range(len(name)):
            data = dict(zip(combined_data.keys(), [combined_data[k][i] for k in combined_data.keys()]))
            self.add_column(**data)

    def add_parameter(self, name, data, type=2, unit="", symbol="", formatstring="", fieldlength=0, description=""):
        self._parameters[name] = SDDSParameter(name=name, data=data, unit=unit, type=type, symbol=symbol, formatstring=formatstring, fieldlength=fieldlength, description=description)

    def add_parameters(self, name, data, type, unit, symbol, formatstring=None, fieldlength=None, description=None):
        combined_data = {k:v for k,v in {'name': name, 'data': data, 'type': type, 'unit': unit, 'symbol': symbol, 'formatstring': formatstring, 'fieldlength': fieldlength, 'description': description}.items() if v is not None}
        for i in range(len(name)):
            data = dict(zip(combined_data.keys(), [combined_data[k][i] for k in combined_data.keys()]))
            self.add_parameter(**data)

    def write_file(self, filename):
        for name, param in self._parameters.items():
            self.sddsObject.defineParameter(param.name, param.symbol, param.unit, param.description, param.formatstring, param.type, param.fieldlength)
            self.sddsObject.setParameterValueList(param.name, param.data)
        for name, column in self._columns.items():
            # print(len([list(column.data)][0]))
            self.sddsObject.defineColumn(column.name, column.symbol, column.unit, column.description, column.formatstring, column.type, column.fieldlength)
            self.sddsObject.setColumnValueLists(column.name, [list(column.data)])
        self.sddsObject.save(filename)

    def read_file(self, filename):
        self.sddsObject.load(filename)
        sddsref = self.sddsObject
        for col in range(len(sddsref.columnName)):
            symbol, unit, description, formatString, type, fieldLength = sddsref.columnDefinition[col]
            if len(sddsref.columnData[col]) == 1:
                column_data = np.array(sddsref.columnData[col][0])
            else:
                column_data = np.array(sddsref.columnData[col])
            self.add_column(sddsref.columnName[col], column_data, type=type, unit=unit, symbol=symbol, formatstring=formatString, fieldlength=fieldLength, description=description)
        # sddsobject.SDDSparameterNames = list()
        for param in range(len(sddsref.parameterName)):
            symbol, unit, description, formatString, type, fieldLength = sddsref.parameterDefinition[param]
            parameter_data = sddsref.parameterData[param]
            self.add_parameter(sddsref.parameterName[param], parameter_data[0], type=type, unit=unit, symbol=symbol, formatstring=formatString, fieldlength=fieldLength, description=description)

    @property
    def data(self):
        return {k:v.data for k, v in {**self._parameters, **self._columns}.items()}
