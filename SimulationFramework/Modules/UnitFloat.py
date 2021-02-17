import warnings
import numpy as np
from .units import unit
import re
np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

# Dicts for prefixes
PREFIX_FACTOR = {
    'yocto-' :1e-24,
    'zepto-' :1e-21,
    'atto-'  :1e-18,
    'femto-' :1e-15,
    'pico-'  :1e-12,
    'nano-'  :1e-9 ,
    'micro-' :1e-6,
    'milli-' :1e-3 ,
    'centi-' :1e-2 ,
    'deci-'  :1e-1,
    'deca-'  :1e+1,
    'hecto-' :1e2  ,
    'kilo-'  :1e3  ,
    'mega-'  :1e6  ,
    'giga-'  :1e9  ,
    'tera-'  :1e12 ,
    'peta-'  :1e15 ,
    'exa-'   :1e18 ,
    'zetta-' :1e21 ,
    'yotta-' :1e24
}
# Inverse
PREFIX = dict( (v,k) for k,v in PREFIX_FACTOR.items())

SHORT_PREFIX_FACTOR = {
    'y'  :1e-24,
    'z'  :1e-21,
    'a'  :1e-18,
    'f'  :1e-15,
    'p'  :1e-12,
    'n'  :1e-9 ,
    'µ'  :1e-6,
    'm'  :1e-3 ,
    'c'  :1e-2 ,
    'd'  :1e-1,
    ''   : 1,
    'da' :1e+1,
    'h'  :1e2  ,
    'k'  :1e3  ,
    'M'  :1e6  ,
    'G'  :1e9  ,
    'T'  :1e12 ,
    'P'  :1e15 ,
    'E'  :1e18 ,
    'Z'  :1e21 ,
    'Y'  :1e24
}
# Inverse
SHORT_PREFIX = dict( (v,k) for k,v in SHORT_PREFIX_FACTOR.items())

def nice_scale_prefix(scale):
    """
    Returns a nice factor and a SI prefix string

    Example:
        scale = 2e-10

        f, u = nice_scale_prefix(scale)


    """

    if scale == 0:
        return 1, ''

    p10 = np.log10(abs(scale))

    if p10 <-2 or p10 > 2:
        f = 10**(p10 //3 *3)
    else:
        f = 1

    if f in SHORT_PREFIX:
        return f, SHORT_PREFIX[f]
    return 1, ''

def nice_array(a):
    """
    Returns a scaled array, the scaling, and a unit prefix

    Example:
        nice_array( np.array([2e-10, 3e-10]) )
    Returns:
        (array([200., 300.]), 1e-12, 'p')

    """

    if np.isscalar(a):
        x = a
    elif len(a) == 1:
        x = a[0]
    else:
        a = np.array(a)
        x = a.ptp()

    fac, prefix = nice_scale_prefix( x )

    return a/fac, fac,  prefix

def unit_power(string, power_factor=1):
    ''' # Takes a string in the form 'a^x' and returns ('a', x) '''
    if isinstance(string, (list, tuple)) and len(string) == 2:
        return string
    if '^' in string:
        pre, power = string.split('^')
        power = power.replace('(','').replace(')','')
        if '/' in power:
            powers = [int(s) for s in power.split('/')]
            power = powers[0]
            for p in powers[1:]:
                power /= p
        return pre, power_factor*float(power)
    return string, power_factor

def unit_power_multiply(string1, string2, power_factor=1):
    '''Takes two strings in the form ('a', x) and ('a', y) and returns ('a', (x+y)) '''
    unit1, power1 = unit_power(string1)
    unit2, power2 = unit_power(string2)
    if unit1 == unit2:
        power = power_factor*(power1 + power2)
        if power == 0:
            return '', 0
        if float(power).is_integer():
             return unit1, int(power)
        return unit1, float(power)

def unit_power_string(power_list, power_factor=1):
    ''' Returns a unit in ('a', x) format and makes a unit string '''
    unit, power = power_list
    power = power_factor * power
    if power == 0:
        return ''
    if float(power).is_integer():
        if power == 1:
            return unit
        return unit+'^'+str(int(power))
    num, denom = power.as_integer_ratio()
    return unit+'^('+str(num)+'/'+str(denom)+')'

def unit_fraction(string):
    ''' split a tring into numerator and denominator taking into account () - input is string in form 'a^x*b/c^y' '''
    num=[]
    denom=[]
    substrings = num
    substring = ''
    inhat = False
    inbracket = False
    individe = False
    inhatdivide = 0
    for s in string:
        if s == '/':
            if not inhat:
                inhatdivide = 0
                individe = True
                if not substring.isnumeric() and not substring == '': substrings.append(substring)
                substring = ''
                if individe and inbracket:
                    substrings = nom
                else:
                    substrings = denom
            else:
                inhatdivide += 1
                if inhatdivide > 1:
                    inhatdivide = 0
                    if not substring.isnumeric() and not substring == '': substrings.append(substring)
                    substring = ''
                    if individe and inbracket:
                        substrings = num
                    else:
                        substrings = denom
                else:
                    substring += s
        elif s == '*':
            individe = False
            inhatdivide = 0
            inhat = False
            if not substring.isnumeric() and not substring == '': substrings.append(substring)
            substring = ''
            if not inbracket:
                substrings = num
        elif s == ' ':
            if inhat:
                inhatdivide = 0
                if substring[-1] == '/':
                    individe = True
                    substring = substring[:-1]
            inhat = False
            if not substring.isnumeric() and not substring == '': substrings.append(substring)
            substring = ''
            if individe:
                substrings = denom
        elif s == '(':
            inbracket = True
        elif s == ')':
            inbracket = False
        elif s == '^':
            inhat = True
            substring += s
        elif inhat and not s.isnumeric():
            inhat = False
            inhatdivide = 0
            if substring[-1] == '/':
                individe = True
                substring = substring[:-1]
            if not substring.isnumeric() and not substring == '': substrings.append(substring)
            substring = s
            if individe:
                substrings = denom
        else:
            substring += s

    if not substring.isnumeric() and not substring == '': substrings.append(substring)
    return num, denom

def expand_units(unit_list, power_factor=1):
    '''Takes a list of units (normally numerator or denominator) and collects units and powers
    Returns units ('a^x') and unitnames (('a',x))'''

    # if isinstance(unit_list[0], (list, tuple)):
    #     return [expand_units(ul) for ul in unit_list]

    unitnames = [unit_power(u, power_factor=power_factor) for u in unit_list]
    unique_units = list(set([u[0] for u in unitnames]))
    units = []
    for uu in unique_units:
        subunits = [u for u in unitnames if u[0] == uu]
        unit = subunits[0]
        if len(subunits) > 1:
            for u in subunits[1:]:
                unit = unit_power_multiply(unit, u)
            if not unit[1] == 0:
                units.append(unit)
        else:
            units.append(unit)
    return units

def collect_units(unit_powers):
    ''' Collects units into numerator ands denominator and uses '*' and '/' correctly '''
    combined_list = sorted(unit_powers, key=lambda x: -x[1])
    if len(combined_list) > 0:
        finalunit = unit_power_string(combined_list[0])
        for u in combined_list[1:]:
            if u[1] > 0:
                finalunit += '*' + unit_power_string(u)
            else:
                finalunit += '/' + unit_power_string(u, power_factor=-1)
    else:
        finalunit = ''
    return finalunit

def unit_powers(string, power_factor=1):
    num, denom = unit_fraction(string)
    # print(expand_units(num), expand_units(denom))
    return expand_units(expand_units(num, power_factor=power_factor) + expand_units(denom, power_factor=(-1*power_factor)))

def unit_multiply(string1, string2=False, divide=False):
    ''' multiply/divide two unit strings '''
    if divide:
        pf = [1,-1]
    else:
        pf = [1,1]
    up1 = expand_units(unit_powers(string1, power_factor=pf[0]))
    if string2:
        up2 = expand_units(unit_powers(string2, power_factor=pf[1]))
    else:
        up2 = []
    # print(expand_units(up1 + up2))
    return collect_units(expand_units(up1+up2))

def unit_to_the_power(string1, power=1):
    ''' raise units to a power'''
    up1 = expand_units(unit_powers(string1, power_factor=power))
    return collect_units(up1)

def get_base_units(string):
    if isinstance(string, (UnitFloat, UnitArray)):
        string = string.units
    units_powers = unit_powers(string)
    return np.sum([np.array(unit(u[0]).unitDimension) * u[1] for u in units_powers], axis=0)

def are_units_equal(string1, string2):
    return (get_base_units(string1) == get_base_units(string2)).all()

class UnitFloats(np.float64):

    def __new__(self, value, units=''):
       self.value = np.float64.__new__(self, value)
       return self.value

    def __init__(self, value, units=''):
        self.units = units

    # @property
    def __repr__(self):
        f, prefix = nice_scale_prefix(self)
        return np.float64.__repr__(self/f)+' '+prefix+self.units

    def _mul_div_units(self, m, newval, divide=False):
        if newval == NotImplemented:
            return newval
        if hasattr(m, 'units'):
            newunit = unit_multiply(self.units, m.units, divide=divide)
            return UnitFloat(newval, newunit)
        else:
            return UnitFloat(newval, self.units)

    def __mul__(self, m):
        newval = np.float64.__mul__(self, m)
        return self._mul_div_units(m, newval)

    def __rmul__(self, m):
        newval = np.float64.__rmul__(self, m)
        return self._mul_div_units(m, newval)

    def __truediv__(self, m):
        newval = np.float64.__truediv__(self, m)
        return self._mul_div_units(m, newval, divide=True)

    def _add_sub_units(self, m, newval):
        if newval == NotImplemented:
            return newval
        if hasattr(m, 'units'):
            if are_units_equal(m.units, self.units):
                return UnitFloat(newval, self.units)
            else:
                warnings.warn('Incompatible Units - ignoring units')
                return UnitFloat(newval, '')
        else:
            return UnitFloat(newval, self.units)

    def __add__(self, m):
        newval = np.float64.__add__(self, m)
        return self._add_sub_units(m, newval)

    def __sub__(self, m):
        newval = np.float64.__sub__(self, m)
        if newval == 0:
            return float(0)
        return self._add_sub_units(m, newval)

    def __pow__(self, m):
        newval = np.float64.__pow__(self, m)
        if newval == NotImplemented:
            return newval
        newunit = unit_to_the_power(self.units, m)
        return UnitFloat(newval, newunit)

    @property
    def val(self):
        return float(self.value)

    # @property
    # def scaled(self):
    #     f, prefix = nice_scale_prefix(self)
    #     scaledvalue = float.__mul__(self, 1/f)
    #     return UnitFloat(scaledvalue, units=prefix+self.units)

class UnitArray(np.ndarray):

    def __new__(cls=np.ndarray, input_array=[], units=None, dtype=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        if isinstance(input_array, (float, int)) or (hasattr(input_array,'size') and input_array.size == 1):
            return UnitFloat(input_array, units=units)
        try:
            obj = np.asarray(input_array).view(cls)
        except np.VisibleDeprecationWarning:
            obj = np.asarray(input_array, dtype=object).view(cls)
        # add the new attribute to the created instance
        if units == None:
            array_units = np.unique([a.units for a in input_array if hasattr(a,'units')])
            if len(array_units) == 1:
                obj.units = array_units[0]
            else:
                obj.units = ''
        else:
            obj.units = units
        # Finally, we must return the newly created object:
        return obj

    # @property
    def __repr__(self):
        # f, prefix = nice_scale_prefix(self[0])
        # arr = np.ndarray.__mul__(self, 1/f)
        return np.ndarray.__repr__(self)[:-1] + ', units=\'' + self.units + '\')'

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.units = getattr(obj, 'units', '')

    def __getitem__(self, key):
        if isinstance(key, slice):
            arr = super().__getitem__(key)
            return UnitArray(arr, units=self.units)
        elif isinstance(key, int):
            return UnitArray(super().__getitem__(key), units=self.units)
        return super().__getitem__(key)

    def __array_wrap__(self, obj, context=None):
        result = obj.view(type(self))
        if context[0].__name__ == 'sqrt':
            result.units = unit_to_the_power(obj.units, 0.5)
        return result

    def _mul_div_units(self, m, newval, divide=False):
        if newval is NotImplemented:
            return newval
        if hasattr(m, 'units'):
            newunit = unit_multiply(self.units, m.units, divide=divide)
            return UnitArray(newval, newunit)
        else:
            return UnitArray(newval, self.units)

    def __mul__(self, m):
        newarr = np.ndarray.__mul__(self, m)
        return self._mul_div_units(m, newarr)

    def __rmul__(self, m):
        newarr = np.ndarray.__rmul__(self, m)
        return self._mul_div_units(m, newarr)

    def __truediv__(self, m):
        newarr = np.ndarray.__truediv__(self, m)
        return self._mul_div_units(m, newarr, divide=True)

    def _add_sub_units(self, m, newval):
        if newval is NotImplemented:
            return newval
        if hasattr(m, 'units'):
            if are_units_equal(m.units, self.units):
                return UnitArray(newval, self.units)
            else:
                print('Incompatible Units - ignoring units')
                return UnitArray(newval, '')
        else:
            return UnitArray(newval, self.units)

    def __add__(self, m):
        newarr = np.ndarray.__add__(self, m)
        return self._add_sub_units(m, newarr)

    def __sub__(self, m):
        newarr = np.ndarray.__sub__(self, m)
        return self._add_sub_units(m, newarr)

    def __pow__(self, m):
        newval = np.ndarray.__pow__(self, m)
        if newval is NotImplemented:
            return newval
        newunit = unit_to_the_power(self.units, m)
        return UnitArray(newval, newunit)

    @property
    def val(self):
        return np.asarray(self, dtype=self.dtype)

class UnitFloat(UnitArray):

    pass
