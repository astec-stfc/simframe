import os
import sys
import yaml
from munch import Munch, unmunchify
from collections import OrderedDict

class FrameworkSettings(Munch):

    isthistheissue = {'global', 'generator', 'files', 'groups', 'elements'}

    def __init__(self, filename=None, new=False):
        super().__init__()
        self.settingsFilename = filename
        for k in self.isthistheissue:
            self[k] = OrderedDict()
        if filename and os.path.exists(filename) and not new:
            self.loadSettings(filename)

    def loadSettings(self, filename):
        self.settingsFilename = filename
        with open(filename, 'r') as stream:
            settings = yaml.safe_load(stream)
        for k, v in settings.items():
            if k in self.isthistheissue:
                self.update({k: v})

    def copy(self):
        return unmunchify(self)

    def add_file(self, name, code, start, end, input={}, charge={}):
        if isinstance(start, (float, int)):
            output = {
                'zstart': start,
                'end_element': end
            }
        else:
            output = {
                'start_element': start,
                'end_element': end
            }
        self['files'][name] = {
            'code': code,
            'output': output,
            'input': input,
            'charge': charge
        }

    def add_group(self, name, type, elements):
        self['groups'][name] = {
            'type': type,
            'elements': elements
        }

    def add_element(self, name, type, length, position_start, position_end, **kwargs):
        self['elements'][name] = {
            'type': type,
            'length': length,
            'position_start': position_start,
            'position_end': position_end
        }
        self['elements'][name].update(**kwargs)

    def add_element_file(self, filename):
        if 'filename' in self.elements:
            self.elements.filename = []
        self.elements.filename.append(filename)
