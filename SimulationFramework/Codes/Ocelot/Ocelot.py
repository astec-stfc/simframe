from ...Framework_objects import *
from ...Framework_elements import *
from ...FrameworkHelperFunctions import _rotation_matrix
from ...Modules import Beams as rbf
from ...Modules.merge_two_dicts import merge_two_dicts
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.track import track
from ocelot.cpbd.io import save_particle_array, load_particle_array
from copy import deepcopy


class ocelotLattice(frameworkLattice):
    def __init__(self, *args, **kwargs):
        super(ocelotLattice, self).__init__(*args, **kwargs)
        self.code = 'ocelot'
        self.particle_definition = self.allElementObjects[self.start].objectname
        self.bunch_charge = None
        self.q = charge(name='START', type='charge', global_parameters=self.global_parameters, **{'total': 250e-12})
        self.trackBeam = True
        self.betax = None
        self.betay = None
        self.alphax = None
        self.alphay = None
        self.commandFiles = {}
        self.lat_obj = None
        self.pin = None
        self.pout = None

    def endScreen(self, **kwargs):
        return screen(name=self.endObject.objectname, type='screen', centre=self.endObject.centre,
                      position_start=self.endObject.position_start, position_end=self.endObject.position_start,
                      global_rotation=self.endObject.global_rotation, global_parameters=self.global_parameters,
                      **kwargs)

    def writeElements(self):
        self.w = None
        if not self.endObject in self.screens_and_bpms:
            self.w = self.endScreen(output_filename=self.endObject.objectname + '.npz')
        elements = self.createDrifts()
        mag_lat = []
        for element in list(elements.values()):
            # print(element.write_Elegant())
            if not element.subelement:
                mag_lat.append(element.write_Ocelot())
        for o in mag_lat:
            print(o)
        self.lat_obj = MagneticLattice(mag_lat)

    def write(self):
        self.writeElements()
        self.lat_obj.save_as_py_file(f'{self.global_parameters["master_subdir"]}/{self.objectname}.py')

    def preProcess(self):
        prefix = self.file_block['input']['prefix'] if 'input' in self.file_block and 'prefix' in self.file_block[
            'input'] else ''
        if self.trackBeam:
            self.hdf5_to_npz(prefix)
        else:
            HDF5filename = prefix + self.particle_definition + '.hdf5'
            rbf.hdf5.read_HDF5_beam_file(self.global_parameters['beam'],
                                         os.path.abspath(self.global_parameters['master_subdir'] + '/' + HDF5filename))

    def hdf5_to_npz(self, prefix='', write=True):
        HDF5filename = prefix+self.particle_definition+'.hdf5'
        rbf.hdf5.read_HDF5_beam_file(self.global_parameters['beam'], os.path.abspath(self.global_parameters['master_subdir'] + '/' + HDF5filename))
        # print('HDF5 Total charge', self.global_parameters['beam']['total_charge'])
        if self.bunch_charge is not None:
            self.q = charge(name='START', type='charge', global_parameters=self.global_parameters,**{'total': abs(self.bunch_charge)})
        else:
            self.q = charge(name='START', type='charge', global_parameters=self.global_parameters,**{'total': abs(self.global_parameters['beam'].Q)})
        ocebeamfilename = self.objectname+'.npz'
        self.pin = rbf.ocelot.write_ocelot_beam_file(self.global_parameters['beam'], ocebeamfilename, write=write)

    def run(self):
        """Run the code with input 'filename'"""
        tws, self.pout = track(self.lat_obj, deepcopy(self.pin))


    def postProcess(self):
        bfname = f'{self.global_parameters["master_subdir"]}/{self.endObject.objectname}.npz'
        save_particle_array(bfname, self.pout)
        # ocebeam = rbf.ocelot.read_ocelot_beam_file(self.global_parameters['beam'], bfname)
        # rbf.ocelot.write_ocelot_beam_file(self.global_parameters['beam'],
        #                                   self.global_parameters['master_subdir'] + '/' + ocebeamfilename)

class gpt_element(frameworkElement):

    def __init__(self, elementName=None, elementType=None, **kwargs):
        super(gpt_element, self).__init__(elementName, elementType, **kwargs)
        # if elementName in gpt_defaults:
        #     for k, v in list(gpt_defaults[elementName].items()):
        #         self.add_default(k, v)

    def write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + '('
        for k in elementkeywords[self.objecttype]['keywords']:
            k = k.lower()
            if getattr(self,k) is not None:
                output += str(getattr(self, k))+', '
            elif k in self.objectdefaults :
                output += self.objectdefaults[k]+', '
        output = output[:-2]
        output+=');\n'
        return output

class gpt_setfile(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_setfile, self).__init__(elementName='setfile', elementType='gpt_setfile', **kwargs)

    def hdf5_to_gpt(self, prefix=''):
        HDF5filename = prefix+self.particle_definition.replace('.gdf','')+'.hdf5'
        print('HDF5filename =',HDF5filename)
        rbf.hdf5.read_HDF5_beam_file(self.global_parameters['beam'], self.global_parameters['master_subdir'] + '/' + HDF5filename)
        # self.global_parameters['beam'].rotate_beamXZ(self.theta, preOffset=self.starting_offset)
        gptbeamfilename = self.particle_definition
        print('gptbeamfilename =',gptbeamfilename)
        rbf.gdf.write_gdf_beam_file(self.global_parameters['beam'], self.global_parameters['master_subdir'] + '/' + gptbeamfilename, normaliseZ=False)

class gpt_charge(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_charge, self).__init__(elementName='settotalcharge', elementType='gpt_charge', **kwargs)

    def write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + '('
        output += str(self.set) + ','
        output += str(-1*abs(self.charge)) + ');\n'
        return output

class gpt_setreduce(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_setreduce, self).__init__(elementName='setreduce', elementType='gpt_setreduce', **kwargs)

    def write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + '('
        output += str(self.set) + ','
        output += str(self.setreduce) + ');\n'
        return output

class gpt_accuracy(gpt_element):

    def __init__(self, accuracy=6, **kwargs):
        super(gpt_accuracy, self).__init__(elementName='accuracy', elementType='gpt_accuracy', **kwargs)
        self.accuracy = accuracy

    def write_GPT(self, *args, **kwargs):
        output = 'accuracy(' + str(self.accuracy) + ');\n'#'setrmacrodist(\"beam\","u",1e-9,0) ;\n'
        return output

class gpt_spacecharge(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_spacecharge, self).__init__(elementName='spacecharge', elementType='gpt_spacecharge', **kwargs)
        self.grids = getGrids()
        self.add_default('ngrids', None)

    def write_GPT(self, *args, **kwargs):
        output = ''#'setrmacrodist(\"beam\","u",1e-9,0) ;\n'
        if isinstance(self.space_charge_mode,str) and self.space_charge_mode.lower() == 'cathode':
            if self.ngrids is None:
                self.ngrids = self.grids.getGridSizes((self.npart/self.sample_interval))
            output += 'spacecharge3Dmesh("Cathode","RestMaxGamma",1000);\n'
        elif isinstance(self.space_charge_mode,str) and self.space_charge_mode.lower() == '3d':
            output += 'Spacecharge3Dmesh();\n'
        elif isinstance(self.space_charge_mode,str) and self.space_charge_mode.lower() == '2d':
            output += 'sc3dmesh();\n'
        else:
            output = ''
        return output

class gpt_tout(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_tout, self).__init__(elementName='tout', elementType='gpt_tout', **kwargs)

    def write_GPT(self, *args, **kwargs):
        self.starttime = 0 if self.starttime < 0 else self.starttime
        output = str(self.objectname) + '('
        if self.starttime is not None:
            output += str(self.starttime) + ','
        else:
            output += str(self.startpos) + '/c,'
        output += str(self.endpos) + ','
        output += str(self.step) + ');\n'
        return output

class gpt_csr1d(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_csr1d, self).__init__(elementName='csr1d', elementType='gpt_csr1d', **kwargs)

    def write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + '();\n'
        return output

class gpt_writefloorplan(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_writefloorplan, self).__init__(elementName='writefloorplan', elementType='gpt_writefloorplan', **kwargs)

    def write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + '(' + self.filename + ');\n'
        return output

class gpt_Zminmax(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_Zminmax, self).__init__(elementName='Zminmax', elementType='gpt_Zminmax', **kwargs)

    def write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + '(' + self.ECS + ', ' + str(self.zmin) + ', ' + str(self.zmax) + ');\n'
        return output

class gpt_forwardscatter(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_forwardscatter, self).__init__(elementName='forwardscatter', elementType='gpt_forwardscatter', **kwargs)

    def write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + '(' + self.ECS + ', \"' + str(self.name) + '\", ' + str(self.probability) + ');\n'
        return output

class gpt_scatterplate(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_scatterplate, self).__init__(elementName='scatterplate', elementType='gpt_scatterplate', **kwargs)

    def write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + '(' + self.ECS + ', ' + str(self.a) + ', ' + str(self.b) + ') scatter=\"' + str(self.model) + '\";\n'
        return output

class gpt_dtmaxt(gpt_element):

    def __init__(self, **kwargs):
        super(gpt_dtmaxt, self).__init__(elementName='dtmaxt', elementType='gpt_dtmaxt', **kwargs)

    def write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + '(' + str(self.tstart) + ', ' + str(self.tend) + ', ' + str(self.dtmax) + ');\n'
        return output
