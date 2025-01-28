from ...Framework_objects import *
from ...Framework_elements import *
from ...FrameworkHelperFunctions import _rotation_matrix
from ...Modules import Beams as rbf
from ...Modules.merge_two_dicts import merge_two_dicts
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.track import track
from ocelot.cpbd.io import save_particle_array
from ocelot.cpbd.navi import Navigator
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
            if not element.subelement:
                mag_lat.append(element.write_Ocelot())
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
        HDF5fnwpath = os.path.abspath(self.global_parameters['master_subdir'] + '/' + HDF5filename)
        rbf.hdf5.read_HDF5_beam_file(self.global_parameters['beam'], HDF5fnwpath)
        # print('HDF5 Total charge', self.global_parameters['beam']['total_charge'])
        if self.bunch_charge is not None:
            self.q = charge(name='START', type='charge', global_parameters=self.global_parameters,**{'total': abs(self.bunch_charge)})
        else:
            self.q = charge(name='START', type='charge', global_parameters=self.global_parameters,**{'total': abs(self.global_parameters['beam'].Q)})
        ocebeamfilename = HDF5fnwpath.replace('hdf5', 'npz')
        self.pin = rbf.ocelot.write_ocelot_beam_file(self.global_parameters['beam'], ocebeamfilename, write=write)

    def run(self):
        """Run the code with input 'filename'"""
        navi = Navigator(self.lat_obj, unit_step=0.01)
        print(self.pin)
        for e in self.lat_obj.sequence:
            print(e)
        tws, self.pout = track(self.lat_obj, deepcopy(self.pin), navi=navi)

    def postProcess(self):
        bfname = f'{self.global_parameters["master_subdir"]}/{self.endObject.objectname}.npz'
        save_particle_array(bfname, self.pout)
        rbf.beam.read_beam_file(self.global_parameters['beam'], bfname)
        rbf.hdf5.write_HDF5_beam_file(self.global_parameters['beam'],
                                      bfname.replace('npz', 'hdf5'), centered=False,
                                      sourcefilename=bfname, pos=0.0,
                                      xoffset=np.mean(self.pout.x()),
                                      yoffset=np.mean(self.pout.y()),
                                      zoffset=self.pout.s,
                                      )
        # rbf.beam.write_HDF5_beam_file(bfname.replace('npz', 'hdf5'))
        # ocebeam = rbf.ocelot.read_ocelot_beam_file(self.global_parameters['beam'], bfname)
        # rbf.ocelot.write_ocelot_beam_file(self.global_parameters['beam'],
        #                                   self.global_parameters['master_subdir'] + '/' + ocebeamfilename)
