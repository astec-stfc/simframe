from ...Framework_objects import getGrids
from ...Framework_elements import *
from ...FrameworkHelperFunctions import expand_substitution
from ...Modules import Beams as rbf
from ...Modules.merge_two_dicts import merge_two_dicts
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.track import track
from ocelot.cpbd.io import save_particle_array
from ocelot.cpbd.navi import Navigator
from ocelot.cpbd.sc import SpaceCharge, LSC
from ocelot.cpbd.csr import CSR
from ocelot.cpbd.wake3D import Wake, WakeTable
from ocelot.cpbd.physics_proc import SaveBeam
from copy import deepcopy
from numpy import array, where


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
        self.names =  None
        self.grids = getGrids()
        self.sample_interval = 1
        self.oceglobal = self.settings['global']['OCELOTsettings'] if 'OCELOTsettings' in list(self.settings['global'].keys()) else {}
        self.unit_step = self.oceglobal['unit_step'] if 'unit_step' in list(self.oceglobal.keys()) else 0.01
        self.smooth = self.oceglobal['smooth_param'] if 'smooth_param' in list(self.oceglobal.keys()) else 0.1
        self.lsc = self.oceglobal['lsc'] if 'lsc' in list(self.oceglobal.keys()) else self.lscDrifts
        self.random_mesh = self.oceglobal['random_mesh'] if 'random_mesh' in list(self.oceglobal.keys()) else True
        self.nbin_csr = self.oceglobal['nbin_csr'] if 'nbin_csr' in list(self.oceglobal.keys()) else 10
        self.mbin_csr = self.oceglobal['mbin_csr'] if 'mbin_csr' in list(self.oceglobal.keys()) else 5
        self.sigmamin_csr = self.oceglobal['sigmamin_csr'] if 'sigmamin_csr' in list(
            self.oceglobal.keys()) else 1e-5
        self.wake_sampling = self.oceglobal['wake_sampling'] if 'wake_sampling' in list(
            self.oceglobal.keys()) else 1000
        self.wake_filter = self.oceglobal['wake_filter'] if 'wake_filter' in list(
            self.oceglobal.keys()) else 10

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
        self.names = array([lat.id for lat in self.lat_obj.sequence])

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
        navi = self.navi_setup()
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

    def navi_setup(self):
        settings = self.settings
        self.unit_step = settings['unit_step'] if 'unit_step' in settings.keys() else self.unit_step
        self.smooth = self.oceglobal['smooth_param'] if 'smooth_param' in list(self.oceglobal.keys()) else 0.1
        navi = Navigator(self.lat_obj, unit_step=self.unit_step)
        if self.lsc:
            lsc = self.physproc_lsc()
            navi.add_physics_proc(lsc, self.lat_obj.sequence[0], self.lat_obj.sequence[-1])
        if 'charge' in list(self.file_block.keys()):
            if 'space_charge_mode' in list(self.file_block['charge'].keys()):
                gridsize = self.grids.getGridSizes((len(self.global_parameters['beam'].x) / self.sample_interval))
                g1 = self.oceglobal['sc_grid'] if 'sc_grid' in list(self.oceglobal.keys()) else gridsize
                grids = [g1 for _ in range(3)]
                sc = self.physproc_sc(grids)
                navi.add_physics_proc(sc, self.lat_obj.sequence[0], self.lat_obj.sequence[-1])
        if 'csr' in list(self.file_block.keys()):
            csr, start, end = self.physproc_csr()
            for i in range(len(csr)):
                navi.add_physics_proc(csr[i], start[i], end[i])
        for name, obj in self.elements.items():
            if (obj['objecttype'] == 'cavity') and ('sub_elements' in list(obj.keys())):
                for sename, seobj in obj['sub_elements'].items():
                    if seobj['type'] == 'longitudinal_wakefield':
                        wake, w_ind = self.physproc_wake(name, seobj['field_definition'])
                        navi.add_physics_proc(wake, self.lat_obj.sequence[w_ind], self.lat_obj.sequence[w_ind + 1])
        for w in self.screens_and_bpms:
            name = w['output_filename'].replace('.sdds', '')
            loc = self.lat_obj.sequence[where(self.names == name)[0][0]]
            navi.add_physics_proc(SaveBeam(filename=f'{name}.npz'), loc, loc)
        return navi

    def physproc_lsc(self):
        lsc = LSC()
        lsc.smooth_param=self.smooth
        return lsc

    def physproc_sc(self, grids):
        sc = SpaceCharge(step=1)
        sc.nmesh_xyz = grids
        sc.random_mesh = self.random_mesh
        return sc

    def physproc_csr(self):
        csrlist = []
        stlist = []
        enlist = []
        if ('start' in list(self.file_block['csr'].keys())) and ('end' in list(self.file_block['csr'].keys())):
            start = self.file_block['csr']['start']
            st = [start] if isinstance(start, str) else start
            end = self.file_block['csr']['end']
            en = [end] if isinstance(end, str) else end
            for i in range(len(st)):
                stelem = self.lat_obj.sequence[where(self.names == st[i])[0][0]]
                enelem = self.lat_obj.sequence[where(self.names == en[i])[0][0]]
                csr = CSR()
                csr.n_bin = self.nbin_csr
                csr.m_bin = self.mbin_csr
                csr.sigma_min = self.sigmamin_csr
                csrlist.append(csr)
                stlist.append(stelem)
                enlist.append(enelem)
        else:
            csr = CSR()
            csr.n_bin = self.nbin_csr
            csr.m_bin = self.mbin_csr
            csr.sigma_min = self.sigmamin_csr
            stlist = [self.lat_obj.sequence[0]]
            enlist = [self.lat_obj.sequence[-1]]
        return [csrlist, stlist, enlist]

    def physproc_wake(self, name, loc):
        wake = Wake(step=100, w_sampling=self.wake_sampling, filter_order=self.wake_filter)
        wake.wake_table = WakeTable(expand_substitution(self, loc))
        w_ind = where(self.names == name)[0][0]
        return [wake, w_ind]

