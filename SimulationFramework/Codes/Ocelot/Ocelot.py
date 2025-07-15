from ...Framework_objects import frameworkLattice, getGrids
from ...Framework_elements import screen
from ...FrameworkHelperFunctions import expand_substitution
from ...Modules import Beams as rbf
from ...Modules.Fields import field
from .mbi import MBI
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.track import track
from ocelot.cpbd.io import save_particle_array, load_particle_array
from ocelot.cpbd.navi import Navigator
from ocelot.cpbd.sc import SpaceCharge, LSC
from ocelot.cpbd.csr import CSR
from ocelot.cpbd.wake3D import Wake, WakeTable
from ocelot.cpbd.physics_proc import SaveBeam
from ocelot.cpbd.transformations.second_order import SecondTM
from ocelot.cpbd.transformations.kick import KickTM
from ocelot.cpbd.transformations.runge_kutta import RungeKuttaTM
from ocelot.cpbd.elements import Octupole, Undulator
from copy import deepcopy
from typing import Dict
from numpy import array, where, mean, savez_compressed, linspace, save
import os
import re
from yaml import safe_load

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/ocelot_defaults.yaml",
    "r",
) as infile:
    oceglobal = safe_load(infile)


class ocelotLattice(frameworkLattice):
    def __init__(self, *args, **kwargs):
        super(ocelotLattice, self).__init__(*args, **kwargs)
        self.code = "ocelot"
        self.allow_negative_drifts = False
        self.bunch_charge = None
        self.trackBeam = True
        self.betax = None
        self.betay = None
        self.alphax = None
        self.alphay = None
        self.commandFiles = {}
        self.lat_obj = None
        self.pin = None
        self.pout = None
        self.tws = None
        self.names = None
        self.mbi = {"min": 1e-6, "max": 50e-6, "nstep": 30, "slices": 0, "set_mbi": False}
        self.mbi_navi = None
        self.grids = getGrids()
        self.oceglobal = deepcopy(oceglobal)
        if "OCELOTsettings" in list(self.settings["global"].keys()):
            for k, v in self.settings["global"]["OCELOTsettings"].items():
                if isinstance(v, Dict):
                    for k1, v1 in v.items():
                        self.oceglobal[k].update({k1: v1})
                else:
                    self.oceglobal.update({k: v})
        for k, v in self.oceglobal.items():
            setattr(self, k, v)
        self.lsc = (
            self.oceglobal["lsc"]
            if "lsc" in list(self.oceglobal.keys())
            else self.lscDrifts
        )
        if (
            "input" in self.file_block
            and "particle_definition" in self.file_block["input"]
        ):
            if (
                self.file_block["input"]["particle_definition"]
                == "initial_distribution"
            ):
                self.particle_definition = "laser"
            else:
                self.particle_definition = self.file_block["input"][
                    "particle_definition"
                ]
        else:
            self.particle_definition = self.allElementObjects[self.start].objectname

    def endScreen(self, **kwargs):
        return screen(
            name=self.endObject.objectname,
            type="screen",
            centre=self.endObject.centre,
            position_start=self.endObject.position_start,
            position_end=self.endObject.position_start,
            global_rotation=self.endObject.global_rotation,
            global_parameters=self.global_parameters,
            **kwargs,
        )

    def writeElements(self):
        self.w = None
        if self.endObject not in self.screens_and_bpms:
            self.w = self.endScreen(output_filename=f"{self.endObject.objectname}.ocelot.npz")
        elements = self.createDrifts()
        mag_lat = []
        for element in list(elements.values()):
            if not element.subelement:
                try:
                    mag_lat.append(element.write_Ocelot())
                except Exception as e:
                    print('Ocelot writeElements error:', element.objectname, e)
        method = {"global": SecondTM, Octupole: KickTM, Undulator: RungeKuttaTM}
        self.lat_obj = MagneticLattice(mag_lat, method=method)
        self.names = array([lat.id for lat in self.lat_obj.sequence])

    def write(self):
        self.writeElements()
        self.lat_obj.save_as_py_file(
            f'{self.global_parameters["master_subdir"]}/{self.objectname}.py'
        )

    def preProcess(self):
        prefix = (
            self.file_block["input"]["prefix"]
            if "input" in self.file_block and "prefix" in self.file_block["input"]
            else ""
        )
        prefix = prefix if self.trackBeam else prefix + self.particle_definition
        self.hdf5_to_npz(prefix)

    def hdf5_to_npz(self, prefix="", write=True):
        HDF5filename = prefix + self.particle_definition
        rootname = re.split(r"[\\/]", HDF5filename)
        rbf.hdf5.read_HDF5_beam_file(self.global_parameters["beam"], HDF5filename)
        rbf.hdf5.write_HDF5_beam_file(
            self.global_parameters["beam"],
            f'{self.global_parameters["master_subdir"]}/{self.allElementObjects[self.start].objectname}.hdf5',
        )
        ocebeamfilename = HDF5filename.replace("hdf5", "ocelot.npz")
        self.pin = rbf.beam.write_ocelot_beam_file(
            self.global_parameters["beam"], ocebeamfilename, write=write
        )

    def run(self):
        """Run the code with input 'filename'"""
        navi = self.navi_setup()
        pin = deepcopy(self.pin)
        if self.sample_interval > 1:
            pin = pin.thin_out(nth=self.sample_interval)
        self.tws, self.pout = track(self.lat_obj, pin, navi=navi)

    def postProcess(self):
        bfname = (
            f'{self.global_parameters["master_subdir"]}/{self.endObject.objectname}.ocelot.npz'
        )
        save_particle_array(bfname, self.pout)
        for elem in self.screens_and_bpms + [self.endObject]:
            ocebeamname = f'{self.global_parameters["master_subdir"]}/{elem.objectname}.ocelot.npz'
            parray = load_particle_array(ocebeamname)
            beam = rbf.beam(ocebeamname)
            rbf.hdf5.write_HDF5_beam_file(
                beam,
                ocebeamname.replace("ocelot.npz", "hdf5"),
                centered=False,
                sourcefilename=ocebeamname,
                pos=0.0,
                xoffset=mean(parray.x()),
                yoffset=mean(parray.y()),
                zoffset=[parray.s],
            )
        twsdat = {e: [] for e in self.tws[0].__dict__.keys()}
        for t in self.tws:
            for k, v in t.__dict__.items():
                # Offset the s values to the start of the lattice
                if k == "s":
                    v += self.startObject["position_start"][2]
                twsdat[k].append(v)
        savez_compressed(
            f'{self.global_parameters["master_subdir"]}/{self.objectname}_twiss.npz',
            **twsdat,
        )
        if self.mbi_navi is not None:
            save(f'{self.global_parameters["master_subdir"]}/{self.objectname}_mbi.dat', self.mbi_navi.bf)

    def navi_setup(self):
        navi_processes = []
        navi_locations_start = []
        navi_locations_end = []
        settings = self.settings
        navi = Navigator(self.lat_obj, unit_step=self.unit_step)
        if self.lsc:
            lsc = self.physproc_lsc()
            navi_processes += [lsc]
            navi_locations_start += [self.lat_obj.sequence[0]]
            navi_locations_end += [self.lat_obj.sequence[-1]]
        space_charge_set = False
        csr_set = False
        if "charge" in list(self.file_block.keys()):
            if (
                "space_charge_mode" in list(self.file_block["charge"].keys())
                and self.file_block["charge"]["space_charge_mode"].lower() == "3d"
            ):
                gridsize = self.grids.getGridSizes(
                    (len(self.global_parameters["beam"].x) / self.sample_interval)
                )
                g1 = (
                    self.sc_grid
                    if hasattr(self, "sc_grid")
                    else gridsize
                )
                grids = [g1 for _ in range(3)]
                sc = self.physproc_sc(grids)
                navi_processes += [sc]
                navi_locations_start += [self.lat_obj.sequence[0]]
                navi_locations_end += [self.lat_obj.sequence[-1]]
                space_charge_set = True
        if "csr" in list(self.file_block.keys()):
            csr, start, end = self.physproc_csr()
            for i in range(len(csr)):
                navi_processes += [csr[i]]
                navi_locations_start += [start[i]]
                navi_locations_end += [end[i]]
        if self.mbi["set_mbi"]:
            print(self.mbi)
            self.mbi_navi = MBI(
                lattice=self.lat_obj,
                lamb_range=list(linspace(float(self.mbi["min"]), float(self.mbi["max"]), int(self.mbi["nstep"]))),
                lsc=space_charge_set,
                csr=csr_set,
                slices=self.mbi["slices"],
            )
            # mbi1.step = self.unit_step
            self.mbi_navi.navi = deepcopy(navi)
            self.mbi_navi.lattice = deepcopy(self.lat_obj)
            self.mbi_navi.lsc = True
            navi.add_physics_proc(self.mbi_navi, self.lat_obj.sequence[0], self.lat_obj.sequence[-1])
        for name, obj in self.elements.items():
            if (obj["objecttype"] == "cavity") and hasattr(obj, "wakefield_definition"):
                if isinstance(obj.wakefield_definition, field):
                    wfname = obj.generate_field_file_name(obj.wakefield_definition, code="astra")
                    wake, w_ind = self.physproc_wake(
                        name, f'{self.global_parameters["master_subdir"]}/{wfname}', obj.n_cells,
                    )
                    navi_processes += [wake]
                    navi_locations_start += [self.lat_obj.sequence[w_ind]]
                    navi_locations_end += [self.lat_obj.sequence[w_ind + 1]]
        for w in self.screens_and_bpms:
            name = w["output_filename"].replace(".sdds", "")
            loc = self.lat_obj.sequence[where(self.names == name)[0][0]]
            subdir = self.global_parameters["master_subdir"]
            navi_processes += [SaveBeam(filename=f"{subdir}/{name}.ocelot.npz")]
            navi_locations_start += [loc]
            navi_locations_end += [loc]
        navi.add_physics_processes(
            navi_processes, navi_locations_start, navi_locations_end
        )
        return navi

    def physproc_lsc(self):
        lsc = LSC()
        lsc.smooth_param = self.smooth_param
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
        if ("start" in list(self.file_block["csr"].keys())) and (
            "end" in list(self.file_block["csr"].keys())
        ):
            start = self.file_block["csr"]["start"]
            st = [start] if isinstance(start, str) else start
            end = self.file_block["csr"]["end"]
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

    def physproc_wake(self, name, loc, ncell):
        wake = Wake(
            step=100, w_sampling=self.wake_sampling, filter_order=self.wake_filter,
        )
        wake.factor = ncell * self.wake_factor
        wake.wake_table = WakeTable(expand_substitution(self, loc))
        w_ind = where(self.names == name)[0][0]
        return [wake, w_ind]
