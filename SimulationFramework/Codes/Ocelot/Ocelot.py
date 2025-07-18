from ...Framework_objects import frameworkLattice, getGrids
from ...Framework_elements import screen
from ...FrameworkHelperFunctions import expand_substitution
from ...Modules import Beams as rbf
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.track import track
from ocelot.cpbd.io import save_particle_array
from ocelot.cpbd.navi import Navigator
from ocelot.cpbd.sc import SpaceCharge, LSC
from ocelot.cpbd.csr import CSR
from ocelot.cpbd.wake3D import Wake, WakeTable
from ocelot.cpbd.physics_proc import SaveBeam
from ocelot.cpbd.beam import ParticleArray
from copy import deepcopy
from numpy import array, where, mean, savez_compressed
import os
from typing import Dict, List


class ocelotLattice(frameworkLattice):

    code: str = "ocelot"
    trackBeam: bool = True
    lat_obj: MagneticLattice | None = None
    pin: ParticleArray | None = None
    pout: ParticleArray | None = None
    tws: Dict | None = None
    names: List = []
    grids: getGrids | None = None
    oceglobal: Dict = {}
    unit_step: float = 0.01
    smooth: float = 0.01
    lsc: bool = True
    random_mesh: bool = True
    nbin_csr: int = 10
    mbin_csr: int = 5
    sigmamin_csr: float = 1e-5
    wake_sampling: int = 1000
    wake_filter: int = 10
    particle_definition: str | None = None
    final_screen: screen | None = None


    def __init__(self, *args, **kwargs):
        super(ocelotLattice, self).__init__(*args, **kwargs)
        self.oceglobal = (
            self.settings["global"]["OCELOTsettings"]
            if "OCELOTsettings" in list(self.settings["global"].keys())
            else self.oceglobal
        )
        for field in self.model_fields_set:
            if field in list(self.oceglobal.keys()):
                setattr(self, field, self.oceglobal[field])

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
            self.particle_definition = self.elementObjects[self.start].objectname

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
        self.final_screen = None
        if not self.endObject in self.screens_and_bpms:
            self.final_screen = self.endScreen(output_filename=self.endObject.objectname + ".npz")
        elements = self.createDrifts()
        mag_lat = []
        for element in list(elements.values()):
            if not element.subelement:
                try:
                    mag_lat.append(element.write_Ocelot())
                except Exception as e:
                    print('Ocelot writeElements error:', element.objectname, e)
        self.lat_obj = MagneticLattice(mag_lat)
        self.names = array([lat.id for lat in self.lat_obj.sequence])

    def write(self):
        self.writeElements()
        self.lat_obj.save_as_py_file(
            f'{self.global_parameters["master_subdir"]}/{self.objectname}.py'
        )

    def preProcess(self):
        super().preProcess()
        prefix = (
            self.file_block["input"]["prefix"]
            if "input" in self.file_block and "prefix" in self.file_block["input"]
            else ""
        )
        if self.trackBeam:
            self.hdf5_to_npz(prefix)
        else:
            HDF5filename = prefix + self.particle_definition + ".hdf5"
            rbf.hdf5.read_HDF5_beam_file(
                self.global_parameters["beam"],
                os.path.abspath(
                    self.global_parameters["master_subdir"] + "/" + HDF5filename
                ),
            )

    def hdf5_to_npz(self, prefix="", write=True):
        HDF5filename = prefix + self.particle_definition + ".hdf5"
        HDF5fnwpath = os.path.abspath(
            self.global_parameters["master_subdir"] + "/" + HDF5filename
        )
        rbf.hdf5.read_HDF5_beam_file(self.global_parameters["beam"], HDF5fnwpath)
        ocebeamfilename = HDF5fnwpath.replace("hdf5", "npz")
        self.pin = rbf.beam.write_ocelot_beam_file(
            self.global_parameters["beam"], ocebeamfilename, write=write
        )

    def run(self):
        """Run the code with input 'filename'"""
        navi = self.navi_setup()
        self.tws, self.pout = track(self.lat_obj, deepcopy(self.pin), navi=navi)

    def postProcess(self):
        super().postProcess()
        bfname = (
            f'{self.global_parameters["master_subdir"]}/{self.endObject.objectname}.npz'
        )
        save_particle_array(bfname, self.pout)
        rbf.beam.read_beam_file(self.global_parameters["beam"], bfname)
        rbf.hdf5.write_HDF5_beam_file(
            self.global_parameters["beam"],
            bfname.replace("npz", "hdf5"),
            centered=False,
            sourcefilename=bfname,
            pos=0.0,
            xoffset=mean(self.pout.x()),
            yoffset=mean(self.pout.y()),
            zoffset=self.pout.s,
        )
        twsdat = {e: [] for e in self.tws[0].__dict__.keys()}
        for t in self.tws:
            for k, v in t.__dict__.items():
                twsdat[k].append(v)
        savez_compressed(
            f'{self.global_parameters["master_subdir"]}/{self.objectname}_twiss.npz',
            **twsdat,
        )

    def navi_setup(self):
        navi_processes = []
        navi_locations_start = []
        navi_locations_end = []
        settings = self.settings
        self.unit_step = (
            settings["unit_step"] if "unit_step" in settings.keys() else self.unit_step
        )
        self.smooth = (
            self.oceglobal["smooth_param"]
            if "smooth_param" in list(self.oceglobal.keys())
            else 0.1
        )
        navi = Navigator(self.lat_obj, unit_step=self.unit_step)
        if self.lsc:
            lsc = self.physproc_lsc()
            navi_processes += [lsc]
            navi_locations_start += [self.lat_obj.sequence[0]]
            navi_locations_end += [self.lat_obj.sequence[-1]]
        if "charge" in list(self.file_block.keys()):
            if (
                "space_charge_mode" in list(self.file_block["charge"].keys())
                and self.file_block["charge"]["space_charge_mode"].lower() == "3d"
            ):
                gridsize = self.grids.getGridSizes(
                    (len(self.global_parameters["beam"].x) / self.sample_interval)
                )
                g1 = (
                    self.oceglobal["sc_grid"]
                    if "sc_grid" in list(self.oceglobal.keys())
                    else gridsize
                )
                grids = [g1 for _ in range(3)]
                sc = self.physproc_sc(grids)
                navi_processes += [sc]
                navi_locations_start += [self.lat_obj.sequence[0]]
                navi_locations_end += [self.lat_obj.sequence[-1]]
        if "csr" in list(self.file_block.keys()):
            csr, start, end = self.physproc_csr()
            for i in range(len(csr)):
                navi_processes += [csr[i]]
                navi_locations_start += [start[i]]
                navi_locations_end += [end[i]]
        for name, obj in self.elements.items():
            if (obj["objecttype"] == "cavity") and ("sub_elements" in list(obj.keys())):
                for sename, seobj in obj["sub_elements"].items():
                    if seobj["type"] == "wakefield":
                        wake, w_ind = self.physproc_wake(
                            name, seobj["field_definition"]
                        )
                        navi_processes += [wake]
                        navi_locations_start += [self.lat_obj.sequence[w_ind]]
                        navi_locations_end += [self.lat_obj.sequence[w_ind + 1]]
        for w in self.screens_and_bpms:
            name = w["output_filename"].replace(".sdds", "")
            loc = self.lat_obj.sequence[where(self.names == name)[0][0]]
            subdir = self.global_parameters["master_subdir"]
            navi_processes += [SaveBeam(filename=f"{subdir}/{name}.npz")]
            navi_locations_start += [loc]
            navi_locations_end += [loc]
        navi.add_physics_processes(
            navi_processes, navi_locations_start, navi_locations_end
        )
        return navi

    def physproc_lsc(self):
        lsc = LSC()
        lsc.smooth_param = self.smooth
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

    def physproc_wake(self, name, loc):
        wake = Wake(
            step=100, w_sampling=self.wake_sampling, filter_order=self.wake_filter
        )
        wake.wake_table = WakeTable(expand_substitution(self, loc))
        w_ind = where(self.names == name)[0][0]
        return [wake, w_ind]
