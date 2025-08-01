"""
Simframe Ocelot Module

Various objects and functions to handle OCELOT lattices and commands. See `Ocelot github`_ for more details.

    .. _Ocelot github: https://github.com/ocelot-collab/ocelot

Classes:
    - :class:`~SimulationFramework.Codes.Ocelot.Ocelot.ocelotLattice`: The Ocelot lattice object, used for\
    converting the :class:`~SimulationFramework.Framework_elements.frameworkObject` s defined in the\
    :class:`~SimulationFramework.Framework_elements.frameworkLattice` into an Ocelot lattice object,
    and for tracking through it.
"""
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
from ocelot.cpbd.beam import ParticleArray
from ocelot.cpbd.transformations.second_order import SecondTM
from ocelot.cpbd.transformations.kick import KickTM
from ocelot.cpbd.transformations.runge_kutta import RungeKuttaTM
from ocelot.cpbd.elements import Octupole, Undulator
from copy import deepcopy
from typing import Dict
from numpy import array, where, mean, savez_compressed, linspace, save
import os
from yaml import safe_load

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/ocelot_defaults.yaml",
    "r",
) as infile:
    oceglobal = safe_load(infile)
from typing import Dict, List


class ocelotLattice(frameworkLattice):
    """
    Class for defining the OCELOT lattice object, used for
    converting the :class:`~SimulationFramework.Framework_elements.frameworkObject`s defined in the
    :class:`~SimulationFramework.Framework_elements.frameworkLattice` into an Ocelot lattice object,
    and for tracking through it.
    """

    code: str = "ocelot"
    """String indicating the lattice object type"""

    trackBeam: bool = True
    """Flag to indicate whether to track the beam"""

    lat_obj: MagneticLattice = None
    """Lattice object as an Ocelot `MagneticLattice`_
    
    .. _MagneticLattice: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/magnetic_lattice.py
    """

    pin: ParticleArray = None
    """Initial particle distribution as an Ocelot `ParticleArray`_
    
    .. _ParticleArray: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/beam.py"""

    pout: ParticleArray = None
    """Final particle distribution as an Ocelot `ParticleArray`_"""

    tws: List = None
    """List containing Ocelot `Twiss`_ objects
    
    .. _Twiss: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/beam.py
    """

    names: List = None
    """Names of elements in the lattice"""

    grids: getGrids = None
    """Class for calculating the required number of space charge grids"""

    oceglobal: Dict = {}
    """Global settings for Ocelot, read in from `ocelotLattice.settings["global"]["OCELOTsettings"]` and
    `ocelot_defaults.yaml`"""

    unit_step: float = 0.01
    """Step for Ocelot `PhysProc`_ objects
    
    .. _PhysProc: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/physics_proc.py
    """

    smooth: float = 0.01
    """Smoothing parameter"""

    lsc: bool = True
    """Flag to enable LSC calculations"""

    random_mesh: bool = True
    """Random meshing for space charge calculations"""

    nbin_csr: int = 10
    """Number of longitudinal bins for CSR calculations"""

    mbin_csr: int = 5
    """Number of macroparticle bins for CSR calculations"""

    sigmamin_csr: float = 1e-5
    """Minimum size for CSR calculations"""

    wake_sampling: int = 1000
    """Number of samples for wake calculations"""

    wake_filter: int = 10
    """Filter parameter for wake calculations"""

    particle_definition: str = None
    """Initial particle distribution as a string"""

    final_screen: screen | None = None
    """Final screen object"""

    mbi_navi: MBI | None = None
    """Physics process for calculating microbunching gain"""

    def __init__(self, *args, **kwargs):
        super(ocelotLattice, self).__init__(*args, **kwargs)
        self.oceglobal = (
            self.settings["global"]["OCELOTsettings"]
            if "OCELOTsettings" in list(self.settings["global"].keys())
            else self.oceglobal
        )
        for f in self.model_fields_set:
            if f in list(self.oceglobal.keys()):
                setattr(self, field, self.oceglobal[f])

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
        self.grids = getGrids()

    def endScreen(self, **kwargs) -> screen:
        """
        Create a final screen object for dumping the particle output after tracking.

        Returns
        -------
        :class:`~SimulationFramework.Elements.screen.screen`
        """
        return screen(
            objectname=self.endObject.objectname,
            objecttype="screen",
            centre=self.endObject.centre,
            position_start=self.endObject.position_start,
            position_end=self.endObject.position_start,
            global_rotation=self.endObject.global_rotation,
            global_parameters=self.global_parameters,
            **kwargs,
        )

    def writeElements(self) -> None:
        """
        Create Ocelot objects for all the elements in the lattice and set the
        :attr:`~SimulationFramework.Codes.Ocelot.Ocelot.ocelotLattice.lat_obj` and
        :attr:`~SimulationFramework.Codes.Ocelot.Ocelot.ocelotLattice.names`.
        """
        self.final_screen = None
        if not self.endObject in self.screens_and_bpms:
            self.final_screen = self.endScreen(
                output_filename=self.endObject.objectname + ".npz"
            )
        elements = self.createDrifts()
        mag_lat = []
        for element in list(elements.values()):
            if not element.subelement:
                try:
                    mag_lat.append(element.write_Ocelot())
                except Exception as e:
                    print("Ocelot writeElements error:", element.objectname, e)
        method = {"global": SecondTM, Octupole: KickTM, Undulator: RungeKuttaTM}
        self.lat_obj = MagneticLattice(mag_lat, method=method)
        self.names = [str(x) for x in array([lat.id for lat in self.lat_obj.sequence])]

    def write(self) -> None:
        """
        Create the lattice object via :func:`~SimulationFramework.Codes.Ocelot.Ocelot.ocelotLattice.writeElements`
        and save it as a python file to `master_subdir`.
        """
        self.writeElements()
        self.lat_obj.save_as_py_file(
            f'{self.global_parameters["master_subdir"]}/{self.objectname}.py'
        )

    def preProcess(self) -> None:
        """
        Get the initial particle distribution defined in `file_block['input']['prefix']` if it exists.
        """
        super().preProcess()
        prefix = (
            self.file_block["input"]["prefix"]
            if "input" in self.file_block and "prefix" in self.file_block["input"]
            else ""
        )
        prefix = prefix if self.trackBeam else prefix + self.particle_definition
        self.hdf5_to_npz(prefix)

    def hdf5_to_npz(self, prefix: str="", write: bool=True) -> None:
        """
        Convert the initial HDF5 particle distribution to Ocelot format and set
        :attr:`~SimulationFramework.Codes.Ocelot.Ocelot.ocelotLattice.pin` accordingly.

        Parameters
        ----------
        prefix: str
            Prefix for particle file
        write: bool
            Flag to indicate whether to save the file
        """
        HDF5filename = prefix + self.particle_definition + ".hdf5"
        if os.path.isfile(expand_substitution(self, HDF5filename)):
            filepath = expand_substitution(self, HDF5filename)
        else:
            filepath = self.global_parameters["master_subdir"] + "/" + HDF5filename
        rbf.hdf5.read_HDF5_beam_file(
            self.global_parameters["beam"],
            os.path.abspath(filepath),
        )
        rbf.hdf5.write_HDF5_beam_file(
            self.global_parameters["beam"],
            f'{self.global_parameters["master_subdir"]}/{self.allElementObjects[self.start].objectname}.hdf5',
        )
        ocebeamfilename = HDF5filename.replace("hdf5", "ocelot.npz")
        self.pin = rbf.beam.write_ocelot_beam_file(
            self.global_parameters["beam"], ocebeamfilename, write=write
        )

    def run(self) -> None:
        """
        Run the code, and set :attr:`~tws` and :attr:`~pout`
        """
        navi = self.navi_setup()
        pin = deepcopy(self.pin)
        if self.sample_interval > 1:
            pin = pin.thin_out(nth=self.sample_interval)
        self.tws, self.pout = track(
            self.lat_obj, pin, navi=navi, calc_tws=True, twiss_disp_correction=True
        )

    def postProcess(self) -> None:
        """
        Convert the outputs from Ocelot to HDF5 format and save them to `master_subdir`.
        """
        super().postProcess()
        bfname = f'{self.global_parameters["master_subdir"]}/{self.endObject.objectname}.ocelot.npz'
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
            save(
                f'{self.global_parameters["master_subdir"]}/{self.objectname}_mbi.dat',
                self.mbi_navi.bf,
            )

    def navi_setup(self) -> Navigator:
        """
        Set up the physics processes for Ocelot (i.e. space charge, CSR, wakes etc).

        Returns
        -------
        Navigator
            An Ocelot `Navigator`_ object

        .. _Navigator: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/navi.py
        """
        navi_processes = []
        navi_locations_start = []
        navi_locations_end = []
        # settings = self.settings
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
                g1 = self.sc_grid if hasattr(self, "sc_grid") else gridsize
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
            self.mbi_navi = MBI(
                lattice=self.lat_obj,
                lamb_range=list(
                    linspace(
                        float(self.mbi["min"]),
                        float(self.mbi["max"]),
                        int(self.mbi["nstep"]),
                    )
                ),
                lsc=space_charge_set,
                csr=csr_set,
                slices=self.mbi["slices"],
            )
            # mbi1.step = self.unit_step
            self.mbi_navi.navi = deepcopy(navi)
            self.mbi_navi.lattice = deepcopy(self.lat_obj)
            self.mbi_navi.lsc = True
            navi.add_physics_proc(
                self.mbi_navi, self.lat_obj.sequence[0], self.lat_obj.sequence[-1]
            )
        for name, obj in self.elements.items():
            if obj.objecttype == "cavity":
                fieldstr = "wakefield_definition"
            elif obj.objecttype == "wakefield":
                fieldstr = "field_definition"
            else:
                fieldstr = None
            if fieldstr is not None:
                if getattr(obj, fieldstr) is not None:
                    wake, w_ind = self.physproc_wake(
                        name, getattr(obj, fieldstr), obj.n_cells
                    )
                    navi_processes += [wake]
                    navi_locations_start += [self.lat_obj.sequence[w_ind]]
                    navi_locations_end += [self.lat_obj.sequence[w_ind + 1]]
        for w in self.screens_and_bpms:
            name = w.output_filename.replace(".sdds", "")
            loc = self.lat_obj.sequence[self.names.index(name)]
            subdir = self.global_parameters["master_subdir"]
            navi_processes += [SaveBeam(filename=f"{subdir}/{name}.ocelot.npz")]
            navi_locations_start += [loc]
            navi_locations_end += [loc]
        navi.add_physics_processes(
            navi_processes, navi_locations_start, navi_locations_end
        )
        return navi

    def physproc_lsc(self) -> LSC:
        """
        Get an Ocelot `LSC`_ physics process

        Returns
        -------
        LSC
            The Ocelot LSC PhysProc

        .. LSC: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/sc.py
        """
        lsc = LSC()
        lsc.smooth_param = self.smooth_param
        return lsc

    def physproc_sc(self, grids: List[int]) -> SpaceCharge:
        """
        Get an Ocelot `SpaceCharge`_ physics process

        Parameters
        ----------
        grids: List[int]
            The space charge grid number in x,y,z

        Returns
        -------
        SpaceCharge
            The Ocelot SpaceCharge PhysProc

        .. _SpaceCharge: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/sc.py
        """
        sc = SpaceCharge(step=1)
        sc.nmesh_xyz = grids
        sc.random_mesh = self.random_mesh
        return sc

    def physproc_csr(self) -> tuple:
        """
        Get Ocelot `CSR`_ physics processes based on the start and end positions provided in `file_block`.
        If these are not provided, just include CSR for the entire lattice.

        Returns
        -------
        tuple
            A list of CSR PhysProcs, and their start and end positions

        .. _CSR: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/csr.py
        """
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

    def physproc_wake(
            self,
            name: str,
            loc: field | str,
            ncell: int,
    ) -> tuple:
        """
        Get an Ocelot `Wake`_ physics process based on the wakefield provided.

        Parameters
        ----------
        name: str
            Name of lattice object associated with the wake
        loc: :class:`~SimulationFramework.Modules.Fields.field` or str
            If `field`, then write the field file to ASTRA format
        ncell: int
            Number of cells, which provides a multiplication factor for the wake

        Returns
        -------
        tuple
            A Wake PhysProc, and its index in the lattice

        .. _Wake: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/wake.py
        """
        if isinstance(loc, field.field):
            loc = loc.write_field_file(code="astra")
        wake = Wake(
            step=100,
            w_sampling=self.wake_sampling,
            filter_order=self.wake_filter,
        )
        wake.factor = ncell * self.wake_factor
        wake.wake_table = WakeTable(expand_substitution(self, loc))
        w_ind = self.names.index(name)
        return [wake, w_ind]
