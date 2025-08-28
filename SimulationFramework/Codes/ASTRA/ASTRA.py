"""
Simframe ASTRA Module

Various objects and functions to handle ASTRA lattices and commands. See `ASTRA manual`_ for more details.

    .. _ASTRA manual: https://www.desy.de/~mpyflo/Astra_manual/Astra-Manual_V3.2.pdf

Classes:
    - :class:`~SimulationFramework.Codes.ASTRA.ASTRA.astraLattice`: The ASTRA lattice object, used for\
    converting the :class:`~SimulationFramework.Framework_elements.frameworkObject` s defined in the\
    :class:`~SimulationFramework.Framework_elements.frameworkLattice` into a string representation of\
    the lattice suitable for an ASTRA input file.

    - :class:`~SimulationFramework.Codes.ASTRA.ASTRA.astra_header`: Class for defining the &HEADER portion\
    of the ASTRA input file.

    - :class:`~SimulationFramework.Codes.ASTRA.ASTRA.astra_newrun`: Class for defining the &NEWRUN portion\
    of the ASTRA input file.

    - :class:`~SimulationFramework.Codes.ASTRA.ASTRA.astra_charge`: Class for defining the &CHARGE portion\
    of the ASTRA input file.

    - :class:`~SimulationFramework.Codes.ASTRA.ASTRA.astra_output`: Class for defining the &OUTPUT portion\
    of the ASTRA input file.

    - :class:`~SimulationFramework.Codes.ASTRA.ASTRA.astra_errors`: Class for defining the &ERRORS portion\
    of the ASTRA input file.
"""

import os
from copy import deepcopy
import numpy as np
import lox
from lox.worker.thread import ScatterGatherDescriptor
from typing import ClassVar, Dict, List

from ...Framework_objects import (
    frameworkLattice,
    frameworkCounter,
    frameworkElement,
    getGrids,
    elementkeywords,
)
from ...Framework_elements import global_error, wakefield, screen
from ...FrameworkHelperFunctions import expand_substitution, saveFile
from ...Modules.merge_two_dicts import merge_two_dicts
from ...Modules import Beams as rbf
from ...Modules.Fields import field

section_header_text_ASTRA = {
    "cavities": {"header": "CAVITY", "bool": "LEField"},
    "wakefields": {"header": "WAKE", "bool": "LWAKE"},
    "solenoids": {"header": "SOLENOID", "bool": "LBField"},
    "quadrupoles": {"header": "QUADRUPOLE", "bool": "LQuad"},
    "dipoles": {"header": "DIPOLE", "bool": "LDipole"},
    "astra_newrun": {"header": "NEWRUN"},
    "astra_output": {"header": "OUTPUT"},
    "astra_charge": {"header": "CHARGE"},
    "global_error": {"header": "ERROR"},
    "apertures": {"header": "APERTURE", "bool": "LApert"},
}


class astraLattice(frameworkLattice):
    """
    Class for defining the ASTRA lattice object, used for
    converting the :class:`~SimulationFramework.Framework_elements.frameworkObject`s defined in the
    :class:`~SimulationFramework.Framework_elements.frameworkLattice` into a string representation of
    the lattice suitable for an ASTRA input file.
    """

    screen_threaded_function: ClassVar[ScatterGatherDescriptor] = (
        ScatterGatherDescriptor
    )
    """Function for converting all screen outputs from ASTRA into the SimFrame generic 
    :class:`~SimulationFramework.Modules.Beams.beam` object and writing files"""

    code: str = "astra"
    """String indicating the lattice object type"""

    allow_negative_drifts: bool = True
    """Flag to indicate whether negative drifts are allowed"""

    _bunch_charge: float | None = None
    """Bunch charge"""

    _toffset: float | None = None
    """Time offset of reference particle"""

    headers: Dict = {}
    """Headers to be included in the ASTRA lattice file"""

    starting_offset: List = [0, 0, 0]
    """Initial offset of first element"""

    starting_rotation: float = 0
    """Initial rotation of first element"""

    def __init__(self, *args, **kwargs):
        super(astraLattice, self).__init__(*args, **kwargs)
        self.starting_offset = (
            eval(expand_substitution(self, self.file_block["starting_offset"]))
            if "starting_offset" in self.file_block
            else [0, 0, 0]
        )

        # This calculated the starting rotation based on the input file and the number of dipoles
        self.starting_rotation = (
            -1 * self.elementObjects[self.start].global_rotation[2]
            if self.elementObjects[self.start].global_rotation is not None
            else 0
        )
        self.starting_rotation = (
            eval(expand_substitution(self, str(self.file_block["starting_rotation"])))
            if "starting_rotation" in self.file_block
            else self.starting_rotation
        )

        # Create a "newrun" block
        if "input" not in self.file_block:
            self.file_block["input"] = {}
        if "ASTRAsettings" not in self.globalSettings:
            self.globalSettings["ASTRAsettings"] = {}
        self.headers["newrun"] = astra_newrun(
            self.starting_offset,
            self.starting_rotation,
            global_parameters=self.global_parameters,
            **merge_two_dicts(
                self.file_block["input"], self.globalSettings["ASTRAsettings"]
            ),
        )
        # If the initial distribution is derived from a generator file, we should use that
        if (
            "input" in self.file_block
            and "particle_definition" in self.file_block["input"]
        ):
            if (
                self.file_block["input"]["particle_definition"]
                == "initial_distribution"
            ):
                self.headers["newrun"].input_particle_definition = "laser.astra"
                self.headers["newrun"].output_particle_definition = "laser_input.astra"
            else:
                self.headers["newrun"].input_particle_definition = self.file_block[
                    "input"
                ]["particle_definition"]
                self.headers["newrun"].output_particle_definition = (
                    self.objectname + "_input.astra"
                )
        else:
            self.headers["newrun"].input_particle_definition = (
                self.elementObjects[self.start].objectname + ".astra"
            )
            self.headers["newrun"].output_particle_definition = (
                self.objectname + "_input.astra"
            )

        # Create an "output" block
        if "output" not in self.file_block:
            self.file_block["output"] = {}
        self.headers["output"] = astra_output(
            self.screens_and_markers_and_bpms,
            self.starting_offset,
            self.starting_rotation,
            global_parameters=self.global_parameters,
            **merge_two_dicts(
                self.file_block["output"], self.globalSettings["ASTRAsettings"]
            ),
        )

        # Create a "charge" block
        if "charge" not in self.file_block:
            self.file_block["charge"] = {}
        if "charge" not in self.globalSettings:
            self.globalSettings["charge"] = {}
        space_charge_dict = merge_two_dicts(
            self.file_block["charge"],
            self.globalSettings["charge"],
        )
        self.headers["charge"] = astra_charge(
            global_parameters=self.global_parameters,
            **merge_two_dicts(
                space_charge_dict,
                self.globalSettings["ASTRAsettings"],
            ),
        )

        # Create an "error" block
        if "global_errors" not in self.file_block:
            self.file_block["global_errors"] = {}
        if "global_errors" not in self.globalSettings:
            self.globalSettings["global_errors"] = {}
        if "global_errors" in self.file_block or "global_errors" in self.globalSettings:
            self.global_error = global_error(
                objectname=self.objectname + "_global_error",
                objecttype="global_error",
                global_parameters=self.global_parameters,
            )
            self.headers["global_errors"] = astra_errors(
                element=self.global_error,
                global_parameters=self.global_parameters,
                **merge_two_dicts(
                    self.file_block["global_errors"],
                    self.globalSettings["global_errors"],
                ),
            )
        # print 'errors = ', self.file_block, self.headers['global_errors']

    @property
    def space_charge_mode(self) -> str:
        """
        The space charge type for ASTRA, i.e. "2D", "3D".

        Returns
        -------
        str
            The space charge type for ASTRA
        """
        return self.headers["charge"].space_charge_mode

    @space_charge_mode.setter
    def space_charge_mode(self, mode: str) -> None:
        """
        Sets the space charge mode for the &HEADER object

        Parameters
        ----------
        mode: str
            Space charge mode
        """
        self.headers["charge"].space_charge_mode = mode

    @property
    def sample_interval(self) -> int:
        """
        Factor by which to reduce the number of particles in the simulation, i.e. every 10th particle.

        Returns
        -------
        int
            The sampling interval `n_red` in ASTRA
        """
        return self._sample_interval

    @sample_interval.setter
    def sample_interval(self, interval: int) -> None:
        """
        Sets the factor by which to reduce the number of particles in the simulation in the &NEWRUN header,
        and scales the number of space charge bins in the &CHARGE header accordingly;
        see :func:`~SimulationFramework.Codes.ASTRA.ASTRA.astra_newrun.framework_dict`,
        :func:`~SimulationFramework.Codes.ASTRA.ASTRA.astra_charge.grid_size`.

        Parameters
        ----------
        interval:
            Sampling interval
        """
        # print('Setting new ASTRA sample_interval = ', interval)
        self._sample_interval = interval
        self.headers["newrun"].sample_interval = interval
        self.headers["charge"].sample_interval = interval

    @property
    def bunch_charge(self) -> float:
        """
        Bunch charge in coulombs

        Returns
        -------
        float:
            Bunch charge
        """
        return self._bunch_charge

    @bunch_charge.setter
    def bunch_charge(self, charge: float) -> None:
        """
        Sets the bunch charge for this object and also in :class:`~SimulationFramework.Codes.ASTRA.ASTRA.astra_newrun`.

        Parameters
        ----------
        charge: float
            Bunch charge in coulombs
        """
        # print('Setting new ASTRA sample_interval = ', interval)
        self._bunch_charge = charge
        self.headers["newrun"].bunch_charge = charge

    @property
    def toffset(self) -> float:
        """
        Get the time offset for the reference particle.

        Returns
        -------
        float
            The time offset in seconds
        """
        return self._toffset

    @toffset.setter
    def toffset(self, toffset: float) -> None:
        """
        Set the time offset for this object and the :class:`~SimulationFramework.Codes.ASTRA.ASTRA.astra_newrun` object.

        Parameters
        ----------
        toffset: float
            The time offset in seconds
        """
        # print('Setting new ASTRA sample_interval = ', interval)
        self._toffset = toffset
        self.headers["newrun"].toffset = 1e9 * toffset

    def writeElements(self) -> str:
        """
        Write the lattice elements defined in this object into an ASTRA-compatible format; see
        :attr:`~SimulationFramework.Framework_objects.frameworkLattice.elementObjects`.

        Elements are grouped together by type and counted using
        :class:`~SimulationFramework.Framework_objects.frameworkCounter`

        The appropriate headers required for ASTRA are written at the top of the file, see the `write_ASTRA`
        function in :class:`~SimulationFramework.Codes.ASTRA.ASTRA.astra_newrun`,
        :class:`~SimulationFramework.Codes.ASTRA.ASTRA.astra_header`,
        :class:`~SimulationFramework.Codes.ASTRA.ASTRA.astra_errors`.

        Returns
        -------
        str
            The lattice represented as a string compatible with ASTRA
        """
        fulltext = ""
        # Create objects for the newrun, output and charge blocks
        self.headers["output"].start_element = self.elementObjects[self.start]
        self.headers["output"].end_element = self.elementObjects[self.end]
        self.headers["output"].screens = self.screens_and_bpms
        # write the headers and their elements
        for header in self.headers:
            fulltext += self.headers[header].write_ASTRA(0) + "\n"
        # Initialise a counter object
        counter = frameworkCounter(sub={"kicker": "dipole", "collimator": "aperture"})
        for t in [
            ["cavities"],
            ["wakefields", "wakefields_and_cavity_wakefields"],
            ["solenoids"],
            ["quadrupoles"],
            ["dipoles", "dipoles_and_kickers"],
            ["apertures"],
        ]:
            fulltext += "&" + section_header_text_ASTRA[t[0]]["header"] + "\n"
            elements = getattr(self, t[-1])
            fulltext += (
                section_header_text_ASTRA[t[0]]["bool"]
                + " = "
                + str(len(elements) > 0)
                + "\n"
            )
            for element in elements:
                element.starting_offset = self.starting_offset
                element.starting_rotation = self.starting_rotation
                if element.objecttype == "cavity":
                    elemstr = element.write_ASTRA(
                        counter.counter(element.objecttype),
                        auto_phase=self.headers["newrun"].auto_phase,
                    )
                    if t[0] == "wakefields":
                        if hasattr(element, "wakefield_definition") and isinstance(
                            element.wakefield_definition, field
                        ):
                            original_properties = deepcopy(element.objectproperties)
                            original_properties.objectname = (
                                f"{element.objectname}_wake"
                            )
                            original_properties.objecttype = "wakefield"
                            setattr(
                                original_properties,
                                "field_definition",
                                original_properties.wakefield_definition,
                            )
                            wake_element = wakefield(
                                **{
                                    k: getattr(original_properties, k)
                                    for k in original_properties.model_fields_set
                                }
                            )
                            wake_element.cells = original_properties.get_cells()
                            elemstr = wake_element.write_ASTRA(
                                counter.counter("wakefields")
                            )
                        else:
                            elemstr = None
                else:
                    elemstr = element.write_ASTRA(counter.counter(element.objecttype))
                if elemstr is not None and not elemstr == "":
                    fulltext += elemstr + "\n"
                    if element.objecttype == "kicker":
                        counter.add(element.objecttype)
                    elif t == "wakefields":
                        counter.add("wakefields", element.cells)
                    elif (
                        element.objecttype == "aperture"
                        or element.objecttype == "collimator"
                    ):
                        counter.add(element.objecttype, element.number_of_elements)
                    else:
                        counter.add(element.objecttype)
            fulltext += "\n/\n"
        return fulltext

    def write(self) -> None:
        """
        Writes the ASTRA input file from :func:`~SimulationFramework.Codes.ASTRA.ASTRA.astraLattice.writeElements`
        to <master_subdir>/<self.objectname>.in.
        """
        code_file = (
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".in"
        )
        saveFile(code_file, self.writeElements())

    def preProcess(self) -> None:
        """
        Convert the beam file from the previous lattice section into ASTRA format and set the number of
        particles based on the input distribution, see
        :func:`~SimulationFramework.Codes.ASTRA.ASTRA.astra_newrun.hdf5_to_astra`.
        """
        super().preProcess()
        prefix = self.get_prefix()
        self.headers["newrun"].hdf5_to_astra(prefix, self.initial_twiss)
        self.headers["charge"].npart = len(self.global_parameters["beam"].x)

    @lox.thread
    def screen_threaded_function(
        self,
        scr: screen,
        objectname: str,
        cathode: bool,
        mult: int,
    ) -> None:
        """
        Convert output from ASTRA screen to HDF5 format

        Parameters
        ----------
        scr: :class:`~SimulationFramework.Elements.screen.screen`
            Screen object
        objectname: str
            Name of screen object
        cathode: bool
            True if beam was emitted from a cathode
        mult: int
            Multiplication factor for ASTRA-type filenames
        """
        return scr.astra_to_hdf5(objectname, cathode, mult)

    def find_ASTRA_filename(
        self,
        elem: frameworkElement,
        mult: int,
        lattice: str,
        master_run_no: int,
    ) -> bool:
        """
        Determine if an output was created by ASTRA for a given element based on its position and the filename.

        Parameters
        ----------
        elem: :class:`~SimulationFramework.Framework_objects.frameworkElement
            The element to be checked
        mult: int
            Multiplication factor for formatting ASTRA-type output
        lattice: str
            The lattice name
        master_run_no: int
            Master run number for ASTRA-type output (i.e. `<filename>.001`)

        Returns
        -------
        bool
            True if the file was found.
        """
        # print('find_ASTRA_filename', lattice, elem.middle[2], elem.zstart[2])
        for i in [0, -0.001, 0.001]:
            tempfilename = (
                lattice
                + "."
                + str(int(round((elem.middle[2] + i - elem.zstart[2]) * mult))).zfill(4)
                + "."
                + str(master_run_no).zfill(3)
            )
            # print(self.middle[2]+i-self.zstart[2], tempfilename, os.path.isfile(self.global_parameters['master_subdir'] + '/' + tempfilename))
            if os.path.isfile(
                self.global_parameters["master_subdir"] + "/" + tempfilename
            ):
                return True
        return False

    def get_screen_scaling(self) -> int:
        """
        Determine the screen scaling factor for screens and BPMs

        Returns
        -------
        int
            The scaling factor depending on the `master_run_no` parameter
        """
        for e in self.screens_and_bpms:
            if not self.starting_offset == [0, 0, 0]:
                e.zstart = self.elementObjects[self.start].start
            else:
                e.zstart = [0, 0, 0]
        master_run_no = (
            self.global_parameters["run_no"]
            if "run_no" in self.global_parameters
            else 1
        )
        for mult in [100, 1000, 10]:
            foundscreens = [
                self.find_ASTRA_filename(e, mult, self.objectname, master_run_no)
                for e in self.screens_and_bpms
            ]
            # print('get_screen_scaling', mult, foundscreens)
            if all(foundscreens):
                return mult
        return 100

    def postProcess(self) -> None:
        """
        Convert the beam file(s) from the ASTRA output into HDF5 format, see
        :func:`~SimulationFramework.Codes.ASTRA.ASTRA.astra_to_hdf5`.
        """
        super().postProcess()
        cathode = (
            self.headers["newrun"].input_particle_definition == "initial_distribution"
        )
        mult = self.get_screen_scaling()
        for e in self.screens_and_bpms:
            if not self.starting_offset == [0, 0, 0]:
                e.zstart = self.elementObjects[self.start].start
            else:
                e.zstart = [0, 0, 0]
            if not os.name == "nt":
                self.screen_threaded_function(
                    e, self.objectname, cathode=cathode, mult=mult
                )
            else:
                self.screen_threaded_function.scatter(
                    e, self.objectname, cathode=cathode, mult=mult
                )
        if os.name == "nt":
            self.screen_threaded_function.gather()
        self.astra_to_hdf5(cathode=cathode)

    def astra_to_hdf5(self, cathode: bool = False) -> None:
        """
        Convert the ASTRA particle distribution file to HDF5 format and write to `master_subdir`.

        Parameters
        ----------
        cathode: bool
            True if the beam was emitted from a cathode.
        """

        master_run_no = (
            self.global_parameters["run_no"]
            if "run_no" in self.global_parameters
            else 1
        )
        if not self.starting_offset == [0, 0, 0]:
            zstart = self.elementObjects[self.start].start
        else:
            zstart = [0, 0, 0]
        startpos = np.array(self.elementObjects[self.start].start) - np.array(zstart)
        endpos = np.array(self.elementObjects[self.end].end) - np.array(zstart)
        astrabeamfilename = (
            self.objectname
            + "."
            + str(int(round(endpos[2] * 100))).zfill(4)
            + "."
            + str(master_run_no).zfill(3)
        )
        if not os.path.isfile(
            self.global_parameters["master_subdir"] + "/" + astrabeamfilename
        ):
            # print('Can\'t find ASTRA beam file: ', astrabeamfilename)
            astrabeamfilename = (
                self.objectname
                + "."
                + str(int(round((endpos[2] - startpos[2]) * 1000))).zfill(4)
                + "."
                + str(master_run_no).zfill(3)
            )
            # print('Trying relative naming convention: ', astrabeamfilename)
        rbf.astra.read_astra_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + astrabeamfilename,
            normaliseZ=False,
        )
        rbf.hdf5.rotate_beamXZ(
            self.global_parameters["beam"],
            -1 * self.starting_rotation,
            preOffset=[0, 0, 0],
            postOffset=-1 * np.array(self.starting_offset),
        )
        HDF5filename = self.elementObjects[self.end].objectname + ".hdf5"
        toffset = self.global_parameters["beam"].toffset
        rbf.hdf5.write_HDF5_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + HDF5filename,
            centered=False,
            sourcefilename=astrabeamfilename,
            pos=self.elementObjects[self.end].middle,
            cathode=cathode,
            toffset=toffset,
        )
        # print('ASTRA/astra_to_hdf5', 'finished')


class astra_header(frameworkElement):
    """
    Generic class for generating ASTRA namelists
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(astra_header, self).__init__(
            *args,
            **kwargs,
        )

    def framework_dict(self) -> Dict:
        return dict()

    def write_ASTRA(self, n: int) -> str:
        """
        Write the text for the ASTRA namelist based on its :attr:`~framework_dict`.

        Parameters
        ----------
        n: int
            Index of the ASTRA element

        Returns
        -------
        str
            ASTRA-compatible string representing the namelist
        """
        keyword_dict = dict()
        for k in elementkeywords[self.objecttype]["keywords"]:
            if hasattr(self, k.lower()):
                if getattr(self, k.lower()) is not None:
                    keyword_dict[k.lower()] = {"value": getattr(self, k.lower())}
        output = "&" + section_header_text_ASTRA[self.objecttype]["header"] + "\n"
        output += (
            self._write_ASTRA_dictionary(
                merge_two_dicts(self.framework_dict(), keyword_dict), None
            )
            + "\n/\n"
        )
        return output


class astra_newrun(astra_header):
    """
    Class for generating the &NEWRUN namelist for ASTRA. See `ASTRA manual`_ for more details.
    """

    sample_interval: int = 1
    """Downsampling factor (as 2**(3 * sample_interval))"""

    run: int = 1
    """Run number"""
    head: str = "trial"
    """Run name"""
    lprompt: bool = False
    """If true a pause statement is included at the end
    of the run to avoid vanishing of the window in case of an error."""

    input_particle_definition: str = ""
    """Name of input particle definition"""

    output_particle_definition: str = ""
    """Name of output particle definition"""

    high_res: bool = True
    """If true, particle distributions are saved with increased accuracy."""

    auto_phase: bool = True
    """Phase RF cavities automatically"""

    bunch_charge: float | None = None
    """Bunch charge"""

    toffset: float | None = None
    """Time offset of reference particle"""

    track_all: bool = True
    """If false, only the reference particle will be tracked"""

    phase_scan: bool = False
    """If true, the RF phases of the cavities will be scanned between 0 and 360 degree.
    Results are saved in the PScan file. The tracking between cavities will be done
    with the user-defined phases."""

    check_ref_part: bool = False
    """If true, the run will be interrupted if the reference particle is lost during the on-
    and off-axis reference particle tracking."""

    h_max: float = 0.07
    """Maximum time step for the Runge-Kutta integration."""

    h_min: float = 0.07
    """Minimum time step for the Runge-Kutta integration."""

    def __init__(
        self,
        offset,
        rotation,
        objectname="newrun",
        objecttype="astra_newrun",
        *args,
        **kwargs,
    ):
        super(astra_header, self).__init__(
            offset=offset,
            rotation=rotation,
            objectname=objectname,
            objecttype=objecttype,
            *args,
            **kwargs,
        )

    def framework_dict(self) -> Dict:
        """
        Create formatted dictionary for generating ASTRA &NEWRUN namelist, based on the properties
        of the class.

        Returns
        -------
        Dict
            Formatted dictionary for ASTRA &NEWRUN
        """
        astradict = {
            "Distribution": {"value": "'" + self.output_particle_definition + "'"},
            "high_res": {"value": self.high_res, "default": True},
            "n_red": {"value": self.sample_interval, "default": 1},
            "auto_phase": {"value": self.auto_phase, "default": True},
            "Toff": {"value": self.toffset, "default": None},
            "track_all": {"value": self.track_all, "default": True},
            "phase_scan": {"value": self.phase_scan, "default": False},
            "check_ref_part": {"value": self.check_ref_part, "default": False},
            "h_min": {"value": self.h_min, "default": 0.07},
            "h_max": {"value": self.h_max, "default": 0.07},
        }
        if self.bunch_charge is not None:
            astradict["Qbunch"] = {"value": 1e9 * self.bunch_charge, "default": None}
        return astradict

    def hdf5_to_astra(
        self, prefix: str = "", initial_twiss: Dict = {"horizontal": {}, "vertical": {}}
    ) -> None:
        """
        Convert beam input file to ASTRA format and write to `master_subdir`.

        Parameters
        ----------
        prefix: str
            File location / name
        initial_twiss: Dict
            Dictionary containing initial Twiss parameters.
        """
        HDF5filename = (
            prefix + self.input_particle_definition.replace(".astra", "") + ".hdf5"
        )
        if os.path.isfile(expand_substitution(self, HDF5filename)):
            filepath = expand_substitution(self, HDF5filename)
        else:
            filepath = self.global_parameters["master_subdir"] + "/" + HDF5filename
        rbf.hdf5.read_HDF5_beam_file(
            self.global_parameters["beam"],
            filepath,
        )
        self.global_parameters["beam"].beam.rematchXPlane(**initial_twiss["horizontal"])
        self.global_parameters["beam"].beam.rematchYPlane(**initial_twiss["vertical"])
        rbf.hdf5.rotate_beamXZ(
            self.global_parameters["beam"],
            self.rotation,
            preOffset=self.offset,
        )
        astrabeamfilename = self.output_particle_definition
        rbf.astra.write_astra_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + astrabeamfilename,
            normaliseZ=False,
        )


class astra_output(astra_header):
    """
    Class for generating the &OUTPUT namelist for ASTRA. See `ASTRA manual`_ for more details.
    """

    lmagnetized: bool = False
    """If true, solenoid fields are neglected in the calculation of the beam emittance."""

    refs: bool = True
    """If true, output files according to Table 3 and Table 4 are generated. See `ASTRA manual`_"""

    emits: bool = True
    """If true, output files according to Table 3 and Table 4 are generated. See `ASTRA manual`_"""

    phases: bool = True
    """If true, output files according to Table 3 and Table 4 are generated. See `ASTRA manual`_"""

    high_res: bool = True
    """If true, particle distributions are saved with increased accuracy."""

    tracks: bool = True
    """If true, output files according to Table 3 and Table 4 are generated. See `ASTRA manual`_"""

    def __init__(
        self,
        screens,
        offset,
        rotation,
        objectname="output",
        objecttype="astra_output",
        *args,
        **kwargs,
    ):
        super(astra_header, self).__init__(
            screens=screens,
            offset=offset,
            rotation=rotation,
            objectname=objectname,
            objecttype=objecttype,
            *args,
            **kwargs,
        )

    def framework_dict(self) -> Dict:
        """
        Create formatted dictionary for generating ASTRA &OUTPUT namelist, based on the properties
        of the class.

        Returns
        -------
        Dict
            Formatted dictionary for ASTRA &OUTPUT

        """
        self.start_element.starting_offset = self.offset
        self.end_element.starting_offset = self.offset
        self.start_element.starting_rotation = self.rotation
        self.end_element.starting_rotation = self.rotation
        # print self.end_element.objectname, self.end_element.end, self.start_element.objectname, self.start_element.end
        keyworddict = dict(
            [
                [
                    "zemit",
                    {
                        "value": int(
                            (self.end_element.start[2] - self.start_element.start[2])
                            / 0.01
                        )
                    },
                ],
                ["zstart", {"value": self.start_element.start[2]}],
                ["zstop", {"value": self.end_element.end[2]}],
                ["Lsub_cor", {"value": True}],
                ["lmagnetized", {"value": self.lmagnetized}],
                ["refs", {"value": self.refs}],
                ["emits", {"value": self.emits}],
                ["phases", {"value": self.phases}],
                ["high_res", {"value": self.high_res}],
                ["tracks", {"value": self.tracks}],
            ]
        )
        for i, element in enumerate(self.screens, 1):
            element.starting_offset = self.offset
            element.starting_rotation = self.rotation
            keyworddict["Screen(" + str(i) + ")"] = {"value": element.middle[2]}
            # if abs(element.theta) > 0:
            # keyworddict['Scr_xrot('+str(i)+')'] = {'value': element.theta}
        return keyworddict


class astra_charge(astra_header):
    """
    Class for generating the &CHARGE namelist for ASTRA. See `ASTRA manual`_ for more details.
    """

    npart: int = 2 ** (3 * 5)
    """Number of particles"""

    sample_interval: int = 1
    """Downsampling interval calculated as 2 ** (3 * sample_interval)"""

    space_charge_mode: str = "False"
    """Space charge mode"""

    space_charge_2D: bool = True
    """Enable 2D space charge calculations"""

    space_charge_3D: bool = False
    """Enable 3D space charge calculations"""

    cathode: bool = False
    """Flag to indicate whether the bunch was emitted from a cathode."""

    min_grid: float = 3.424657e-13
    """Minimum grid length during emission."""

    max_scale: float = 0.1
    """If one of the space charge scaling factors exceeds the limit 1± max_scale a new
    space charge calculation is initiated."""

    cell_var: float = 2
    """Variation of the cell height in radial direction."""

    nrad: int | None = None
    """Number of grid cells in radial direction up to the bunch radius."""

    nlong_in: int | None = None
    """Maximum number of grid cells in longitudinal direction within the bunch length."""

    smooth_x: int = 2
    """Smoothing parameter for x-direction. Only for 3D FFT algorithm."""

    smooth_y: int = 2
    """Smoothing parameter for y-direction. Only for 3D FFT algorithm."""

    smooth_z: int = 2
    """Smoothing parameter for z-direction. Only for 3D FFT algorithm."""

    def __init__(
        self,
        objectname="charge",
        objecttype="astra_charge",
        *args,
        **kwargs,
    ):
        super(astra_header, self).__init__(
            objectname=objectname, objecttype=objecttype, *args, **kwargs
        )
        self.grids = getGrids()

    @property
    def space_charge(self) -> bool:
        """
        Flag to indicate whether space charge is enabled.

        Returns
        -------
        bool
            True if enabled
        """
        return not (
            self.space_charge_mode == "False"
            or self.space_charge_mode is False
            or self.space_charge_mode is None
            or self.space_charge_mode == "None"
        )

    @property
    def grid_size(self) -> int:
        """
        Get the number of space charge bins, see
        :func:`~SimulationFramework.Framework_objects.getGrids.getGridSizes`.

        Returns
        -------
        int
            The number of space charge bins based on the number of particles
        """
        # print('asking for grid sizes n = ', self.npart, ' is ', self.grids.getGridSizes(self.npart))
        return self.grids.getGridSizes(self.npart / self.sample_interval)

    def framework_dict(self) -> Dict:
        """
        Create formatted dictionary for generating ASTRA &CHARGE namelist, based on the properties
        of the class.

        Returns
        -------
        Dict
            Formatted dictionary for ASTRA &CHARGE
        """
        sc_dict = dict(
            [
                ["Lmirror", {"value": self.cathode, "default": False}],
                ["cell_var", {"value": self.cell_var, "default": self.cell_var}],
                ["min_grid", {"value": self.min_grid, "default": self.min_grid}],
                ["max_scale", {"value": self.max_scale, "default": self.max_scale}],
                ["smooth_x", {"value": self.smooth_x, "default": self.smooth_x}],
                ["smooth_y", {"value": self.smooth_y, "default": self.smooth_y}],
                ["smooth_z", {"value": self.smooth_z, "default": self.smooth_z}],
                ["LSPCH", {"value": self.space_charge, "default": True}],
                ["LSPCH3D", {"value": self.space_charge_3D, "default": True}],
            ]
        )
        # print('astra_charge', 'self.space_charge_2D', self.space_charge_2D, 'self.nrad', self.nrad, 'self.nlong_in', self.nlong_in)
        if self.space_charge_2D:
            sc_n_dict = dict(
                [
                    ["nrad", {"value": self.grid_size, "default": 32}],
                    ["nlong_in", {"value": self.grid_size, "default": 32}],
                ]
            )
            if hasattr(self, "nrad") and self.nrad is not None:
                sc_n_dict.update({"nrad": {"value": self.nrad}})
            if hasattr(self, "nlong_in") and self.nlong_in is not None:
                sc_n_dict.update({"nlong_in": {"value": self.nlong_in}})

        elif self.space_charge_3D:
            sc_n_dict = dict(
                [
                    ["nxf", {"value": self.grid_size, "default": 8}],
                    ["nyf", {"value": self.grid_size, "default": 8}],
                    ["nzf", {"value": self.grid_size, "default": 8}],
                ]
            )
        else:
            sc_n_dict = dict([])
        # print('astra_charge dict', merge_two_dicts(sc_dict, sc_n_dict))
        return merge_two_dicts(sc_dict, sc_n_dict)


class astra_errors(astra_header):
    """
    Class for generating the &ERROR namelist for ASTRA. See `ASTRA manual`_ for more details.
    """

    global_errors: bool = True
    """If false, no errors will be generated."""

    log_error: bool = True
    """If true an additional log file will be generated which contains the actual
    element and bunch setting"""

    generate_output: bool = True
    """If true an output file will be generated"""

    suppress_output: bool = False
    """If true any generation of output other than the error file is suppressed."""

    def __init__(
        self,
        element=None,
        objectname="astra_error",
        objecttype="global_error",
        *args,
        **kwargs,
    ):
        super(astra_errors, self).__init__(
            objectname=objectname,
            objecttype=objecttype,
            *args,
            **kwargs,
        )
        self._element = element

    def write_ASTRA(self, n):
        keyword_dict = {}
        conversion = dict(
            [a, b]
            for a, b in zip(
                elementkeywords[self.objecttype]["keywords"],
                elementkeywords[self.objecttype]["astra_keywords"],
            )
        )
        for k in elementkeywords[self.objecttype]["keywords"]:
            if hasattr(self, k.lower()):
                if getattr(self, k.lower()) is not None:
                    keyword_dict[conversion[k].lower()] = {
                        "value": getattr(self, k.lower())
                    }
        joineddict = merge_two_dicts(self.framework_dict(), keyword_dict)
        if len(joineddict) > 0:
            output = "&" + section_header_text_ASTRA[self.objecttype]["header"] + "\n"
            output += (
                self._write_ASTRA_dictionary(
                    merge_two_dicts(self.framework_dict(), keyword_dict), None
                )
                + "\n/\n"
            )
        else:
            output = ""
        return output
