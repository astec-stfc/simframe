"""
Simframe ELEGANT Module

Various objects and functions to handle ELEGANT lattices and commands. See `Elegant manual`_ for more details.

    .. _Elegant manual: https://ops.aps.anl.gov/manuals/elegant_latest/elegant.html

Classes:
    - :class:`~SimulationFramework.Codes.Elegant.Elegant.elegantLattice`: The ELEGANT lattice object, used for\
    converting the :class:`~SimulationFramework.Framework_elements.frameworkObject` s defined in the\
    :class:`~SimulationFramework.Framework_elements.frameworkLattice` into a string representation of\
    the lattice suitable for ELEGANT input and lattice files.

    - :class:`~SimulationFramework.Codes.Elegant.Elegant.elegantCommandFile`: Base class for defining\
    commands in an ELEGANT input file.

    - :class:`~SimulationFramework.Codes.Elegant.Elegant.elegant_global_settings_command`: Class for defining the\
    &global_settings portion of the ELEGANT input file.

    - :class:`~SimulationFramework.Codes.Elegant.Elegant.elegant_run_setup_command`: Class for defining the\
    &run_setup portion of the ELEGANT input file.

    - :class:`~SimulationFramework.Codes.Elegant.Elegant.elegant_error_elements_command`: Class for defining the\
    &error_elements portion of the ELEGANT input file.

    - :class:`~SimulationFramework.Codes.Elegant.Elegant.elegant_error_elements_command`: Class for defining the\
    &error_elements portion of the ELEGANT input file.

    - :class:`~SimulationFramework.Codes.Elegant.Elegant.elegant_scan_elements_command`: Class for defining the\
    &scan_elements portion of the ELEGANT input file.

    - :class:`~SimulationFramework.Codes.Elegant.Elegant.elegant_run_control_command`: Class for defining the\
    &run_control portion of the ELEGANT input file.

    - :class:`~SimulationFramework.Codes.Elegant.Elegant.elegant_twiss_output_command`: Class for defining the\
    &twiss_output portion of the ELEGANT input file.

    - :class:`~SimulationFramework.Codes.Elegant.Elegant.elegant_floor_coordinates_command`: Class for defining the\
    &floor_coordinates portion of the ELEGANT input file.

    - :class:`~SimulationFramework.Codes.Elegant.Elegant.elegant_matrix_output_command`: Class for defining the\
    &matrix_output portion of the ELEGANT input file.

    - :class:`~SimulationFramework.Codes.Elegant.Elegant.elegant_sdds_beam_command`: Class for defining the\
    &sdds_beam portion of the ELEGANT input file.

    - :class:`~SimulationFramework.Codes.Elegant.Elegant.elegant_track_command`: Class for defining the\
    &track portion of the ELEGANT input file.

    - :class:`~SimulationFramework.Codes.Elegant.Elegant.elegant_track_command`: Class for defining the\
    &track portion of the ELEGANT input file.

    - :class:`~SimulationFramework.Codes.Elegant.Elegant.elegantOptimisation`: Class for defining the\
    commands for optimization in the ELEGANT input file.

    - :class:`~SimulationFramework.Codes.Elegant.Elegant.sddsFile`: Class for creating, modifying and\
    saving SDDS files.
"""

import os
from copy import copy
import subprocess
import numpy as np
from warnings import warn
try:
    import sdds
except Exception:
    print("No SDDS available!")
import lox
from lox.worker.thread import ScatterGatherDescriptor
from typing import ClassVar, Callable
from ...Framework_objects import (
    frameworkLattice,
    frameworkCommand,
    elementkeywords,
    keyword_conversion_rules_elegant,
)
from ...Framework_elements import charge, screen
from ...FrameworkHelperFunctions import saveFile, expand_substitution
from ...Modules import Beams as rbf
from typing import Dict, List, Any


class elegantLattice(frameworkLattice):
    """
    Class for defining the ELEGANT lattice object, used for
    converting the :class:`~SimulationFramework.Framework_elements.frameworkObject`s defined in the
    :class:`~SimulationFramework.Framework_elements.frameworkLattice` into a string representation of
    the lattice suitable for an ELEGANT input file.
    """

    screen_threaded_function: ClassVar[ScatterGatherDescriptor] = (
        ScatterGatherDescriptor
    )
    """Function for converting all screen outputs from ELEGANT into the SimFrame generic 
    :class:`~SimulationFramework.Modules.Beams.beam` object and writing files"""

    code: str = "elegant"
    """String indicating the lattice object type"""

    allow_negative_drifts: bool = False
    """Flag to indicate whether negative drifts are allowed"""

    particle_definition: str | None = None
    """String representation of the initial particle distribution"""

    bunch_charge: float | None = None
    """Bunch charge"""

    q: charge | None = None
    """:class:`~SimulationFramework.Elements.charge.charge` object"""

    trackBeam: bool = True
    """Flag to indicate whether to track the beam"""

    betax: float | None = None
    """Initial beta_x for matching"""

    betay: float | None = None
    """Initial beta_y for matching"""

    alphax: float | None = None
    """Initial alpha_x for matching"""

    alphay: float | None = None
    """Initial alpha_y for matching"""

    commandFiles: Dict = {}
    """Dictionary of :class:`~SimulationFramework.Codes.Elegant.Elegant.elegantCommandFile`
    objects for writing to the ELEGANT input file"""

    final_screen: screen | None = None
    """:class:`SimulationFramework.Elements.screen.screen` object at the end of the line"""

    commandFilesOrder: List = []
    """Order in which commands are to be written in the ELEGANT input file"""

    def __init__(self, *args, **kwargs):
        super(elegantLattice, self).__init__(*args, **kwargs)
        self.particle_definition = self.elementObjects[self.start].objectname
        self.q = charge(
            objectname="START",
            objecttype="charge",
            global_parameters=self.global_parameters,
            **{"total": 250e-12},
        )

    def endScreen(self, **kwargs) -> screen:
        """
        Create a final screen object for dumping the particle output after tracking.

        Returns
        -------
        :class:`~SimulationFramework.Elements.screen.screen`
        """
        return screen(
            objectname="end",
            objecttype="screen",
            centre=self.endObject.centre,
            global_rotation=self.endObject.global_rotation,
            global_parameters=self.global_parameters,
            **kwargs,
        )

    def writeElements(self) -> str:
        """
        Write the lattice elements defined in this object into an ELEGANT-compatible format; see
        :attr:`~SimulationFramework.Framework_objects.frameworkLattice.elementObjects`.

        Returns
        -------
        str
            The lattice represented as a string compatible with ELEGANT
        """
        self.final_screen = None
        if self.endObject not in self.screens_and_markers_and_bpms:
            self.final_screen = self.endScreen(
                output_filename=self.endObject.objectname + ".sdds"
            )
        elements = self.createDrifts()
        fulltext = ""
        fulltext += self.q.write_Elegant()
        for element in list(elements.values()):
            # print(element.write_Elegant())
            if not element.subelement:
                fulltext += element.write_Elegant()
        fulltext += (
            self.final_screen.write_Elegant() if self.final_screen is not None else ""
        )
        fulltext += self.objectname + ": Line=(START, "
        for e, element in list(elements.items()):
            if not element.subelement:
                if len((fulltext + e).splitlines()[-1]) > 60:
                    fulltext += "&\n"
                fulltext += e + ", "
        fulltext = (
            fulltext[:-2] + ", END )\n"
            if self.final_screen is not None
            else fulltext[:-2] + ")\n"
        )
        return fulltext

    def processRunSettings(self) -> tuple:
        """
        Process the runSettings object to extract the number of runs and the random number seed,
        and extract error definitions or a parameter scan definiton pertaining to this lattice section.

        Returns
        -------
        tuple
            nruns: Number of runs
            seed: Random number seedoutput
            elementErrors: Dict of errors on elements
            elementScan: Dict of elements and parameters to scan
        """
        nruns = self.runSettings.nruns
        seed = self.runSettings.seed
        elementErrors = (
            None
            if (self.runSettings.elementErrors is None)
            else self.processElementErrors(self.runSettings.elementErrors)
        )
        elementScan = (
            None
            if (self.runSettings.elementScan is None)
            else self.processElementScan(self.runSettings.elementScan, nruns)
        )
        return nruns, seed, elementErrors, elementScan

    def processElementErrors(self, elementErrors: Dict) -> Dict:
        """
        Process the elementErrors dictionary to prepare it for use with the current lattice section in ELEGANT

        Parameters
        ----------
        elementErrors: Dict
            Dictionary of element names and error definitions

        Returns
        -------
        Dict
            Formatted dictionary of errors on elements
        """

        output = {}
        default_err = {
            "amplitude": 1e-6,
            "fractional": 0,
            "type": '"gaussian"',
        }

        for ele, error_defn in elementErrors.items():
            # identify element names with wildcard characters
            wildcard = "*" in ele

            # raise errors for non-wildcarded element names that don't exist in the global lattice
            if (ele not in self.allElements) and (not wildcard):
                raise KeyError(
                    "Lattice element %s does not exist in the current lattice"
                    % str(ele)
                )

            # check if the lattice element (or a wildcard match) exist in the local lattice section
            # fetch the element type (or the types of matching elements, if using a wildcard name)
            element_exists = False
            if (ele in self.elements) and not wildcard:
                element_exists = True
                element_types = [self.elementObjects[ele].objecttype]
            elif wildcard:
                element_matches = [
                    x for x in self.elements if (ele.replace("*", "") in x)
                ]
                if len(element_matches) != 0:
                    element_exists = True
                    element_types = [
                        self.elementObjects[x].objecttype for x in element_matches
                    ]

            # if the element exists in the local lattice, do processing
            if element_exists:
                output[ele] = {}

                # check that all matching elements have the same type
                ele_type = str(element_types[0])
                has_expected_type = [(x == ele_type) for x in element_types]
                if not all(has_expected_type):
                    raise TypeError(
                        "All lattice elements matching a wilcarded element name must have the same type"
                    )

                # check error definition associated with each of the element parameters
                # for example, an element corresponding to an RF cavity might have parameters 'amplitude' and 'phase'
                for param in error_defn:
                    # check that the current element type has this parameter
                    if param not in elementkeywords[ele_type]["keywords"]:
                        raise KeyError(
                            "Element type %s has no associated keyword %s"
                            % (str(ele_type), str(param))
                        )

                    # check for keyword conversions between simframe and elegant
                    # for example, in simframe the elegant parameter 'voltage' for RF cavities is called 'amplitude'
                    conversions = keyword_conversion_rules_elegant[ele_type]
                    keyword = conversions[param] if (param in conversions) else param
                    output[ele][keyword] = copy(default_err)

                    # fill in the define error parameters
                    for k in default_err:
                        if k in error_defn[param]:
                            output[ele][keyword][k] = error_defn[param][k]

                    # bind errors across wildcarded elements
                    if wildcard:
                        output[ele][keyword]["bind"] = 1
                        output[ele][keyword]["bind_across_names"] = 1
        return output

    def processElementScan(self, elementScan: Dict, nsteps: int) -> Dict | None:
        """
        Process the elementScan dictionary to prepare it for use with the current lattice section in ELEGANT

        #TODO deprecated?

        Parameters
        ----------
        elementScan: Dict[name, item]
            Dictionary of elements and parameters to scan

        Returns
        -------
        Dict or None
            Dictionary of processed elements to scan if valid, else None
        """
        # extract the name of the beamline element, and the parameter to scan
        ele, param = elementScan["name"], elementScan["item"]

        # raise errors for element names that don't exist anywhere in the global lattice
        if ele not in self.allElements:
            raise KeyError(
                "Lattice element %s does not exist in the current lattice" % str(ele)
            )

        # check if the lattice element exists in the local lattice section and fetch the element type
        element_exists = ele in self.elements
        if element_exists:
            ele_type = self.elementObjects[ele].objecttype

            # check that the element type has the parameter corresponding to the scan variable
            if param not in elementkeywords[ele_type]["keywords"]:
                raise KeyError(
                    "Element type %s has no associated parameter %s"
                    % (str(ele_type), str(param))
                )

            # check for keyword conversions between simframe and elegant
            conversions = keyword_conversion_rules_elegant[ele_type]
            keyword = conversions[param] if (param in conversions) else param

            # build the scan value array
            scan_values = np.linspace(
                elementScan["min"], elementScan["max"], int(nsteps) - 1
            )

            # the first scan step is always the baseline simulation, for fiducialization
            multiplicative = elementScan["multiplicative"]
            if multiplicative:
                if 1.0 not in list(scan_values):
                    scan_values = [1.0] + list(scan_values)
            else:
                if 0.0 not in list(scan_values):
                    scan_values = [0.0] + list(scan_values)

            # build the SDDS file with the scan values
            scan_fname = "%s-%s.sdds" % (ele, param)
            scanSDDS = sddsFile()
            scanSDDS.add_parameter("name", [ele], type=sdds.SDDS(0).SDDS_STRING)
            scanSDDS.add_parameter("item", [keyword], type=sdds.SDDS(0).SDDS_STRING)
            scanSDDS.add_parameter("multiplicative", [int(multiplicative)])
            scanSDDS.add_parameter("nominal", [self.elements[ele][param]])
            scanSDDS.add_column("values", scan_values)
            scanSDDS.save(self.global_parameters["master_subdir"] + "/" + scan_fname)

            output = {
                "name": ele,
                "item": keyword,
                "differential": int(not multiplicative),
                "multiplicative": int(multiplicative),
                "enumeration_file": scan_fname,
                "enumeration_column": "values",
            }
            return output

        else:
            return None

    def write(self) -> None:
        """
        Write the ELEGANT lattice and command files to `master_subdir` using the functions
        :func:`~SimulationFramework.Codes.Elegant.Elegant.writeElements` and
        based on the output of :func:`~SimulationFramework.Codes.Elegant.Elegant.createCommandFiles`.
        """
        lattice_file = (
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".lte"
        )
        saveFile(lattice_file, self.writeElements())
        # try:
        command_file = (
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".ele"
        )
        saveFile(command_file, "", "w")
        if len(self.commandFilesOrder) > 0:
            for cfileid in self.commandFilesOrder:
                if cfileid in self.commandFiles:
                    cfile = self.commandFiles[cfileid]
                    saveFile(command_file, cfile.write_Elegant(), "a")
        else:
            warn("commandFilesOrder length is zero; run createCommandFiles first")
        # except Exception:
        #     pass

    def createCommandFiles(self) -> None:
        """
        Create the :class:`~SimulationFramework.Codes.Elegant.elegantCommandFile` objects
        based on the run settings, lattice and beam parameters, including scans of elements,
        if defined.

        Updates :attr:`~SimulationFramework.Codes.Elegant.Elegant.commandFiles` and
        :attr:`~SimulationFramework.Codes.Elegant.Elegant.commandFilesOrder`
        """
        if not isinstance(self.commandFiles, dict) or self.commandFiles == {}:
            # print('createCommandFiles is creating new command files!')
            # print('processRunSettings')
            nruns, seed, elementErrors, elementScan = self.processRunSettings()
            # print('global_settings')
            self.commandFiles["global_settings"] = elegant_global_settings_command(
                lattice=self, warning_limit=0
            )
            # print('run_setup')
            self.commandFiles["run_setup"] = elegant_run_setup_command(
                lattice=self,
                p_central=np.mean(self.global_parameters["beam"].BetaGamma),
                seed=seed,
                losses="%s.loss",
            )

            # print('generate commands for monte carlo jitter runs')
            if elementErrors is not None:
                self.commandFiles["run_control"] = elegant_run_control_command(
                    lattice=self,
                    n_steps=nruns,
                    n_passes=1,
                    reset_rf_for_each_step=0,
                    first_is_fiducial=1,
                )
                self.commandFiles["error_elements"] = elegant_error_elements_command(
                    lattice=self, elementErrors=elementErrors, nruns=nruns
                )
                for e in elementErrors:
                    for item in elementErrors[e]:
                        self.commandFiles["error_element_" + e + "_" + item] = (
                            elegantCommandFile(
                                objecttype="error_element",
                                name=e,
                                item=item,
                                allow_missing_elements=1,
                                **elementErrors[e][item],
                            )
                        )
            elif elementScan is not None:
                # print('generate commands for parameter scans without fiducialisation (i.e. jitter scans)')
                self.commandFiles["run_control"] = elegant_run_control_command(
                    lattice=self,
                    n_steps=nruns - 1,
                    n_passes=1,
                    n_indices=1,
                    reset_rf_for_each_step=0,
                    first_is_fiducial=1,
                )
                self.commandFiles["scan_elements"] = elegant_scan_elements_command(
                    lattice=self, elementScan=elementScan, nruns=nruns
                )
            else:
                # print('run_control for standard runs with no jitter')
                self.commandFiles["run_control"] = elegant_run_control_command(
                    lattice=self, n_steps=1, n_passes=1
                )

            # print('twiss_output')
            self.commandFiles["twiss_output"] = elegant_twiss_output_command(
                lattice=self,
                beam=self.global_parameters["beam"],
                betax=self.betax,
                betay=self.betay,
                alphax=self.alphax,
                alphay=self.alphay,
            )
            # print('floor_coordinates')
            self.commandFiles["floor_coordinates"] = elegant_floor_coordinates_command(
                lattice=self
            )
            # print('matrix_output')
            self.commandFiles["matrix_output"] = elegant_matrix_output_command(
                lattice=self
            )
            # print('sdds_beam')
            self.commandFiles["sdds_beam"] = elegant_sdds_beam_command(
                lattice=self,
                elegantbeamfilename=self.objectname + ".sdds",
                sample_interval=self.sample_interval,
                reuse_bunch=1,
                fiducialization_bunch=0,
                center_arrival_time=0,
            )
            # print('track')
            self.commandFiles["track"] = elegant_track_command(
                lattice=self, trackBeam=self.trackBeam
            )
            self.commandFilesOrder = list(
                self.commandFiles.keys()
            )  # ['global_settings', 'run_setup', 'error_elements', 'scan_elements', 'run_control', 'twiss', 'sdds_beam', 'track']

    def preProcess(self) -> None:
        """
        Prepare the input distribution for ELEGANT based on the `prefix` in the settings
        file for this lattice section, and create the ELEGANT command files.
        """
        super().preProcess()
        prefix = (
            self.file_block["input"]["prefix"]
            if "input" in self.file_block and "prefix" in self.file_block["input"]
            else ""
        )
        HDF5filename = prefix + self.particle_definition + ".hdf5"
        if os.path.isfile(expand_substitution(self, HDF5filename)):
            filepath = expand_substitution(self, HDF5filename)
        else:
            filepath = self.global_parameters["master_subdir"] + "/" + HDF5filename
        rbf.hdf5.read_HDF5_beam_file(
            self.global_parameters["beam"],
            os.path.abspath(filepath),
        )
        self.global_parameters["beam"].beam.rematchXPlane(
            **self.initial_twiss["horizontal"]
        )
        self.global_parameters["beam"].beam.rematchYPlane(
            **self.initial_twiss["vertical"]
        )
        if self.trackBeam:
            self.hdf5_to_sdds(prefix)
        self.createCommandFiles()

    @lox.thread
    def screen_threaded_function(self, scr: screen, sddsindex: int) -> None:
        """
        Convert output from ELEGANT screen to HDF5 format

        Parameters
        ----------
        scr: :class:`~SimulationFramework.Elements.screen.screen`
            Screen object
        sddsindex: int
            SDDS object index
        """
        try:
            return scr.sdds_to_hdf5(sddsindex)
        except Exception:
            return None

    def postProcess(self) -> None:
        """
        PostProcess the simulation results, i.e. gather the screens and markers
        and write their outputs to HDF5.

        :attr:`~SimulationFramework.Codes.Elegant.Elegant.commandFiles` is also cleared
        """
        super().postProcess()
        if self.trackBeam:
            for i, s in enumerate(self.screens_and_markers_and_bpms):
                self.screen_threaded_function.scatter(s, i)
            if (
                self.final_screen is not None
                and not self.final_screen.output_filename.lower()
                in [
                    s.output_filename.lower() for s in self.screens_and_markers_and_bpms
                ]
            ):
                self.screen_threaded_function.scatter(
                    self.final_screen, len(self.screens_and_markers_and_bpms)
                )
        self.screen_threaded_function.gather()
        self.commandFiles = {}

    def hdf5_to_sdds(self, write: bool=True) -> None:
        """
        Convert the HDF5 beam input file to an SDDS file, and create a
        :class:`~SimulationFramework.Elements.charge.charge` object as the first element
        """
        if self.bunch_charge is not None:
            self.q = charge(
                objectname="START",
                objecttype="charge",
                global_parameters=self.global_parameters,
                **{"total": abs(self.bunch_charge)},
            )
        else:
            self.q = charge(
                objectname="START",
                objecttype="charge",
                global_parameters=self.global_parameters,
                **{"total": abs(self.global_parameters["beam"].Q)},
            )
        sddsbeamfilename = self.objectname + ".sdds"
        if write:
            rbf.sdds.write_SDDS_file(
                self.global_parameters["beam"],
                self.global_parameters["master_subdir"] + "/" + sddsbeamfilename,
                xyzoffset=self.startObject.position_start,
            )

    def run(self):
        """
        Run the code with input 'filename'
        """
        if not os.name == "nt":
            command = self.executables[self.code] + [self.objectname + ".ele"]
            if self.global_parameters["simcodes_location"] is None:
                my_env = {**os.environ}
            else:
                my_env = {
                    **os.environ,
                    "RPN_DEFNS": os.path.abspath(
                        self.global_parameters["simcodes_location"]
                    )
                    + "/Elegant/defns_linux.rpn",
                }
            with open(
                os.path.abspath(
                    self.global_parameters["master_subdir"]
                    + "/"
                    + self.objectname
                    + ".log"
                ),
                "w",
            ) as f:
                subprocess.call(
                    command,
                    stdout=f,
                    cwd=self.global_parameters["master_subdir"],
                    env=my_env,
                )
        else:
            code_string = " ".join(self.executables[self.code]).lower()
            command = self.executables[self.code] + [self.objectname + ".ele"]
            if "pelegant" in code_string:
                command = (
                    [command[0]]
                    + [
                        "-env",
                        "RPN_DEFNS",
                        (
                            os.path.abspath(self.global_parameters["simcodes_location"])
                            + "/Elegant/defns.rpn"
                        ).replace("/", "\\"),
                    ]
                    + command[1:]
                )
                command = [c.replace("/", "\\") for c in command]
                with open(
                    os.path.abspath(
                        self.global_parameters["master_subdir"]
                        + "/"
                        + self.objectname
                        + ".log"
                    ),
                    "w",
                ) as f:
                    subprocess.call(
                        command, stdout=f, cwd=self.global_parameters["master_subdir"]
                    )
            else:
                command = [c.replace("/", "\\") for c in command]
                with open(
                    os.path.abspath(
                        self.global_parameters["master_subdir"]
                        + "/"
                        + self.objectname
                        + ".log"
                    ),
                    "w",
                ) as f:
                    subprocess.call(
                        command,
                        stdout=f,
                        cwd=self.global_parameters["master_subdir"],
                        env={
                            "RPN_DEFNS": (
                                os.path.abspath(
                                    self.global_parameters["simcodes_location"]
                                )
                                + "/Elegant/defns.rpn"
                            ).replace("/", "\\")
                        },
                    )

    def elegantCommandFile(self, *args, **kwargs):
        return elegantCommandFile(*args, **kwargs)


class elegantCommandFile(frameworkCommand):
    """
    Generic class for generating elements for an ELEGANT input file
    """
    lattice: frameworkLattice | str = None
    """The :class:`~SimulationFramework.Framework_objects.frameworkLattice` object"""

    def __init__(self, *args, **kwargs):
        super(elegantCommandFile, self).__init__(*args, **kwargs)


class elegant_global_settings_command(elegantCommandFile):
    """
    Global settings for an ELEGANT input file; see `Elegant global settings`_

    .. _Elegant global settings: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu37.html#x45-440007.28
    """

    inhibit_fsync: int = 0
    """See this parameter in `Elegant global settings`_ for more details.    """

    mpi_io_force_file_sync: int = 0
    """See this parameter in `Elegant global settings`_ for more details."""

    mpi_io_read_buffer_size: int = 16777216
    """See this parameter in `Elegant global settings`_ for more details."""

    mpi_io_write_buffer_size: int = 16777216
    """See this parameter in `Elegant global settings`_ for more details."""

    usleep_mpi_io_kludge: int = 0
    """See this parameter in `Elegant global settings`_ for more details."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(elegant_global_settings_command, self).__init__(
            objectname="global_settings",
            objecttype="global_settings",
            *args,
            **kwargs,
        )
        kwargs.update(
            {
                "inhibit_fsync": self.inhibit_fsync,
                "mpi_io_force_file_sync": self.mpi_io_force_file_sync,
                "mpi_io_read_buffer_size": self.mpi_io_read_buffer_size,
                "mpi_io_write_buffer_size": self.mpi_io_write_buffer_size,
                "usleep_mpi_io_kludge": self.usleep_mpi_io_kludge,
            }
        )
        self.add_properties(**kwargs)


class elegant_run_setup_command(elegantCommandFile):
    """
    Run setup for an ELEGANT input file; see `Elegant run setup`_

    .. _Elegant run setup: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu69.html#x77-760007.60
    """

    pcentral: float = 0.0
    """Central momentum in units of beta-gamma"""

    seed: int = 0
    """Seed for random number generators"""

    always_change_p0: int = 1
    """Match the reference momentum to the beam momentum after each element."""

    default_order: int = 3
    """The default order of transfer matrices used for elements having matrices."""

    lattice: frameworkLattice | str = None
    """:class:`~SimulationFramework.Framework_objects.frameworkLattice object"""

    centroid: str = "%s.cen"
    """File to which centroid data is to be written"""

    sigma: str = "%s.sig"
    """File to which sigma data is to be written"""

    def __init__(self, *args, **kwargs):
        super(elegant_run_setup_command, self).__init__(
            objectname="run_setup", objecttype="run_setup", *args, **kwargs
        )
        self.lattice_filename = self.lattice.objectname + ".lte"
        for k in self.model_fields_set:
            if k not in kwargs:
                kwargs.update({k: getattr(self, k)})
        kwargs.pop("lattice")
        self.add_properties(
            lattice=self.lattice_filename,
            use_beamline=self.lattice.objectname,
            s_start=self.lattice.startObject.position_start[2],
            **kwargs,
        )


class elegant_error_elements_command(elegantCommandFile):
    """
    Error control for an ELEGANT input file; see `Elegant error control`_

    .. _Elegant error control: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu33.html#x41-400007.24
    """

    elementErrors: Dict = None
    """Dictionary of elements with errors"""

    nruns: int = 1
    """Number of error runs to perform"""

    lattice: frameworkLattice = None
    """:class:`~SimulationFramework.Framework_objects.frameworkLattice object"""

    no_errors_for_first_step: int = 1
    """Perform the first run without errors"""

    error_log: str = "%s.erl"
    """File to which errors are to be logged"""

    # build commands for randomised errors on specified elements
    def __init__(self, *args, **kwargs):
        super(elegant_error_elements_command, self).__init__(
            objectname="error_control", objecttype="error_control", **kwargs
        )
        self.add_properties(
            **{
                "objecttype": "error_control",
                "no_errors_for_first_step": self.no_errors_for_first_step,
                "error_log": self.error_log,
            }
        )


class elegant_scan_elements_command(elegantCommandFile):
    """
    Error control for an ELEGANT input file; see `Elegant vary element`_

    .. _Elegant vary element: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu85.html#x93-920007.76
    """

    elementScan: Dict = None
    "Element names and parameters to scan"

    nruns: int = 1
    """Number of runs to perform"""

    index_number: int = 0
    """Scan number index"""

    lattice: frameworkLattice = None
    """:class:`~SimulationFramework.Framework_objects.frameworkLattice object"""

    # build command for a systematic parameter scan
    def __init__(self, *args, **kwargs):
        super(elegant_scan_elements_command, self).__init__(
            objectname="vary_element", objecttype="vary_element", *args, **kwargs
        )
        self.elementScan.update(
            {
                "objecttype": "vary_element",
                "index_number": self.index_number,
            }
        )
        self.add_properties(**self.elementScan)


class elegant_run_control_command(elegantCommandFile):
    """
    Run control for an ELEGANT input file; see `Elegant run control`_

    .. _Elegant run control: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu68.html#x76-750007.59
    """
    def __init__(self, *args, **kwargs):
        super(elegant_run_control_command, self).__init__(
            objectname="run_control", objecttype="run_control", *args, **kwargs
        )
        self.add_properties(**kwargs)


class elegant_twiss_output_command(elegantCommandFile):
    """
    Twiss output for an ELEGANT input file; see `Elegant twiss output`_

    .. _Elegant twiss output: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu82.html#x90-890007.73
    """

    beam: rbf.beam
    """Particle distribution"""

    betax: float | None = None
    """Initial beta_x; if `None`, take it from `beam`"""

    betay: float | None = None
    """Initial beta_y; if `None`, take it from `beam`"""

    alphax: float | None = None
    """Initial alpha_x; if `None`, take it from `beam`"""

    alphay: float | None = None
    """Initial alpha_y; if `None`, take it from `beam`"""

    etax: float | None = None
    """Initial eta_x; if `None`, take it from `beam`"""

    etaxp: float | None = None
    """Initial eta_xp; if `None`, take it from `beam`"""

    # build command for a systematic parameter scan
    def __init__(self, *args, **kwargs):
        super(elegant_twiss_output_command, self).__init__(
            objectname="twiss_output",
            objecttype="twiss_output",
            *args,
            **kwargs,
        )
        self.betax = (
            self.betax if self.betax is not None else self.beam.twiss.beta_x_corrected
        )
        self.betay = (
            self.betay if self.betay is not None else self.beam.twiss.beta_y_corrected
        )
        self.alphax = (
            self.alphax
            if self.alphax is not None
            else self.beam.twiss.alpha_x_corrected
        )
        self.alphay = (
            self.alphay
            if self.alphay is not None
            else self.beam.twiss.alpha_y_corrected
        )
        self.etax = self.etax if self.etax is not None else self.beam.twiss.eta_x
        self.etaxp = self.etaxp if self.etaxp is not None else self.beam.twiss.eta_xp

        kwargs.update(
            {
                "matched": 0,
                "output_at_each_step": 0,
                "radiation_integrals": 1,
                "statistics": 1,
                "filename": "%s.twi",
                "beta_x": self.betax,
                "alpha_x": self.alphax,
                "beta_y": self.betay,
                "alpha_y": self.alphay,
                "eta_x": self.etax,
                "etap_x": self.etaxp,
            }
        )
        self.add_properties(**kwargs)


class elegant_floor_coordinates_command(elegantCommandFile):
    """
    Floor coordinates for an ELEGANT input file; see `Elegant floor coordinates`_

    .. _Elegant floor coordinates: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu35.html#x43-420007.26
    """
    lattice: frameworkLattice = None
    """:class:`~SimulationFramework.Framework_objects.frameworkLattice object"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(elegant_floor_coordinates_command, self).__init__(
            objectname="floor_coordinates",
            objecttype="floor_coordinates",
            *args,
            **kwargs,
        )
        kwargs.update(
            {
                "filename": "%s.flr",
                "X0": self.lattice.startObject.position_start[0],
                "Z0": self.lattice.startObject.position_start[2],
                "theta0": 0,
                "magnet_centers": 0,
            }
        )
        self.add_properties(**kwargs)


class elegant_matrix_output_command(elegantCommandFile):
    """
    Matrix output for an ELEGANT input file; see `Elegant matrix output`_

    .. _Elegant matrix output: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu49.html#x57-560007.40
    """

    full_matrix_only: int = 0
    """A flag indicating that only the matrix of the entire accelerator is to be output."""

    SDDS_output_order: int = 2
    """Matrix output order for the SDDS file"""

    SDDS_output: str = "%s.mat"
    """File to which matrix data is to be written"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(elegant_matrix_output_command, self).__init__(
            objectname="matrix_output",
            objecttype="matrix_output",
            *args,
            **kwargs,
        )
        kwargs.update(
            {
                "full_matrix_only": self.full_matrix_only,
                "SDDS_output_order": self.SDDS_output_order,
                "SDDS_output": self.SDDS_output,
            }
        )
        self.add_properties(**kwargs)


class elegant_sdds_beam_command(elegantCommandFile):
    """
    SDDS beam input for an ELEGANT input file; see `Elegant sdds beam`_

    .. _Elegant sdds beam: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu72.html#x80-790007.63
    """

    elegantbeamfilename: str = ""
    """Input filename for ELEGANT"""

    def __init__(self, *args, **kwargs):
        super(elegant_sdds_beam_command, self).__init__(
            objectname="sdds_beam",
            objecttype="sdds_beam",
            *args,
            **kwargs,
        )
        kwargs.update({"input": self.elegantbeamfilename})
        self.add_properties(**kwargs)


class elegant_track_command(elegantCommandFile):
    """
    Track command for an ELEGANT input file; see `Elegant track`_

    .. _Elegant track: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu83.html#x91-900007.74
    """

    trackBeam: bool = True
    """Flag to indicate whether to include the track command"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(elegant_track_command, self).__init__(
            objectname="track",
            objecttype="track",
            *args,
            **kwargs,
        )
        if self.trackBeam:
            self.add_properties(**kwargs)


class elegantOptimisation(elegantCommandFile):
    """
    Class for generating input commands for ELEGANT optimisation.
    See `Elegant optimization variable`_ , `Elegant optimization constraint`_ ,
    and `Elegant optimization term`_

    .. _Elegant optimization variable: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu61.html#x69-680007.52
    .. _Elegant optimization constraint: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu55.html#x63-620007.46
    .. _Elegant optimization term: https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu60.html#x68-670007.51
    """

    variables: Dict = {}
    """Dictionary of names and variables to be changed"""

    constraints: Dict = {}
    """Dictionary of constraints for the optimization"""

    terms: Dict = {}
    """Dictionary of terms to be optimized"""

    settings: Dict = {}
    """Dictionary of optimization settings"""

    def __init__(self, *args, **kwargs):
        super(elegantOptimisation, self).__init__(
            *args,
            **kwargs,
        )
        for k, v in list(self.variables.items()):
            self.add_optimisation_variable(k, **v)

    def add_optimisation_variable(
            self,
            name: str,
            item: str=None,
            lower: float=None,
            upper: float=None,
            step: float=None,
            restrict_range: int=None,
    ):
        """
        Add an optimization variable and create the command

        Parameters
        ----------
        name: str
            Element name
        item: str
            Element parameter to be varied
        lower: float
            Lower limit allowed for `item`
        upper: float
            Upper limit allowed for `item`
        step: int
            Specifies grid size for optimization algorithm
        restrict_range: int
            If nonzero, the initial value is forced inside the allowed range
        """
        self.addCommand(
            name=name,
            type="optimization_variable",
            item=item,
            lower_limit=lower,
            upper_limit=upper,
            step_size=step,
            force_inside=restrict_range,
        )

    def add_optimisation_constraint(
            self,
            name: str,
            item: str=None,
            lower: float=None,
            upper: float=None
    ):
        """
        Add an optimization constraint and create the command

        Parameters
        ----------
        name: str
            Element name
        item: str
            Element parameter to be constrained
        lower: float
            Lower limit allowed for `item`
        upper: float
            Upper limit allowed for `item`
        """
        self.addCommand(
            name=name,
            type="optimization_constraint",
            quantity=item,
            lower=lower,
            upper=upper,
        )

    def add_optimisation_term(
            self,
            name: str,
            item: str=None,
            **kwargs,
    ):
        """
        Add an optimization term and create the command

        Parameters
        ----------
        name: str
            Element name
        item: str
            Element parameter to be constrained
        """
        self.addCommand(name=name, type="optimization_term", term=item, **kwargs)


class sddsFile(object):
    """simple class for writing generic column data to a new SDDS file"""

    def __init__(self):
        """initialise an SDDS instance, prepare for writing to file"""
        self.sdds = sdds.SDDS(0)

    def add_column(self, name, data, **kwargs):
        """add a column of floating point numbers to the file"""
        if not isinstance(name, str):
            raise TypeError("Column names must be string types")
        self.sdds.defineColumn(
            name,
            symbol=kwargs["symbol"] if ("symbol" in kwargs) else "",
            units=kwargs["units"] if ("units" in kwargs) else "",
            description=kwargs["description"] if ("description" in kwargs) else "",
            formatString="",
            type=self.sdds.SDDS_DOUBLE,
            fieldLength=0,
        )

        if isinstance(data, (tuple, list, np.ndarray)):
            self.sdds.setColumnValueList(name, list(data), page=1)
        else:
            raise TypeError("Column data must be a list, tuple or array-like type")

    def add_parameter(self, name, data, **kwargs):
        """add a parameter of floating point numbers to the file"""
        if not isinstance(name, str):
            raise TypeError("Parameter names must be string types")
        if "type" in kwargs:
            type = kwargs["type"]
        else:
            type = self.sdds.SDDS_DOUBLE
        self.sdds.defineParameter(
            name,
            symbol=kwargs["symbol"] if ("symbol" in kwargs) else "",
            units=kwargs["units"] if ("units" in kwargs) else "",
            description=kwargs["description"] if ("description" in kwargs) else "",
            formatString="",
            type=type,
            fixedValue="",
        )

        if isinstance(data, (tuple, list, np.ndarray)):
            self.sdds.setParameterValueList(name, list(data))
        else:
            raise TypeError("Parameter data must be a list, tuple or array-like type")

    def save(self, fname):
        """save the sdds data structure to file"""
        if not isinstance(fname, str):
            raise TypeError("SDDS file name must be a string!")
        self.sdds.save(fname)
