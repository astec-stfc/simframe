"""
Simframe GPT Module

Various objects and functions to handle GPT lattices and commands.

Classes:
    - :class:`~SimulationFramework.Codes.GPT.GPT.gptLattice`: The GPT lattice object, used for\
    converting the :class:`~SimulationFramework.Framework_elements.frameworkObject` s defined in the\
    :class:`~SimulationFramework.Framework_elements.frameworkLattice` into a string representation of\
    the lattice suitable for GPT input and lattice files.

    - :class:`~SimulationFramework.Codes.GPT.GPT.gpt_element`: Base class for defining\
    commands in a GPT input file.

    - :class:`~SimulationFramework.Codes.GPT.GPT.gpt_setfile`: Class for defining the\
    input files for the GPT input file.

    - :class:`~SimulationFramework.Codes.GPT.GPT.gpt_charge`: Class for defining the\
    bunch charge for the GPT input file.

    - :class:`~SimulationFramework.Codes.GPT.GPT.gpt_setreduce`: Class for reducing the\
    number of particles for the GPT input file.

    - :class:`~SimulationFramework.Codes.GPT.GPT.gpt_accuracy`: Class for setting the\
    accuracy for GPT tracking.

    - :class:`~SimulationFramework.Codes.GPT.GPT.gpt_spacecharge`: Class for defining the\
    space charge setup for the GPT input file.

    - :class:`~SimulationFramework.Codes.GPT.GPT.gpt_tout`: Class for defining the\
    number of steps for particle distribution output for the GPT input file.

    - :class:`~SimulationFramework.Codes.GPT.GPT.gpt_csr1d`: Class for defining the\
    CSR calculations for the GPT input file.

    - :class:`~SimulationFramework.Codes.GPT.GPT.gpt_writefloorplan`: Class for setting up the\
    writing of the lattice floor plan for the GPT input file.

    - :class:`~SimulationFramework.Codes.GPT.GPT.gpt_Zminmax`: Class for defining the\
    minimum and maximum z-positions for the GPT input file.

    - :class:`~SimulationFramework.Codes.GPT.GPT.gpt_forwardscatter`: Class for defining\
    scattering parameters for the GPT input file.

    - :class:`~SimulationFramework.Codes.GPT.GPT.gpt_scatterplate`: Class for defining a\
    scattering object for the GPT input file.

    - :class:`~SimulationFramework.Codes.GPT.GPT.gpt_dtmaxt`: Class for defining the\
    step size(s) for the GPT input file.
"""

import os
from copy import deepcopy
import subprocess
import numpy as np
from ...Framework_objects import (
    frameworkLattice,
    frameworkElement,
    elementkeywords,
    getGrids,
)
from ...Framework_elements import screen, marker, gpt_ccs, cavity, wakefield
from ...FrameworkHelperFunctions import saveFile, expand_substitution
from ...Modules import Beams as rbf
from ...Modules.merge_two_dicts import merge_two_dicts
from ...Modules.Fields import field
from ...Modules.units import UnitValue
from typing import Dict, Literal

gpt_defaults = {}


class gptLattice(frameworkLattice):
    """
    Class for defining the GPT lattice object, used for
    converting the :class:`~SimulationFramework.Framework_elements.frameworkObject`s defined in the
    :class:`~SimulationFramework.Framework_elements.frameworkLattice` into a string representation of
    the lattice suitable for a GPT input file.
    """

    code: str = "gpt"
    """String indicating the lattice object type"""

    allow_negative_drifts: bool = True
    """Flag to indicate whether negative drifts are allowed"""

    bunch_charge: float | None = None
    """Bunch charge"""

    headers: Dict = {}
    """Headers to be included in the GPT lattice file"""

    ignore_start_screen: screen | None = None
    """Flag to indicate whether to ignore the first screen in the lattice"""

    screen_step_size: float = 0.1
    """Step size for screen output"""

    time_step_size: str = "0.1/c"
    """Step size for tracking"""

    override_meanBz: float | int | None = None
    """Set the average particle longitudinal velocity manually"""

    override_tout: float | int | None = None
    """Set the time step output manually"""

    accuracy: int = 6
    """Tracking accuracy"""

    endScreenObject: screen | None = None
    """Final screen object for dumping particle distributions"""

    Brho: UnitValue | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        self.headers["setfile"] = gpt_setfile(
            set='"beam"', filename='"' + self.name + '.gdf"'
        )
        self.headers["floorplan"] = gpt_writefloorplan(
            filename='"' + self.objectname + '_floor.gdf"'
        )

    @property
    def space_charge_mode(self) -> str | None:
        """
        Get the space charge mode based on
        :attr:`~SimulationFramework.Framework_objects.frameworkLattice.globalSettings` or
        :attr:`~SimulationFramework.Framework_objects.frameworkLattice.file_block`.

        Returns
        -------
        str
            Space charge mode as string, or None if not provided.

        """
        if (
            "charge" in self.file_block
            and "space_charge_mode" in self.file_block["charge"]
        ):
            return self.file_block["charge"]["space_charge_mode"]
        elif (
            "charge" in self.globalSettings
            and "space_charge_mode" in self.globalSettings["charge"]
        ):
            return self.globalSettings["charge"]["space_charge_mode"]
        else:
            return None

    @space_charge_mode.setter
    def space_charge_mode(self, mode: Literal["2d", "3d", "2D", "3D"]) -> None:
        """
        Set the space charge mode manually ["2D", "3D"].

        Parameters
        ----------
        mode: Literal["2d", "3d", "2D", "3D"]
            The space charge calculation mode
        """
        if "charge" not in self.file_block:
            self.file_block["charge"] = {}
        self.file_block["charge"]["space_charge_mode"] = mode

    def endScreen(self, **kwargs) -> screen:
        """
        Make the final position in the lattice a
        :class:`~SimulationFramework.Elements.screen.screen` object.

        Returns
        -------
        :class:`~SimulationFramework.Elements.screen.screen`
            The final screen in the lattice

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

    def writeElements(self) -> str:
        """
        Write the lattice elements defined in this object into a GPT-compatible format; see
        :attr:`~SimulationFramework.Framework_objects.frameworkLattice.elementObjects`.

        The appropriate headers required for GPT are written at the top of the file, see the `write_GPT`
        function in :class:`~SimulationFramework.Codes.GPT.gpt_element`.

        Returns
        -------
        str
            The lattice represented as a string compatible with GPT
        """
        ccs = gpt_ccs("wcs", [0, 0, 0], [0, 0, 0])
        fulltext = ""
        self.headers["accuracy"] = gpt_accuracy(self.accuracy)
        if "charge" not in self.file_block:
            self.file_block["charge"] = {}
        if "charge" not in self.globalSettings:
            self.globalSettings["charge"] = {}
        space_charge_dict = merge_two_dicts(
            self.file_block["charge"],
            self.globalSettings["charge"],
        )
        if self.particle_definition == "laser" and self.space_charge_mode is not None:
            self.headers["spacecharge"] = gpt_spacecharge(
                **merge_two_dicts(self.global_parameters, space_charge_dict)
            )
            self.headers["spacecharge"].npart = len(self.global_parameters["beam"].x)
            self.headers["spacecharge"].sample_interval = self.sample_interval
            # self.headers["spacecharge"].space_charge_mode = "cathode"
        else:
            self.headers["spacecharge"] = gpt_spacecharge(
                **merge_two_dicts(self.global_parameters, space_charge_dict)
            )
        if (
            self.csr_enable
            and len(self.dipoles) > 0
            and max([abs(d.angle) for d in self.dipoles]) > 0
        ):  # and not os.name == 'nt':
            self.headers["csr1d"] = gpt_csr1d()
            # print('CSR Enabled!', self.objectname, len(self.dipoles))
        # self.headers['forwardscatter'] = gpt_forwardscatter(ECS='"wcs", "I"', name='cathode', probability=0)
        # self.headers['scatterplate'] = gpt_scatterplate(ECS='"wcs", "z", -1e-6', model='cathode', a=1, b=1)
        for header in self.headers:
            fulltext += self.headers[header].write_GPT()
        for i, element in enumerate(list(self.elements.values())):
            if i == 0:
                screen0pos = element.start[2]
                ccs = element.gpt_ccs(ccs)
            if i == 0 and isinstance(element, screen):
                self.ignore_start_screen = element
            else:
                fulltext += element.write_GPT(self.Brho, ccs=ccs)
                if (
                    isinstance(element, cavity)
                    and hasattr(element, "wakefield_definition")
                    and isinstance(element.wakefield_definition, field)
                ):
                    original_properties = deepcopy(element.objectproperties)
                    original_properties.objectname = f"{element.objectname}_wake"
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
                    fulltext += wake_element.write_GPT(self.Brho, ccs=ccs)
                new_ccs = element.gpt_ccs(ccs)
                if not new_ccs == ccs:
                    # print('ccs = ', ccs, '  new_ccs = ', new_ccs)
                    relpos, relrot = ccs.relative_position(
                        element.middle, element.global_rotation
                    )
                    if self.particle_definition == "laser":
                        fulltext += (
                            "screen( "
                            + ccs.name
                            + ', "I", '
                            + str(screen0pos + self.screen_step_size)
                            + ", "
                            + str(relpos[2])
                            + ", "
                            + str(float(self.screen_step_size))
                            + ', "OutputCCS",'
                            + ccs.name
                            + ");\n"
                        )
                    else:
                        fulltext += (
                            "screen( "
                            + ccs.name
                            + ', "I", '
                            + str(screen0pos)
                            + ", "
                            + str(relpos[2])
                            + ", "
                            + str(float(self.screen_step_size))
                            + ', "OutputCCS",'
                            + ccs.name
                            + ");\n"
                        )
                    screen0pos = 0
                    ccs = new_ccs
        if not isinstance(element, (screen, marker)):
            element = self.endScreenObject = self.endScreen()
            fulltext += self.endScreenObject.write_GPT(
                self.Brho, ccs=ccs, output_ccs="wcs"
            )
        else:
            # print('End screen', element.objectname)
            self.endScreenObject = None
        relpos, relrot = ccs.relative_position(
            element.position_end, element.global_rotation
        )
        if self.particle_definition == "laser":
            fulltext += (
                "screen( "
                + ccs.name
                + ', "I", '
                + str(screen0pos + self.screen_step_size)
                + ", "
                + str(relpos[2])
                + ", "
                + str(float(self.screen_step_size))
                + ', "OutputCCS",'
                + ccs.name
                + ");\n"
            )
        else:
            fulltext += (
                "screen( "
                + ccs.name
                + ', "I", '
                + str(screen0pos)
                + ", "
                + str(relpos[2])
                + ", "
                + str(float(self.screen_step_size))
                + ', "OutputCCS",'
                + ccs.name
                # + ", \"GroupName\","
                # + "\"SCREEN-" + ccs.name.strip("\"").upper() + "-END-01\""
                + ");\n"
            )
            zminmax = gpt_Zminmax(
                ECS='"wcs", "I"',
                zmin=self.startObject.position_start[2] - 0.1,
                zmax=self.endObject.position_end[2] + 1,
            )
            fulltext += zminmax.write_GPT()
        return fulltext

    def write(self) -> str:
        """
        Writes the GPT input file from :func:`~SimulationFramework.Codes.GPT.gptLattice.writeElements`
        to <master_subdir>/<self.objectname>.in.
        """
        code_file = (
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".in"
        )
        saveFile(code_file, self.writeElements())
        return self.writeElements()

    def preProcess(self) -> None:
        """
        Convert the beam file from the previous lattice section into GPT format and set the number of
        particles based on the input distribution, see
        :func:`~SimulationFramework.Codes.GPT.GPT.gptLattice.hdf5_to_astra`.
        """
        super().preProcess()
        self.headers["setfile"].particle_definition = self.objectname + ".gdf"
        prefix = self.get_prefix()
        self.hdf5_to_gdf(prefix)

    def run(self) -> None:
        """
        Run the code with input 'filename'

        `GPTLICENSE` must be provided in
        :attr:`~SimulationFramework.Framework_objects.frameworkLattice.global_parameters`.

        Average properties of the distribution are also calculated and written
        to an `<>emit.gdf` file in `master_subdir`.
        """
        main_command = (
            self.executables[self.code]
            + ["-o", self.objectname + "_out.gdf"]
            + ["GPTLICENSE=" + self.global_parameters["GPTLICENSE"]]
            + [self.objectname + ".in"]
        )
        my_env = os.environ.copy()
        my_env["LD_LIBRARY_PATH"] = (
            my_env["LD_LIBRARY_PATH"] + ":/opt/GPT3.3.6/lib/"
            if "LD_LIBRARY_PATH" in my_env
            else "/opt/GPT3.3.6/lib/"
        )
        my_env["OMP_WAIT_POLICY"] = "PASSIVE"
        post_command = (
            [self.executables[self.code][0].replace("gpt", "gdfa")]
            + ["-o", self.objectname + "_emit.gdf"]
            + [self.objectname + "_out.gdf"]
            + [
                "position",
                "Q",
                "avgx",
                "avgy",
                "avgz",
                "stdx",
                "stdBx",
                "stdy",
                "stdBy",
                "stdz",
                "stdt",
                "nemixrms",
                "nemiyrms",
                "nemizrms",
                "numpar",
                "nemirrms",
                "avgG",
                "avgp",
                "stdG",
                "avgt",
                "avgBx",
                "avgBy",
                "avgBz",
                "CSalphax",
                "CSalphay",
                "CSbetax",
                "CSbetay",
            ]
        )
        post_command_t = (
            [self.executables[self.code][0].replace("gpt", "gdfa")]
            + ["-o", self.objectname + "_emitt.gdf"]
            + [self.objectname + "_out.gdf"]
            + [
                "time",
                "Q",
                "avgx",
                "avgy",
                "avgz",
                "stdx",
                "stdBx",
                "stdy",
                "stdBy",
                "stdz",
                "nemixrms",
                "nemiyrms",
                "nemizrms",
                "numpar",
                "nemirrms",
                "avgG",
                "avgp",
                "stdG",
                "avgBx",
                "avgBy",
                "avgBz",
                "CSalphax",
                "CSalphay",
                "CSbetax",
                "CSbetay",
                "avgfBx",
                "avgfEx",
                "avgfBy",
                "avgfEy",
                "avgfBz",
                "avgfEz",
            ]
        )
        post_command_traj = (
            [self.executables[self.code][0].replace("gpt", "gdfa")]
            + ["-o", self.objectname + "traj.gdf"]
            + [self.objectname + "_out.gdf"]
            + ["time", "Q", "avgx", "avgy", "avgz"]
        )
        with open(
            os.path.abspath(
                self.global_parameters["master_subdir"] + "/" + self.objectname + ".bat"
            ),
            "w",
        ) as batfile:
            for command in [
                main_command,
                post_command,
                post_command_t,
                post_command_traj,
            ]:
                output = '"' + command[0] + '" '
                for c in command[1:]:
                    output += c + " "
                output += "\n"
                batfile.write(output)
        with open(
            os.path.abspath(
                self.global_parameters["master_subdir"] + "/" + self.objectname + ".log"
            ),
            "w",
        ) as f:
            # print('gpt command = ', command)
            subprocess.call(
                main_command,
                stdout=f,
                cwd=self.global_parameters["master_subdir"],
                env=my_env,
            )
            subprocess.call(
                post_command, stdout=f, cwd=self.global_parameters["master_subdir"]
            )
            subprocess.call(
                post_command_t, stdout=f, cwd=self.global_parameters["master_subdir"]
            )
            subprocess.call(
                post_command_traj, stdout=f, cwd=self.global_parameters["master_subdir"]
            )

    def postProcess(self) -> None:
        """
        Convert the beam file(s) from the GPT output into HDF5 format, see
        :func:`~SimulationFramework.Elements.screen.screen.gdf_to_hdf5`.
        """
        super().postProcess()
        cathode = self.particle_definition == "laser"
        gdfbeam = rbf.gdf.read_gdf_beam_file_object(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"]
            + "/"
            + self.objectname
            + "_out.gdf",
        )
        for e in self.screens_and_markers_and_bpms:
            if not e == self.ignore_start_screen:
                e.gdf_to_hdf5(
                    self.objectname + "_out.gdf", cathode=cathode, gdfbeam=gdfbeam
                )
            # else:
            # print('Ignoring', self.ignore_start_screen.objectname)
        if self.endScreenObject is not None:
            self.endScreenObject.gdf_to_hdf5(
                self.objectname + "_out.gdf", cathode=cathode
            )

    def hdf5_to_gdf(self, prefix: str="") -> None:
        """
        Convert the HDF5 beam distribution to GDF format.

        Certain properties of this class, including
        :attr:`~SimulationFramework.Codes.GPT.GPT.gptLattice.sample_interval`,
        :attr:`~SimulationFramework.Codes.GPT.GPT.gptLattice.override_meanBz`,
        :attr:`~SimulationFramework.Codes.GPT.GPT.gptLattice.override_tout` are also
        used to update
        :attr:`~SimulationFramework.Codes.GPT.GPT.gptLattice.headers`.

        Parameters
        ----------
        prefix: str
            HDF5 file prefix
        """
        HDF5filename = prefix + self.particle_definition + ".hdf5"
        if os.path.isfile(expand_substitution(self, HDF5filename)):
            filepath = expand_substitution(self, HDF5filename)
        else:
            filepath = self.global_parameters["master_subdir"] + "/" + HDF5filename
        rbf.hdf5.read_HDF5_beam_file(
            self.global_parameters["beam"],
            filepath,
        )
        if self.sample_interval > 1:
            self.headers["setreduce"] = gpt_setreduce(
                set='"beam"',
                setreduce=int(
                    len(self.global_parameters["beam"].x) / self.sample_interval
                ),
            )
        if self.override_meanBz is not None and isinstance(
            self.override_meanBz, (int, float)
        ):
            meanBz = self.override_meanBz
        else:
            meanBz = np.mean(self.global_parameters["beam"].Bz)
            if meanBz < 0.5:
                meanBz = 0.75

        if self.override_tout is not None and isinstance(
            self.override_tout, (int, float)
        ):
            self.headers["tout"] = gpt_tout(
                starttime=0, endpos=self.override_tout, step=str(self.time_step_size)
            )
        else:
            self.headers["tout"] = gpt_tout(
                starttime=0,
                endpos=(self.findS(self.end)[0][1] - self.findS(self.start)[0][1])
                / meanBz
                / 2.998e8,
                step=str(self.time_step_size),
            )
        self.global_parameters["beam"].beam.rematchXPlane(
            **self.initial_twiss["horizontal"]
        )
        self.global_parameters["beam"].beam.rematchYPlane(
            **self.initial_twiss["vertical"]
        )
        gdfbeamfilename = self.objectname + ".gdf"
        cathode = self.particle_definition == "laser"
        rbf.gdf.write_gdf_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + gdfbeamfilename,
            normaliseX=self.elementObjects[self.start].start[0],
            cathode=cathode,
        )
        self.Brho = self.global_parameters["beam"].Brho


class gpt_element(frameworkElement):
    """
    Generic class for generating headers for GPT.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(gpt_element, self).__init__(
            *args,
            **kwargs,
        )

    def write_GPT(self, *args, **kwargs) -> str:
        """
        Write the text for the GPT namelist based on its
        :attr:`~objectdefaults`, :attr:`~objectname`.

        Returns
        -------
        str
            GPT-compatible string representing the namelist
        """
        return self._write_GPT(*args, **kwargs)

    def _write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + "("
        for k in elementkeywords[self.objecttype]["keywords"]:
            k = k.lower()
            if getattr(self, k) is not None:
                output += str(getattr(self, k)) + ", "
            elif k in self.objectdefaults:
                output += self.objectdefaults[k] + ", "
        output = output[:-2]
        output += ");\n"
        return output


class gpt_setfile(gpt_element):
    """
    Class for setting filenames in GPT via `setfile`.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(gpt_setfile, self).__init__(
            objectname="setfile",
            objecttype="gpt_setfile",
            *args,
            **kwargs,
        )


class gpt_charge(gpt_element):
    """
    Class for generating the `settotalcharge` namelist for GPT.
    """

    set: str = '"beam"'
    """Name of beam for `settotalcharge`"""

    charge: float = 0.0
    """Bunch charge"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(gpt_charge, self).__init__(
            objectname="settotalcharge", objecttype="gpt_charge", *args, **kwargs
        )

    def _write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + "("
        output += str(self.set) + ","
        output += str(-1 * abs(self.charge)) + ");\n"
        return output


class gpt_setreduce(gpt_element):
    """
    Class for reducing the number of particles via `setreduce`.

    """

    set: str = '"beam"'
    """Name of the beam for `setreduce`"""

    setreduce: int = 1
    """Factor by which to reduce the number of particles"""

    def __init__(self, **kwargs):
        super(gpt_setreduce, self).__init__(
            objectname="setreduce", objecttype="gpt_setreduce", **kwargs
        )

    def _write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + "("
        output += str(self.set) + ","
        output += str(self.setreduce) + ");\n"
        return output


class gpt_accuracy(gpt_element):
    """
    Class for setting the accuracy of tracking via `accuracy` in GPT.
    """

    def __init__(
        self,
        accuracy,
        *args,
        **kwargs,
    ):
        super(gpt_accuracy, self).__init__(
            accuracy=accuracy,
            objectname="accuracy",
            objecttype="gpt_accuracy",
            *args,
            **kwargs,
        )

    def _write_GPT(self, *args, **kwargs):
        output = (
            "accuracy(" + str(self.accuracy) + ");\n"
        )  # 'setrmacrodist(\"beam\","u",1e-9,0) ;\n'
        return output


class gpt_spacecharge(gpt_element):
    """
    Class for preparing space charge calculations in GPT via `spacecharge`.
    """

    grids: getGrids = None
    """Class for calculating the required number of space charge grids"""

    ngrids: int | None = None
    """Number of space charge grids"""

    space_charge_mode: str | None = None
    """Space charge mode ['2D', '3D']"""

    cathode: bool = False
    """Flag indicating whether the bunch was emitted from a cathode"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(gpt_spacecharge, self).__init__(
            objectname="spacecharge",
            objecttype="gpt_spacecharge",
            *args,
            **kwargs,
        )
        self.grids = getGrids()

    def _write_GPT(self, *args, **kwargs):
        output = ""
        if isinstance(self.space_charge_mode, str) and self.cathode:
            if self.ngrids is None:
                self.ngrids = self.grids.getGridSizes(
                    (self.npart / self.sample_interval)
                )
            output += 'spacecharge3Dmesh("Cathode","RestMaxGamma",1000);\n'
        elif (
            isinstance(self.space_charge_mode, str)
            and self.space_charge_mode.lower() == "3d"
        ):
            output += "Spacecharge3Dmesh();\n"
        elif (
            isinstance(self.space_charge_mode, str)
            and self.space_charge_mode.lower() == "2d"
        ):
            output += "Spacecharge3Dmesh();\n"
        else:
            output = ""
        return output


class gpt_tout(gpt_element):
    """
    Class for setting up the beam dump rate via `tout`.
    """

    startpos: float = 0.0
    """Starting position"""

    endpos: float = 0.0
    """End position"""

    starttime: float | None = None
    """Start time for dumping"""

    step: str = "0.1/c"
    """Dump step as a string [distance / c]"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(gpt_tout, self).__init__(
            objectname="tout",
            objecttype="gpt_tout",
            *args,
            **kwargs,
        )

    def _write_GPT(self, *args, **kwargs):
        self.starttime = 0 if self.starttime < 0 else self.starttime
        output = str(self.objectname) + "("
        if self.starttime is not None:
            output += str(self.starttime) + ","
        else:
            output += str(self.startpos) + "/c,"
        output += str(self.endpos) + ","
        output += str(self.step) + ");\n"
        return output


class gpt_csr1d(gpt_element):
    """
    Class for preparing CSR calculations via `csr1d`.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(gpt_csr1d, self).__init__(
            objectname="csr1d",
            objecttype="gpt_csr1d",
            *args,
            **kwargs,
        )

    def _write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + "();\n"
        return output


class gpt_writefloorplan(gpt_element):
    """
    Class for writing the lattice floor plan via `writefloorplan`.
    """

    filename: str = ""
    """Floor plan filename"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(gpt_writefloorplan, self).__init__(
            objectname="writefloorplan",
            objecttype="gpt_writefloorplan",
            *args,
            **kwargs,
        )

    def _write_GPT(self, *args, **kwargs):
        output = str(self.objectname) + "(" + self.filename + ");\n"
        return output


class gpt_Zminmax(gpt_element):
    """
    Class for setting the boundaries in z for discarding particles via `Zminmax`
    """

    zmin: float = 0.0
    """Minimum longitudinal position"""

    zmax: float = 0.0
    """Maximum longitudinal position"""

    ECS: str = '"wcs", "I"'
    """Element coordinate system as a string"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(gpt_Zminmax, self).__init__(
            objectname="Zminmax",
            objecttype="gpt_Zminmax",
            *args,
            **kwargs,
        )

    def _write_GPT(self, *args, **kwargs):
        output = (
            str(self.objectname)
            + "("
            + self.ECS
            + ", "
            + str(self.zmin)
            + ", "
            + str(self.zmax)
            + ");\n"
        )
        return output


class gpt_forwardscatter(gpt_element):
    """
    Class for scattering particles via `forwardscatter`.
    """

    zmin: float = 0.0
    """Minimum longitudinal position"""

    zmax: float = 0.0
    """Maximum longitudinal position"""

    ECS: str = '"wcs", "I"'
    """Element coordinate system"""

    probability: float = 0.0
    """Scattering probability"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(gpt_forwardscatter, self).__init__(
            objectname="forwardscatter",
            objecttype="gpt_forwardscatter",
            *args,
            **kwargs,
        )

    def _write_GPT(self, *args, **kwargs):
        output = (
            str(self.objectname)
            + "("
            + self.ECS
            + ', "'
            + str(self.name)
            + '", '
            + str(self.probability)
            + ");\n"
        )
        return output


class gpt_scatterplate(gpt_element):
    """
    Class for scattering particles off a plate via `scatterplate`.
    """

    zmin: float = 0.0
    """Minimum longitudinal position"""

    zmax: float = 0.0
    """Maximum longitudinal position"""

    ECS: str = '"wcs", "I"'
    """Element coordinate system"""

    a: float = 0.0
    """Plate length in x-direction"""

    b: float = 0.0
    """Plate length in y-direction"""

    model: str = "cathode"
    """Scattering model to be used ['cathode', 'remove']"""

    def __init__(self, *args, **kwargs):
        super(gpt_scatterplate, self).__init__(
            objectname="scatterplate",
            objecttype="gpt_scatterplate",
            *args,
            **kwargs,
        )

    def _write_GPT(self, *args, **kwargs):
        output = (
            str(self.objectname)
            + "("
            + self.ECS
            + ", "
            + str(self.a)
            + ", "
            + str(self.b)
            + ') scatter="'
            + str(self.model)
            + '";\n'
        )
        return output


class gpt_dtmaxt(gpt_element):
    """
    Class for setting up minimum, maximmum temporal step sizes for tracking via `dtmaxt`.
    """

    tend: float = 0.0
    """Final time value"""

    tstart: float = 0.0
    """Initial time value"""

    dtmax: float = 0.0
    """Maximum temporal step size"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(gpt_dtmaxt, self).__init__(
            objectname="dtmaxt",
            objecttype="gpt_dtmaxt",
            *args,
            **kwargs,
        )

    def _write_GPT(self, *args, **kwargs):
        output = (
            str(self.objectname)
            + "("
            + str(self.tstart)
            + ", "
            + str(self.tend)
            + ", "
            + str(self.dtmax)
            + ");\n"
        )
        return output
