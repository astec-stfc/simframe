"""
Simframe CSRTrack Module

Various objects and functions to handle CSRTrack lattices and commands. See `CSRTrack manual`_ for more details.

    .. _CSRTrack manual: https://www.desy.de/xfel-beam/csrtrack/files/CSRtrack_User_Guide_(actual).pdf

Classes:
    - :class:`~SimulationFramework.Codes.CSRTrack.CSRTrack.csrtrackLattice`: The ASTRA lattice object, used for\
    converting the :class:`~SimulationFramework.Framework_elements.frameworkObject` s defined in the\
    :class:`~SimulationFramework.Framework_elements.frameworkLattice` into a string representation of\
    the lattice suitable for a CSRTrack input file.

    - :class:`~SimulationFramework.Codes.CSRTrack.CSRTrack.csrtrack_element`: Class for defining the a\
    CSRTrack instance of a :class:`~SimulationFramework.Framework_objects.frameworkElement`.

    - :class:`~SimulationFramework.Codes.CSRTrack.CSRTrack.csrtrack_forces`: Class for defining the CSR\
    calculation type.

    - :class:`~SimulationFramework.Codes.CSRTrack.CSRTrack.csrtrack_track_step`: Class for defining the\
    tracking step.

    - :class:`~SimulationFramework.Codes.CSRTrack.CSRTrack.csrtrack_particles`: Class for defining the\
     particle distribution and format.

    - :class:`~SimulationFramework.Codes.CSRTrack.CSRTrack.csrtrack_monitor`: Class for defining monitors.
"""

import os
import yaml
from ...Framework_objects import (
    frameworkLattice,
    frameworkElement,
    frameworkCounter,
    elementkeywords,
)
from ...Framework_elements import screen
from ...FrameworkHelperFunctions import saveFile, expand_substitution
from ...Modules import Beams as rbf
from typing import Dict, List

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/csrtrack_defaults.yaml", "r"
) as infile:
    csrtrack_defaults = yaml.safe_load(infile)


class csrtrackLattice(frameworkLattice):
    """
    Class for defining the CSRTrack lattice object, used for
    converting the :class:`~SimulationFramework.Framework_elements.frameworkObject`s defined in the
    :class:`~SimulationFramework.Framework_elements.frameworkLattice` into a string representation of
    the lattice suitable for a CSRTrack input file.
    """

    code: str = "csrtrack"
    """String indicating the lattice object type"""

    particle_definition: str = ""
    """String representing the initial particle distribution"""

    CSRTrackelementObjects: Dict = {}
    """Dictionary representing all CSRTrack object namelists"""

    def __init__(self, *args, **kwargs):
        super(csrtrackLattice, self).__init__(*args, **kwargs)
        self.set_particles_filename()

    def endScreen(self, **kwargs) -> screen:
        """
        Create a final screen object for dumping the particle output after tracking.

        Returns
        -------
        :class:`~SimulationFramework.Elements.screen.screen`
        """
        return screen(
            name="end",
            type="screen",
            position_start=self.endObject.position_start,
            position_end=self.endObject.position_start,
            global_rotation=self.endObject.global_rotation,
            global_parameters=self.global_parameters,
            **kwargs,
        )

    def set_particles_filename(self) -> None:
        """
        Set up the `CSRTrackelementObjects namelist for the initial particle distribution,
        based on the `particle_definition` and the `global_parameters` of the lattice.
        """
        self.CSRTrackelementObjects["particles"] = csrtrack_particles(
            particle_definition=self.particle_definition,
            global_parameters=self.global_parameters,
        )
        self.CSRTrackelementObjects["particles"].format = "astra"
        if self.particle_definition == "initial_distribution":
            self.CSRTrackelementObjects["particles"].particle_definition = "laser.astra"
            self.CSRTrackelementObjects["particles"].add_default(
                "array", "#file{name=laser.astra}"
            )
        else:
            self.CSRTrackelementObjects["particles"].particle_definition = (
                self.elementObjects[self.start].objectname
            )
            self.CSRTrackelementObjects["particles"].add_default(
                "array",
                "#file{name="
                + self.elementObjects[self.start].objectname
                + ".astra"
                + "}",
            )

    @property
    def dipoles_screens_and_bpms(self) -> List:
        """
        Get a list of the dipoles, screens and BPMs sorted by their position in the lattice

        Returns
        -------
        List
            A sorted list of :class:`~SimulationFramework.Framework_objects.frameworkElement`
        """
        return sorted(
            self.getElementType("dipole")
            + self.getElementType("screen")
            + self.getElementType("beam_position_monitor"),
            key=lambda x: x.position_end[2],
        )

    def setCSRMode(self) -> None:
        """
        Set up the `forces` key in `CSRTrackelementObjects based on the `csr_mode` defined in the settings
        file for this lattice section. `csr_mode` can be either ["csr_g_to_p" (2D) or "projected" (1D)]
        """
        if "csr" in self.file_block and "csr_mode" in self.file_block["csr"]:
            if self.file_block["csr"]["csr_mode"] == "3D":
                self.CSRTrackelementObjects["forces"] = csrtrack_forces(
                    type="csr_g_to_p"
                )
            elif self.file_block["csr"]["csr_mode"] == "1D":
                self.CSRTrackelementObjects["forces"] = csrtrack_forces(
                    type="projected"
                )
        else:
            self.CSRTrackelementObjects["forces"] = csrtrack_forces()

    def writeElements(self) -> str:
        """
        Write the lattice elements defined in this object into a CSRTrack-compatible format; see
        :attr:`~SimulationFramework.Framework_objects.frameworkLattice.elementObjects`.

        The appropriate headers required for ASTRA are written at the top of the file, see the `_write_CSRTrack`
        function in :class:`~SimulationFramework.Codes.CSRTrack.csrtrack_element`.

        Returns
        -------
        str
            The lattice represented as a string compatible with CSRTrack
        """
        fulltext = "io_path{logfile = log.txt}\nlattice{\n"
        counter = frameworkCounter(sub={"beam_position_monitor": "screen"})
        for e in self.dipoles_screens_and_bpms:
            # if not e.type == 'dipole':
            # self.CSRTrackelementObjects[e.name] = csrtrack_online_monitor(filename=e.name+'.fmt2', monitor_type='phase', marker='screen'+str(counter.counter(e.type)), particle='all')
            fulltext += e.write_CSRTrack(counter.counter(e.objecttype))
            counter.add(e.objecttype)
        fulltext += self.endScreen().write_CSRTrack(
            counter.counter(self.endScreen().objecttype)
        )
        fulltext += "}\n"
        self.set_particles_filename()
        self.setCSRMode()
        self.CSRTrackelementObjects["track_step"] = csrtrack_track_step()
        self.CSRTrackelementObjects["tracker"] = csrtrack_tracker(
            end_time_marker="screen"
            + str(counter.counter(self.endScreen().objecttype))
            + "a"
        )
        self.CSRTrackelementObjects["monitor"] = csrtrack_monitor(
            name=self.end + ".fmt2", global_parameters=self.global_parameters
        )
        for c in self.CSRTrackelementObjects:
            fulltext += self.CSRTrackelementObjects[c].write_CSRTrack()
        return fulltext

    def write(self) -> str:
        """
        Writes the CSRTrack input file from :func:`~SimulationFramework.Codes.CSRTrack.csrtrackLattice.writeElements`
        to <master_subdir>/csrtrk.in.
        """
        code_file = self.global_parameters["master_subdir"] + "/csrtrk.in"
        saveFile(code_file, self.writeElements())

    def preProcess(self) -> None:
        """
        Convert the beam file from the previous lattice section into CSRTrack format and set the number of
        particles based on the input distribution, see
        :func:`~SimulationFramework.Codes.CSRTrack.csrtrack_particles.hdf5_to_astra`.
        """
        super().preProcess()
        prefix = (
            self.file_block["input"]["prefix"]
            if "input" in self.file_block and "prefix" in self.file_block["input"]
            else ""
        )
        self.CSRTrackelementObjects["particles"].hdf5_to_astra(prefix)

    def postProcess(self) -> None:
        """
        Convert the beam file from the CSRTrack output into HDF5 format, see
        :func:`~SimulationFramework.Codes.CSRTrack.csrtrack_monitor.csrtrack_to_hdf5`.
        """
        super().postProcess()
        self.CSRTrackelementObjects["monitor"].csrtrack_to_hdf5()


class csrtrack_element(frameworkElement):
    """
    Base class for CSRTrack elements, including namelists for the lattice file.
    """

    def __init__(self, objectname=None, objecttype=None, **kwargs):
        super(csrtrack_element, self).__init__(objectname, objecttype, **kwargs)
        self.header = ""
        if objectname in csrtrack_defaults:
            for k, v in list(csrtrack_defaults[objectname].items()):
                self.add_default(k, v)

    def CSRTrack_str(self, s: bool) -> str:
        """
        Convert a boolean into a string for CSRTrack.

        Parameters
        ----------
        s: bool
            Boolean to convert

        Returns
        -------
        str
            'yes' for `True`, 'no' for `False`, or the original string if otherwise
        """
        if s is True:
            return "yes"
        elif s is False:
            return "no"
        else:
            return str(s)

    def _write_CSRTrack(self) -> str:
        """
        Create the string for the header object in CSRTrack format.

        Returns
        -------
        str
            CSRTrack-compatible string for this element.
        """
        output = str(self.header) + "{\n"
        for k in elementkeywords[self.objecttype]["keywords"]:
            k = k.lower()
            if getattr(self, k) is not None:
                output += k + "=" + self.CSRTrack_str(getattr(self, k)) + "\n"
            elif k in self.objectdefaults:
                output += k + "=" + self.CSRTrack_str(self.objectdefaults[k]) + "\n"
        output += "}\n"
        return output


# class csrtrack_online_monitor(csrtrack_element):
#
#     def __init__(self, marker="", **kwargs):
#         super(csrtrack_online_monitor, self).__init__(
#             "online_monitor", "csrtrack_online_monitor", **kwargs
#         )
#         self.header = "online_monitor"
#         self.end_time_marker = marker + "b"


class csrtrack_forces(csrtrack_element):
    """
    Class for CSRTrack forces.
    """

    def __init__(self, **kwargs):
        super(csrtrack_forces, self).__init__("forces", "csrtrack_forces", **kwargs)
        self.header = "forces"


class csrtrack_track_step(csrtrack_element):
    """
    Class for defining CSRTrack the tracking step.
    """

    def __init__(self, **kwargs):
        super(csrtrack_track_step, self).__init__(
            "track_step", "csrtrack_track_step", **kwargs
        )
        self.header = "track_step"


class csrtrack_tracker(csrtrack_element):
    """
    Class for defining the CSRTrack tracker.
    """

    def __init__(self, end_time_marker="", **kwargs):
        super(csrtrack_tracker, self).__init__("tracker", "csrtrack_tracker", **kwargs)
        self.header = "tracker"
        self.end_time_marker = end_time_marker


class csrtrack_monitor(csrtrack_element):
    """
    Class for defining CSRTrack monitors.
    """

    def __init__(self, **kwargs):
        super(csrtrack_monitor, self).__init__(
            objectname="monitor", objecttype="csrtrack_monitor", **kwargs
        )
        self.header = "monitor"

    def csrtrack_to_hdf5(self) -> None:
        """
        Convert the particle distribution from a CSRTrack monitor into HDF5 format,
        and write it to `master_subdir`.
        """
        csrtrackbeamfilename = self.name
        astrabeamfilename = csrtrackbeamfilename.replace(".fmt2", ".astra")
        rbf.astra.convert_csrtrackfile_to_astrafile(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + csrtrackbeamfilename,
            self.global_parameters["master_subdir"] + "/" + astrabeamfilename,
        )
        rbf.astra.read_astra_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + astrabeamfilename,
            normaliseZ=False,
        )
        HDF5filename = self.name.replace(".fmt2", ".hdf5")
        rbf.hdf5.write_HDF5_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + HDF5filename,
            sourcefilename=csrtrackbeamfilename,
        )


class csrtrack_particles(csrtrack_element):
    """
    Class for defining CSRTrack particles.
    """

    def __init__(self, **kwargs):
        super(csrtrack_particles, self).__init__(
            "particles", "csrtrack_particles", **kwargs
        )
        self.header = "particles"

    def hdf5_to_astra(self, prefix: str = "") -> None:
        """
        Convert HDF5 particle distribution to ASTRA format, suitable for inputting to CSRTrack.

        Parameters
        ----------
        prefix: str
            Prefix for filename
        """
        HDF5filename = prefix + self.particle_definition.replace(".astra", "") + ".hdf5"
        if os.path.isfile(expand_substitution(self, HDF5filename)):
            filepath = expand_substitution(self, HDF5filename)
        else:
            filepath = self.global_parameters["master_subdir"] + "/" + HDF5filename
        rbf.hdf5.read_HDF5_beam_file(
            self.global_parameters["beam"],
            filepath,
        )
        astrabeamfilename = self.particle_definition + ".astra"
        rbf.astra.write_astra_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + astrabeamfilename,
            normaliseZ=False,
        )
