import os
import numpy as np
from SimulationFramework.Framework_objects import frameworkElement, elements_Elegant
from SimulationFramework.Modules import Beams as rbf
from warnings import warn
from SimulationFramework.Modules.gdf_beam import gdf_beam
from pydantic import field_validator


class screen(frameworkElement):
    """
    Class defining a screen element
    """

    beam: rbf.beam | None = None
    """:class:`~SimulationFramework.Modules.Beams.beam object"""

    output_filename: str | None = None
    """Output filename for the screen"""

    def model_post_init(self, __context):
        self.beam = rbf.beam()
        if self.output_filename is None:
            self.output_filename = self.objectname + ".sdds"
        super().model_post_init(__context)


    def _write_ASTRA(self, n, **kwargs) -> str:
        """
        Writes the screen element string for ASTRA.

        Note that in astra `Scr_xrot` means a rotation about the y-axis and vice versa.

        Parameters
        ----------
        n: int
            Screen index

        Returns
        -------
        str
            String representation of the element for ASTRA
        """
        return self._write_ASTRA_dictionary(
            dict(
                [
                    ["Screen", {"value": self.middle[2], "default": 0}],
                    ["Scr_xrot", {"value": self.y_rot + self.dy_rot, "default": 0}],
                    ["Scr_yrot", {"value": self.x_rot + self.dx_rot, "default": 0}],
                ]
            ),
            n,
        )

    def _write_Elegant(self) -> str:
        """
        Writes the screen element string for ELEGANT.

        Returns
        -------
        str
            String representation of the element for ELEGANT
        """
        wholestring = ""
        etype = self._convertType_Elegant(self.objecttype)
        string = self.objectname + ": " + etype
        # if self.length > 0:
        #     d = drift(self.objectname+'-drift-01', type='drift', **{'length': self.length/2})
        #     wholestring+=d._write_Elegant()
        for key, value in self.objectproperties.items():
            if (
                not key == "name"
                and not key == "type"
                and not key == "commandtype"
                and self._convertKeyword_Elegant(key) in elements_Elegant[etype]
            ):

                value = (
                    getattr(self, key)
                    if hasattr(self, key) and getattr(self, key) is not None
                    else value
                )
                key = self._convertKeyword_Elegant(key)
                tmpstring = ", " + key + " = " + str(value)
                if len(string + tmpstring) > 76:
                    wholestring += string + ",&\n"
                    string = ""
                    string += tmpstring[2::]
                else:
                    string += tmpstring
        wholestring += string + ";\n"
        # if self.length > 0:
        #     d = drift(self.objectname+'-drift-02', type='drift', **{'length': self.length/2})
        #     wholestring+=d._write_Elegant()
        return wholestring

    def _write_CSRTrack(self, n) -> str:
        """
        Writes the screen element string for CSRTrack.

        Parameters
        ----------
        n: int
            Modulator index

        Returns
        -------
        str
            String representation of the element for CSRTrack
        """
        z = self.middle[2]
        return (
            """quadrupole{\nposition{rho="""
            + str(z)
            + """, psi=0.0, marker=screen"""
            + str(n)
            + """a}\nproperties{strength=0.0, alpha=0, horizontal_offset=0,vertical_offset=0}\nposition{rho="""
            + str(z + 1e-6)
            + """, psi=0.0, marker=screen"""
            + str(n)
            + """b}\n}\n"""
        )

    def _write_GPT(self, Brho, ccs="wcs", output_ccs=None, *args, **kwargs):
        relpos, _ = ccs.relative_position(self.middle, self.global_rotation)
        ccs_label, value_text = ccs.ccs_text(self.middle, self.rotation)
        self.gpt_screen_position = relpos[2]
        output = "screen( " + ccs.name + ', "I", ' + str(relpos[2]) + ',"OutputCCS",'
        if output_ccs is not None:
            output += '"' + str(output_ccs) + '"'
        else:
            output += ccs.name
        output += ', "GroupName", "' + self.objectname + '"'
        output += ");\n"
        return output

    def find_ASTRA_filename(
        self, lattice: str, master_run_no: int, mult: int
    ) -> str | None:
        """
        Determine the ASTRA filename for the screen object.

        Parameters
        ----------
        lattice: str
            The name of the lattice
        master_run_no: int
            The run number
        mult: int
            Multiplication factor for ASTRA-type output

        Returns
        -------
        str or None
            The ASTRA filename for the screen object, or None if the file does not exist.
        """
        for i in [0, -0.001, 0.001]:
            tempfilename = (
                    lattice
                    + "."
                    + str(int(round((self.middle[2] + i - self.zstart[2]) * mult))).zfill(4)
                    + "."
                    + str(master_run_no).zfill(3)
            )
            tempfilenamenozstart = (
                    lattice
                    + "."
                    + str(int(round((self.middle[2] + i) * mult))).zfill(4)
                    + "."
                    + str(master_run_no).zfill(3)
            )
            # print(self.middle[2]+i-self.zstart[2], tempfilename, os.path.isfile(self.global_parameters['master_subdir'] + '/' + tempfilename))
            if os.path.isfile(
                    self.global_parameters["master_subdir"] + "/" + tempfilename
            ):
                return tempfilename
            elif os.path.isfile(
                self.global_parameters["master_subdir"] + "/" + tempfilenamenozstart
            ):
                return tempfilenamenozstart
        return None

    def astra_to_hdf5(
        self, lattice: str, cathode: bool = False, mult: int = 100
    ) -> None:
        """
        Convert the ASTRA beam file name to HDF5 format and write the beam file.

        Parameters
        ----------
        lattice: str
            Lattice name
        cathode: bool
            True if beam was emitted from a cathode
        mult: int
            Multiplication factor for ASTRA-type filenames
        """
        master_run_no = (
            self.global_parameters["run_no"]
            if "run_no" in self.global_parameters
            else 1
        )
        astrabeamfilename = self.find_ASTRA_filename(lattice, master_run_no, mult)
        if astrabeamfilename is None:
            warn(f"Screen Error: {lattice}, {self.middle[2]}, {self.zstart[2]}")
        else:
            rbf.astra.read_astra_beam_file(
                self.beam,
                (
                    self.global_parameters["master_subdir"] + "/" + astrabeamfilename
                ).strip('"'),
                normaliseZ=False,
            )
            rbf.hdf5.rotate_beamXZ(
                self.beam,
                -1 * self.starting_rotation[2],
                preOffset=[0, 0, 0],
                postOffset=-1 * np.array(self.starting_offset),
            )
            HDF5filename = (self.objectname + ".hdf5").strip('"')
            toffset = self.beam.toffset
            rbf.hdf5.write_HDF5_beam_file(
                self.beam,
                self.global_parameters["master_subdir"] + "/" + HDF5filename,
                centered=False,
                sourcefilename=astrabeamfilename,
                pos=self.middle,
                cathode=cathode,
                toffset=toffset,
            )
            if self.global_parameters["delete_tracking_files"]:
                os.remove(
                    (
                        self.global_parameters["master_subdir"]
                        + "/"
                        + astrabeamfilename
                    ).strip('"')
                )

    def sdds_to_hdf5(self, sddsindex: int = 1, toffset: float = 0.0) -> None:
        """
        Convert the SDDS beam file name to HDF5 format and write the beam file.

        Parameters
        ----------
        sddsindex: int
            Index for SDDS file
        """
        self.beam.sddsindex = sddsindex
        elegantbeamfilename = self.output_filename.replace(".sdds", ".SDDS").strip('"')
        rbf.sdds.read_SDDS_beam_file(
            self.beam,
            self.global_parameters["master_subdir"] + "/" + elegantbeamfilename,
        )
        HDF5filename = (
            self.output_filename.replace(".sdds", ".hdf5")
            .replace(".SDDS", ".hdf5")
            .strip('"')
        )
        rbf.hdf5.write_HDF5_beam_file(
            self.beam,
            self.global_parameters["master_subdir"] + "/" + HDF5filename,
            centered=False,
            sourcefilename=elegantbeamfilename,
            pos=self.middle,
            zoffset=self.end,
            toffset=toffset,
        )
        if self.global_parameters["delete_tracking_files"]:
            os.remove(
                (
                    self.global_parameters["master_subdir"] + "/" + elegantbeamfilename
                ).strip('"')
            )

    def gdf_to_hdf5(
        self, gptbeamfilename: str, cathode: bool = False, gdf: gdf_beam | None = None
    ) -> None:
        """
        Convert the GDF beam file to HDF5 format and write the beam file.

        Parameters
        ----------
        gptbeamfilename: str
            Name of GPT beam file
        cathode: bool
            True if beam was emitted from a cathode
        gdf: gdfbeam or None
            GDF beam object
        """
        # gptbeamfilename = self.objectname + '.' + str(int(round((self.allElementObjects[self.end].position_end[2])*100))).zfill(4) + '.' + str(master_run_no).zfill(3)
        # try:
        # print('Converting screen', self.objectname,'at', self.gpt_screen_position)
        rbf.gdf.read_gdf_beam_file(
            self.beam,
            self.global_parameters["master_subdir"] + "/" + gptbeamfilename,
            position=self.objectname,
            gdfbeam=gdf,
        )
        HDF5filename = self.objectname + ".hdf5"
        rbf.hdf5.write_HDF5_beam_file(
            self.beam,
            self.global_parameters["master_subdir"] + "/" + HDF5filename,
            centered=False,
            sourcefilename=gptbeamfilename,
            pos=self.middle,
            xoffset=self.end[0],
            cathode=cathode,
            toffset=(-1 * np.mean(self.beam.t)),
        )
        # except:
        #     print('Error with screen', self.objectname,'at', self.gpt_screen_position)
        if self.global_parameters["delete_tracking_files"]:
            os.remove(
                (self.global_parameters["master_subdir"] + "/" + gptbeamfilename).strip(
                    '"'
                )
            )
