import os
import numpy as np
from SimulationFramework.Framework_objects import frameworkElement, elements_Elegant
from SimulationFramework.Modules import Beams as rbf
from SimulationFramework.Modules.merge_two_dicts import merge_two_dicts


class screen(frameworkElement):

    def __init__(self, name=None, type="screen", **kwargs):
        super().__init__(name, type, **kwargs)
        self.beam = rbf.beam()
        if "output_filename" not in kwargs:
            self.output_filename = str(self.objectname) + ".sdds"

    def _write_ASTRA(self, n, **kwargs):
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

    def _write_Elegant(self):
        wholestring = ""
        etype = self._convertType_Elegant(self.objecttype)
        string = self.objectname + ": " + etype
        # if self.length > 0:
        #     d = drift(self.objectname+'-drift-01', type='drift', **{'length': self.length/2})
        #     wholestring+=d._write_Elegant()
        for key, value in list(
            merge_two_dicts(self.objectproperties, self.objectdefaults).items()
        ):
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

    def _write_CSRTrack(self, n):
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
        output = "screen( " + ccs.name + ', "I", ' + str(relpos[2]) + ",\"OutputCCS\","
        if output_ccs is not None:
            output += '"' + str(output_ccs) + '"'
        else:
            output += ccs.name
        output += ", \"GroupName\", \"" + self.objectname + "\""
        output += ");\n"
        return output

    def find_ASTRA_filename(self, lattice, master_run_no, mult):
        for i in [0, -0.001, 0.001]:
            tempfilename = (
                lattice
                + "."
                + str(int(round((self.middle[2] + i - self.zstart[2]) * mult))).zfill(4)
                + "."
                + str(master_run_no).zfill(3)
            )
            # print(self.middle[2]+i-self.zstart[2], tempfilename, os.path.isfile(self.global_parameters['master_subdir'] + '/' + tempfilename))
            if os.path.isfile(
                self.global_parameters["master_subdir"] + "/" + tempfilename
            ):
                return tempfilename
        return None

    def astra_to_hdf5(self, lattice, cathode=False, mult=100):
        master_run_no = (
            self.global_parameters["run_no"]
            if "run_no" in self.global_parameters
            else 1
        )
        astrabeamfilename = self.find_ASTRA_filename(lattice, master_run_no, mult)
        if astrabeamfilename is None:
            print(("Screen Error: ", lattice, self.middle[2], self.zstart[2]))
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
                -1 * self.starting_rotation,
                preOffset=[0, 0, 0],
                postOffset=-1 * np.array(self.starting_offset),
            )
            HDF5filename = (self.objectname + ".hdf5").strip('"')
            toffset = self.beam["toffset"]
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

    def sdds_to_hdf5(self, sddsindex=1):
        self.beam.sddsindex = sddsindex
        elegantbeamfilename = self.output_filename.replace(".sdds", ".SDDS").strip('"')
        # print('sdds_to_hdf5')
        rbf.sdds.read_SDDS_beam_file(
            self.beam,
            self.global_parameters["master_subdir"] + "/" + elegantbeamfilename,
        )
        # print('sdds_to_hdf5', 'read_SDDS_beam_file')
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
            toffset=(-1 * np.mean(self.global_parameters["beam"].t)),
        )
        # print('sdds_to_hdf5', 'write_HDF5_beam_file')
        if self.global_parameters["delete_tracking_files"]:
            os.remove(
                (
                    self.global_parameters["master_subdir"] + "/" + elegantbeamfilename
                ).strip('"')
            )

    def gdf_to_hdf5(self, gptbeamfilename, cathode=False, gdfbeam=None):
        # gptbeamfilename = self.objectname + '.' + str(int(round((self.allElementObjects[self.end].position_end[2])*100))).zfill(4) + '.' + str(master_run_no).zfill(3)
        # try:
        # print('Converting screen', self.objectname,'at', self.gpt_screen_position)
        rbf.gdf.read_gdf_beam_file(
            self.beam,
            self.global_parameters["master_subdir"] + "/" + gptbeamfilename,
            position=self.objectname,
            gdfbeam=gdfbeam,
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
