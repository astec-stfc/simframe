import os
import numpy as np
import lox

# from .ASTRARules import ASTRARules
from ...Framework_objects import (
    frameworkLattice,
    frameworkCounter,
    frameworkElement,
    getGrids,
    elementkeywords,
)
from ...Framework_elements import global_error
from ...FrameworkHelperFunctions import expand_substitution, saveFile
from ...Modules.merge_two_dicts import merge_two_dicts
from ...Modules import Beams as rbf

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
    def __init__(self, *args, **kwargs):
        super(astraLattice, self).__init__(*args, **kwargs)
        self.code = "astra"
        self.allow_negative_drifts = True
        self._bunch_charge = None
        self._toffset = None
        self.headers = dict()
        self.starting_offset = (
            eval(expand_substitution(self, self.file_block["starting_offset"]))
            if "starting_offset" in self.file_block
            else [0, 0, 0]
        )

        # This calculated the starting rotation based on the input file and the number of dipoles
        self.starting_rotation = (
            -1 * self.allElementObjects[self.start].global_rotation[2]
            if self.allElementObjects[self.start].global_rotation is not None
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
            )
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
                self.allElementObjects[self.start].objectname + ".astra"
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
            )
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
            )
        )

        # Create an "error" block
        if "global_errors" not in self.file_block:
            self.file_block["global_errors"] = {}
        if "global_errors" not in self.globalSettings:
            self.globalSettings["global_errors"] = {}
        if "global_errors" in self.file_block or "global_errors" in self.globalSettings:
            self.global_error = global_error(
                name=self.objectname + "_global_error",
                global_parameters=self.global_parameters,
            )
            self.headers["global_errors"] = astra_errors(
                element=self.global_error,
                global_parameters=self.global_parameters,
                **merge_two_dicts(
                    self.file_block["global_errors"],
                    self.globalSettings["global_errors"],
                )
            )
        # print 'errors = ', self.file_block, self.headers['global_errors']

    @property
    def space_charge_mode(self):
        return self.headers["charge"].space_charge_mode

    @space_charge_mode.setter
    def space_charge_mode(self, mode):
        self.headers["charge"].space_charge_mode = mode

    @property
    def sample_interval(self):
        return self._sample_interval

    @sample_interval.setter
    def sample_interval(self, interval):
        # print('Setting new ASTRA sample_interval = ', interval)
        self._sample_interval = interval
        self.headers["newrun"].sample_interval = interval
        self.headers["charge"].sample_interval = interval

    @property
    def bunch_charge(self):
        return self._bunch_charge

    @bunch_charge.setter
    def bunch_charge(self, charge):
        # print('Setting new ASTRA sample_interval = ', interval)
        self._bunch_charge = charge
        self.headers["newrun"].bunch_charge = charge

    @property
    def toffset(self):
        return self._toffset

    @toffset.setter
    def toffset(self, toffset):
        # print('Setting new ASTRA sample_interval = ', interval)
        self._toffset = toffset
        self.headers["newrun"].toffset = 1e9 * toffset

    def writeElements(self):
        fulltext = ""
        # Create objects for the newrun, output and charge blocks
        self.headers["output"].start_element = self.allElementObjects[self.start]
        self.headers["output"].end_element = self.allElementObjects[self.end]
        self.headers["output"].screens = self.screens_and_bpms
        # write the headers and their elements
        for header in self.headers:
            fulltext += self.headers[header].write_ASTRA(0) + "\n"
        # Initialise a counter object
        counter = frameworkCounter(sub={"kicker": "dipole", "collimator": "aperture"})
        for t in [
            ["cavities"],
            ["wakefields"],
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
                        auto_phase=self.headers["newrun"]["auto_phase"],
                    )
                else:
                    elemstr = element.write_ASTRA(counter.counter(element.objecttype))
                if elemstr is not None and not elemstr == "":
                    fulltext += elemstr + "\n"
                    if element.objecttype == "kicker":
                        counter.add(element.objecttype)
                    elif element.objecttype == "longitudinal_wakefield":
                        counter.add(element.objecttype, element.cells)
                    elif (
                        element.objecttype == "aperture"
                        or element.objecttype == "collimator"
                    ):
                        counter.add(element.objecttype, element.number_of_elements)
                    else:
                        counter.add(element.objecttype)
            fulltext += "\n/\n"
        return fulltext

    def write(self):
        self.code_file = (
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".in"
        )
        saveFile(self.code_file, self.writeElements())

    def preProcess(self):
        prefix = (
            self.file_block["input"]["prefix"]
            if "input" in self.file_block and "prefix" in self.file_block["input"]
            else ""
        )
        self.headers["newrun"].hdf5_to_astra(prefix)
        self.headers["charge"].npart = len(self.global_parameters["beam"].x)

    @lox.thread
    def screen_threaded_function(self, screen, objectname, cathode, mult):
        return screen.astra_to_hdf5(objectname, cathode, mult)

    def find_ASTRA_filename(self, elem, mult, lattice, master_run_no):
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

    def get_screen_scaling(self):
        for e in self.screens_and_bpms:
            if not self.starting_offset == [0, 0, 0]:
                e.zstart = self.allElementObjects[self.start].start
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

    def postProcess(self):
        cathode = (
            self.headers["newrun"]["particle_definition"] == "initial_distribution"
        )
        mult = self.get_screen_scaling()
        for e in self.screens_and_bpms:
            if not self.starting_offset == [0, 0, 0]:
                e.zstart = self.allElementObjects[self.start].start
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

    def astra_to_hdf5(self, cathode=False):
        # print('ASTRA/astra_to_hdf5', cathode)
        master_run_no = (
            self.global_parameters["run_no"]
            if "run_no" in self.global_parameters
            else 1
        )
        if not self.starting_offset == [0, 0, 0]:
            zstart = self.allElementObjects[self.start].start
        else:
            zstart = [0, 0, 0]
        startpos = np.array(self.allElementObjects[self.start].start) - np.array(zstart)
        endpos = np.array(self.allElementObjects[self.end].end) - np.array(zstart)
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
        HDF5filename = self.allElementObjects[self.end].objectname + ".hdf5"
        toffset = self.global_parameters["beam"]["toffset"]
        rbf.hdf5.write_HDF5_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + HDF5filename,
            centered=False,
            sourcefilename=astrabeamfilename,
            pos=self.allElementObjects[self.end].middle,
            cathode=cathode,
            toffset=toffset,
        )
        # print('ASTRA/astra_to_hdf5', 'finished')


class astra_header(frameworkElement):

    def __init__(self, name=None, type=None, **kwargs):
        super(astra_header, self).__init__(name, type, **kwargs)

    def framework_dict(self):
        return dict()

    def write_ASTRA(self, n):
        keyword_dict = dict()
        for k in elementkeywords[self.objecttype]["keywords"]:
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
    def __init__(self, offset, rotation, **kwargs):
        super(astra_header, self).__init__("newrun", "astra_newrun", **kwargs)
        self.starting_offset = offset
        self.starting_rotation = rotation
        self.sample_interval = 1
        if "run" not in kwargs:
            self.run = 1
        if "head" not in kwargs:
            self.head = "trial"
        if "lprompt" not in kwargs:
            self.add_property("lprompt", False)

    def framework_dict(self):
        astradict = dict(
            [
                [
                    "Distribution",
                    {"value": "'" + self.output_particle_definition + "'"},
                ],
                ["high_res", {"value": self.high_res, "default": True}],
                ["n_red", {"value": self.sample_interval, "default": 1}],
                ["auto_phase", {"value": self.auto_phase, "default": True}],
                ["Toff", {"value": self.toffset, "default": None}],
            ]
        )
        if self.bunch_charge is not None:
            astradict["Qbunch"] = {"value": 1e9 * self.bunch_charge, "default": None}
        return astradict

    def hdf5_to_astra(self, prefix=""):
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
        rbf.hdf5.rotate_beamXZ(
            self.global_parameters["beam"],
            self.starting_rotation,
            preOffset=self.starting_offset,
        )
        # shutil.copyfile(self.global_parameters['master_subdir'] + '/' + HDF5filename, self.global_parameters['master_subdir'] + '/' + self.output_particle_definition.replace('.astra','')+'.hdf5') #copy src to dst
        astrabeamfilename = self.output_particle_definition
        rbf.astra.write_astra_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + astrabeamfilename,
            normaliseZ=False,
        )


class astra_output(astra_header):
    def __init__(self, screens, offset, rotation, **kwargs):
        super(astra_header, self).__init__("output", "astra_output", **kwargs)
        self.screens = screens
        self.starting_offset = offset
        self.starting_rotation = rotation

    def framework_dict(self):
        self.start_element.starting_offset = self.starting_offset
        self.end_element.starting_offset = self.starting_offset
        self.start_element.starting_rotation = self.starting_rotation
        self.end_element.starting_rotation = self.starting_rotation
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
            ]
        )
        for i, element in enumerate(self.screens, 1):
            element.starting_offset = self.starting_offset
            element.starting_rotation = self.starting_rotation
            keyworddict["Screen(" + str(i) + ")"] = {"value": element.middle[2]}
            # if abs(element.theta) > 0:
            # keyworddict['Scr_xrot('+str(i)+')'] = {'value': element.theta}
        return keyworddict


class astra_charge(astra_header):
    def __init__(self, **kwargs):
        super(astra_header, self).__init__("charge", "astra_charge", **kwargs)
        self.npart = 2 ** (3 * 5)
        self.sample_interval = 1
        self.grids = getGrids()

    @property
    def space_charge(self):
        return not (
            self.space_charge_mode == "False"
            or self.space_charge_mode is False
            or self.space_charge_mode is None
            or self.space_charge_mode == "None"
        )

    @property
    def space_charge_2D(self):
        return self.space_charge and self.space_charge_mode != "3D"

    @property
    def space_charge_3D(self):
        return self.space_charge and not self.space_charge_2D

    @property
    def grid_size(self):
        # print 'asking for grid sizes n = ', self.npart, ' is ', self.grids.getGridSizes(self.npart)
        return self.grids.getGridSizes((self.npart / self.sample_interval))

    def framework_dict(self):
        sc_dict = dict(
            [
                ["Lmirror", {"value": self.cathode, "default": False}],
                ["cell_var", {"value": self.cell_var, "default": None}],
                ["min_grid", {"value": self.min_grid, "default": None}],
                ["max_scale", {"value": self.max_scale, "default": None}],
                ["LSPCH", {"value": self.space_charge, "default": True}],
                ["LSPCH3D", {"value": self.space_charge_3D, "default": True}],
            ]
        )
        if self.space_charge_2D:
            sc_n_dict = dict(
                [
                    ["nrad", {"value": self.grid_size, "default": 32}],
                    ["nlong_in", {"value": self.grid_size, "default": 32}],
                ]
            )
            if hasattr(self, "nrad"):
                sc_n_dict.update({"nrad": {"value": self.nrad}})
            if hasattr(self, "nlong_in"):
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
        return merge_two_dicts(sc_dict, sc_n_dict)


class astra_errors(astra_header):
    def __init__(self, element=None, **kwargs):
        super(astra_errors, self).__init__("astra_error", "global_error", **kwargs)
        self._element = element
        self.add_property("global_errors", True)
        self.add_property("Log_Error", True)
        self.add_property("generate_output", True)
        self.add_property("Suppress_output", False)

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
            # print k, conversion[k]
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
