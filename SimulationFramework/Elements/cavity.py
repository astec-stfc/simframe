from SimulationFramework.Framework_objects import frameworkElement, elements_Elegant
from SimulationFramework.FrameworkHelperFunctions import expand_substitution
from SimulationFramework.Modules.merge_two_dicts import merge_two_dicts
import numpy as np


class cavity(frameworkElement):

    def __init__(self, name=None, type="cavity", **kwargs):
        super().__init__(name, type, **kwargs)
        self.add_default("tcolumn", '"t"')
        self.add_default("zcolumn", '"Z"')
        self.add_default("ezcolumn", '"Ez"')
        self.add_default("wzcolumn", '"W"')
        self.add_default("wxcolumn", '"W"')
        self.add_default("wycolumn", '"W"')
        self.add_default("wcolumn", '"Ez"')
        self.add_default("change_p0", 1)
        self.add_default("n_kicks", self.n_cells)
        # self.add_default('method', '"non-adaptive runge-kutta"')
        self.add_default("end1_focus", 1)
        self.add_default("end2_focus", 1)
        self.add_default("body_focus_model", "SRS")
        self.add_default("lsc_bins", 100)
        self.add_default("current_bins", 0)
        self.add_default("interpolate_current_bins", 1)
        self.add_default("smooth_current_bins", 1)
        self.add_default("coupling_cell_length", 0)
        self.add_default("field_amplitude", 0)
        self.add_default("crest", 0)

    def update_field_definition(self) -> None:
        """Updates the field definitions to allow for the relative sub-directory location"""
        if hasattr(self, "field_definition") and self.field_definition is not None:
            self.field_definition = expand_substitution(self, self.field_definition)
        if (
            hasattr(self, "field_definition_sdds")
            and self.field_definition_sdds is not None
        ):
            self.field_definition_sdds = expand_substitution(
                self, self.field_definition_sdds
            )
        if (
            hasattr(self, "field_definition_gdf")
            and self.field_definition_gdf is not None
        ):
            self.field_definition_gdf = expand_substitution(
                self, self.field_definition_gdf
            )
        if (
            hasattr(self, "longitudinal_wakefield_sdds")
            and self.longitudinal_wakefield_sdds is not None
        ):
            self.longitudinal_wakefield_sdds = expand_substitution(
                self, self.longitudinal_wakefield_sdds
            )
        if (
            hasattr(self, "transverse_wakefield_sdds")
            and self.transverse_wakefield_sdds is not None
        ):
            self.transverse_wakefield_sdds = expand_substitution(
                self, self.transverse_wakefield_sdds
            )

    @property
    def cells(self):
        if (self.n_cells == 0 or self.n_cells is None) and self.cell_length > 0:
            cells = round((self.length - self.cell_length) / self.cell_length)
            cells = int(cells - (cells % 3))
        elif (
            self.n_cells > 0 and (self.cell_length is not None and self.cell_length) > 0
        ):
            if self.cell_length == self.length:
                cells = 1
            else:
                cells = int(self.n_cells - (self.n_cells % 3))
        else:
            cells = None
        return cells

    def write_ASTRA(self, n, **kwargs):
        auto_phase = kwargs["auto_phase"] if "auto_phase" in kwargs else True
        crest = self.crest if not auto_phase else 0
        basename = self.generate_field_file_name(self.field_definition)
        efield_def = ["FILE_EFieLD", {"value": "'" + basename + "'", "default": ""}]
        return self._write_ASTRA(
            dict(
                [
                    ["C_pos", {"value": self.start[2] + self.dz, "default": 0}],
                    efield_def,
                    ["C_numb", {"value": self.cells}],
                    ["Nue", {"value": float(self.frequency) / 1e9, "default": 2998.5}],
                    [
                        "MaxE",
                        {"value": float(self.field_amplitude) / 1e6, "default": 0},
                    ],
                    ["Phi", {"value": crest - self.phase, "default": 0.0}],
                    ["C_smooth", {"value": self.smooth, "default": None}],
                    [
                        "C_xoff",
                        {
                            "value": self.start[0] + self.dx,
                            "default": None,
                            "type": "not_zero",
                        },
                    ],
                    [
                        "C_yoff",
                        {
                            "value": self.start[1] + self.dy,
                            "default": None,
                            "type": "not_zero",
                        },
                    ],
                    [
                        "C_xrot",
                        {
                            "value": self.y_rot + self.dy_rot,
                            "default": None,
                            "type": "not_zero",
                        },
                    ],
                    [
                        "C_yrot",
                        {
                            "value": self.x_rot + self.dx_rot,
                            "default": None,
                            "type": "not_zero",
                        },
                    ],
                    [
                        "C_zrot",
                        {
                            "value": self.z_rot + self.dz_rot,
                            "default": None,
                            "type": "not_zero",
                        },
                    ],
                ]
            ),
            n,
        )

    def _write_Elegant(self):
        self.update_field_definition()
        original_field_definition_sdds = self.field_definition_sdds
        if self.field_definition_sdds is not None:
            self.field_definition_sdds = (
                '"' + self.generate_field_file_name(self.field_definition_sdds) + '"'
            )
        original_longitudinal_wakefield_sdds = self.longitudinal_wakefield_sdds
        if self.longitudinal_wakefield_sdds is not None:
            self.longitudinal_wakefield_sdds = (
                '"'
                + self.generate_field_file_name(self.longitudinal_wakefield_sdds)
                + '"'
            )
        original_transverse_wakefield_sdds = self.transverse_wakefield_sdds
        if self.transverse_wakefield_sdds is not None:
            self.transverse_wakefield_sdds = (
                '"'
                + self.generate_field_file_name(self.transverse_wakefield_sdds)
                + '"'
            )
        wholestring = ""
        etype = self._convertType_Elegant(self.objecttype)
        if (
            not hasattr(self, "longitudinal_wakefield_sdds")
            or self.longitudinal_wakefield_sdds is None
        ) and (
            not hasattr(self, "transverse_wakefield_sdds")
            or self.transverse_wakefield_sdds is None
        ):
            # print('cavity ', self.objectname, ' is an RFCA!')
            etype = "rfca"
        if self.field_definition_sdds is not None:
            etype = "rftmez0"
            self.ez_peak = self.field_amplitude
        string = self.objectname + ": " + etype
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
                key = self._convertKeyword_Elegant(key).lower()
                if etype == "rftmez0" and key == "freq":
                    key = "frequency"
                if (
                    self.objecttype == "cavity"
                    or self.objecttype == "rf_deflecting_cavity"
                ):
                    if etype == "rftmez0":
                        # If using rftmez0 or similar
                        value = (
                            ((value) / 360.0) * (2 * 3.14159)
                            if key == "phase"
                            else value
                        )
                    else:
                        # In ELEGANT all phases are +90degrees!!
                        value = 90 - value if key == "phase" else value
                    # In ELEGANT the voltages need to be compensated
                    value = (
                        abs(
                            (self.cells + 4.1)
                            * self.cell_length
                            * (1 / np.sqrt(2))
                            * value
                        )
                        if key == "volt"
                        else value
                    )
                    # If using rftmez0 or similar
                    value = (
                        abs(1e-3 / (np.sqrt(2)) * value) if key == "ez_peak" else value
                    )
                    # In CAVITY NKICK = n_cells
                    value = 3 * self.cells if key == "n_kicks" else value
                    if key == "n_bins" and value > 0:
                        print(
                            "WARNING: Cavity n_bins is not zero - check log file to ensure correct behaviour!"
                        )
                    value = 1 if value is True else value
                    value = 0 if value is False else value
                tmpstring = ", " + key + " = " + str(value)
                if len(string + tmpstring) > 76:
                    wholestring += string + ",&\n"
                    string = ""
                    string += tmpstring[2::]
                else:
                    string += tmpstring
        wholestring += string + ";\n"
        self.field_definition_sdds = original_field_definition_sdds
        self.longitudinal_wakefield_sdds = original_longitudinal_wakefield_sdds
        self.transverse_wakefield_sdds = original_transverse_wakefield_sdds
        return wholestring

    def write_GPT(self, Brho, ccs="wcs", *args, **kwargs):
        self.update_field_definition()
        ccs_label, value_text = ccs.ccs_text(self.middle, self.rotation)
        relpos, relrot = ccs.relative_position(self.middle, self.global_rotation)
        """
        map1D_TM("wcs","z",linacposition,"mockup2m.gdf","Z","Ez",ffacl,phil,w);
        wakefield("wcs","z",  6.78904 + 4.06667 / 2, 4.06667, 50, "Sz5um10mm.gdf", "z","","","Wz", "FieldFactorWz", 10 * 122 / 4.06667) ;
        """
        if self.crest is None:
            self.crest = 0
        subname = str(relpos[2]).replace(".", "")
        if expand_substitution(self, self.field_definition_gdf) is not None:
            output = (
                "f"
                + subname
                + " = "
                + str(self.frequency)
                + ";\n"
                + "w"
                + subname
                + " = 2*pi*f"
                + subname
                + ";\n"
                + "phi"
                + subname
                + " = "
                + str((self.crest + 90 - self.phase) % 360.0)
                + "/deg;\n"
            )
            if self.Structure_Type == "TravellingWave":
                output += (
                    "ffac"
                    + subname
                    + " = "
                    + str(
                        (1 + (0.005 * self.length**1.5))
                        * (9.0 / (2.0 * np.pi))
                        * self.field_amplitude
                    )
                    + ";\n"
                )
            else:
                output += "ffac" + subname + " = " + str(self.field_amplitude) + ";\n"

            # if False and self.Structure_Type == 'TravellingWave' and hasattr(self, 'attenuation_constant') and hasattr(self, 'shunt_impedance') and hasattr(self, 'design_power') and hasattr(self, 'design_gamma'):
            #     '''
            #     trwlinac(ECS,ao,Rs,Po,P,Go,thetao,phi,w,L)
            #     '''
            #     relpos, relrot = ccs.relative_position(self.middle, self.global_rotation)
            #     power = float(self.field_amplitude) / 25e6 * float(self.design_power)
            #     output += 'trwlinac' + '( ' + ccs.name + ', "z", '+ str(relpos[2]+self.coupling_cell_length) + ', ' + str(self.attenuation_constant / self.length) + ', ' + str(float(self.shunt_impedance) / self.length)\
            #             + ', ' + str(float(self.design_power) / self.length) + ', ' + str(power / self.length) + ', ' + str(1000/0.511) + ', ' + str(self.crest)\
            #             + ', '+str(self.phase)+', w'+subname+', ' + str(self.length) + ');\n'
            # else:
            output += (
                "map1D_TM"
                + "( "
                + ccs.name
                + ", "
                + ccs_label
                + ", "
                + value_text
                + ', "'
                + str(self.generate_field_file_name(self.field_definition_gdf))
                + '", "Z","Ez", ffac'
                + subname
                + ", phi"
                + subname
                + ", w"
                + subname
                + ");\n"
            )
            if expand_substitution(self, self.wakefield_gdf) is not None:
                output += (
                    "wakefield("
                    + ccs.name
                    + ", "
                    + ccs_label
                    + ", "
                    + value_text
                    + ", "
                    + str(self.length)
                    + ', 50, "'
                    + str(self.generate_field_file_name(self.wakefield_gdf))
                    + '", "z","Wx","Wy","Wz", "FieldFactorWz", '
                    + str(self.cells)
                    + " / "
                    + str(self.length)
                    + ', "FieldFactorWx", '
                    + str(self.cells)
                    + " / "
                    + str(self.length)
                    + ', "FieldFactorWy", '
                    + str(self.cells)
                    + " / "
                    + str(self.length)
                    + ") ;\n"
                )
            else:
                if (
                    expand_substitution(self, self.longitudinal_wakefield_gdf)
                    is not None
                ):
                    output += (
                        "wakefield("
                        + ccs.name
                        + ", "
                        + ccs_label
                        + ", "
                        + value_text
                        + ", "
                        + str(self.length)
                        + ', 50, "'
                        + str(
                            self.generate_field_file_name(
                                self.longitudinal_wakefield_gdf
                            )
                        )
                        + '", "z","","","Wz", "FieldFactorWz", '
                        + str(self.cells)
                        + " / "
                        + str(self.length)
                        + ") ;\n"
                    )
                if expand_substitution(self, self.transverse_wakefield_gdf) is not None:
                    output += (
                        "wakefield("
                        + ccs.name
                        + ", "
                        + ccs_label
                        + ", "
                        + value_text
                        + ", "
                        + str(self.length)
                        + ', 50, "'
                        + str(
                            self.generate_field_file_name(self.transverse_wakefield_gdf)
                        )
                        + '", "z","Wx","Wy","", "FieldFactorWx", '
                        + str(self.cells)
                        + " / "
                        + str(self.length)
                        + ', "FieldFactorWy", '
                        + str(self.cells)
                        + " / "
                        + str(self.length)
                        + ") ;\n"
                    )
        else:
            output = ""
        return output
