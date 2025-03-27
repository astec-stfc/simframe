from SimulationFramework.Framework_objects import (
    frameworkElement,
    elements_Elegant,
    type_conversion_rules_Ocelot,
)
from SimulationFramework.Modules.merge_two_dicts import merge_two_dicts
import numpy as np


class cavity(frameworkElement):

    def __init__(self, name=None, type="cavity", **kwargs):
        super().__init__(name, type, **kwargs)
        self.add_default("tcolumn", '"t"')
        self.add_default("zcolumn", '"z"')
        self.add_default("ezcolumn", '"Ez"')
        self.add_default("change_p0", 1)
        self.add_default("n_kicks", self.n_cells)
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
        self.add_default("field_reference_position", "start")

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

    def _write_ASTRA(self, n, **kwargs):
        field_ref_pos = self.get_field_reference_position()
        auto_phase = kwargs["auto_phase"] if "auto_phase" in kwargs else True
        crest = self.crest if not auto_phase else 0
        field_file_name = self.generate_field_file_name(
            self.field_definition, code="astra"
        )
        efield_def = [
            "FILE_EFieLD",
            {"value": "'" + field_file_name + "'", "default": ""},
        ]
        return self._write_ASTRA_dictionary(
            dict(
                [
                    ["C_pos", {"value": field_ref_pos[2] + self.dz, "default": 0}],
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
                            "value": field_ref_pos[0] + self.dx,
                            "default": None,
                            "type": "not_zero",
                        },
                    ],
                    [
                        "C_yoff",
                        {
                            "value": field_ref_pos[1] + self.dy,
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

    def set_wakefield_column_names(self, wakefield_file_name: str) -> None:
        self.tcolumn = '"t"'
        if self.wakefield_definition.field_type == "3DWake":
            self.wakefile = wakefield_file_name
            self.wxcolumn = '"Wx"'
            self.wycolumn = '"Wy"'
            self.wzcolumn = '"Wz"'
            self.wakefieldcolumstring = '"z", "Wx", "Wy", "Wz"'
        elif self.wakefield_definition.field_type == "LongitudinalWake":
            self.wzcolumn = '"Wz"'
            self.zwakefile = wakefield_file_name
            self.wakefieldcolumstring = '"z", "Wz"'
        elif self.wakefield_definition.field_type == "TransverseWake":
            self.wxcolumn = '"Wx"'
            self.wycolumn = '"Wy"'
            self.wakefieldcolumstring = '"z", "Wx", "Wy", "Wz"'
            self.trwakefile = wakefield_file_name

    def _write_Elegant(self):
        wholestring = ""
        etype = self._convertType_Elegant(self.objecttype)
        if (
            not hasattr(self, "wakefield_definition")
            or self.wakefield_definition is None
            or self.wakefield_definition == ""
        ):
            etype = "rfca"
            if self.field_definition is not None:
                etype = "rftmez0"
                self.ez_peak = self.field_amplitude
                self.field_file_name = self.generate_field_file_name(
                    self.field_definition, code="elegant"
                )
        else:
            wakefield_file_name = self.generate_field_file_name(
                self.wakefield_definition, code="elegant"
            )
            self.set_wakefield_column_names(wakefield_file_name)
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
                if hasattr(self, key) and getattr(self, key) is not None:
                    value = getattr(self, key)
                key = self._convertKeyword_Elegant(key).lower()
                # rftmez0 uses frequency instead of freq
                if etype == "rftmez0" and key == "freq":
                    key = "frequency"

                if (
                    self.objecttype == "cavity"
                    or self.objecttype == "rf_deflecting_cavity"
                ):
                    if key == "phase":
                        if etype == "rftmez0":
                            # If using rftmez0 or similar
                            value = (value / 360.0) * (2 * 3.14159)
                        else:
                            # In ELEGANT all phases are +90degrees!!
                            value = 90 - value

                    # In ELEGANT the voltages need to be compensated
                    if key == "volt":
                        value = abs(
                            (self.cells + 4.1)
                            * self.cell_length
                            * (1 / np.sqrt(2))
                            * value
                        )
                    # If using rftmez0 or similar
                    if key == "ez_peak":
                        value = abs(1e-3 / (np.sqrt(2)) * value)

                    # In CAVITY NKICK = n_cells
                    if key == "n_kicks" and self.cells > 0:
                        value = 3 * self.cells

                    if key == "n_bins" and value > 0:
                        print(
                            "WARNING: Cavity n_bins is not zero - check log file to ensure correct behaviour!"
                        )
                    value = 1 if value is True else value
                    value = 0 if value is False else value
                # print("elegant cavity", key, value)
                tmpstring = ", " + key + " = " + str(value)
                if len(string + tmpstring) > 76:
                    wholestring += string + ",&\n"
                    string = ""
                    string += tmpstring[2::]
                else:
                    string += tmpstring
        wholestring += string + ";\n"
        return wholestring

    def _write_Ocelot(self):
        obj = type_conversion_rules_Ocelot[self.objecttype](eid=self.objectname)
        k1 = self.k1 if self.k1 is not None else 0
        k2 = self.k2 if self.k2 is not None else 0
        keydict = merge_two_dicts(
            {"k1": k1, "k2": k2},
            merge_two_dicts(self.objectproperties, self.objectdefaults),
        )
        for key, value in keydict.items():
            if key not in [
                "name",
                "type",
                "commandtype",
            ]:  # and self._convertKeword_Ocelot(key) in elements_Ocelot[self.objecttype]:
                value = (
                    getattr(self, key)
                    if hasattr(self, key) and getattr(self, key) is not None
                    else value
                )
                if self.objecttype in ["cavity", "rf_deflecting_cavity"]:
                    if key == "field_amplitude":
                        value = (
                            value
                            * 1e-9
                            * abs(
                                (self.cells + 5.5) * self.cell_length * (1 / np.sqrt(2))
                            )
                        )
                setattr(obj, self._convertKeword_Ocelot(key), value)
        scr = type_conversion_rules_Ocelot["screen"](eid=f"{self.objectname}_END")
        return [obj, scr]

    def _write_GPT(self, Brho, ccs="wcs", *args, **kwargs):
        field_ref_pos = self.get_field_reference_position()
        ccs_label, value_text = ccs.ccs_text(field_ref_pos, self.rotation)
        relpos, _ = ccs.relative_position(field_ref_pos, self.global_rotation)
        field_file_name = self.generate_field_file_name(
            self.field_definition, code="gpt"
        )
        self.generate_field_file_name(
            self.wakefield_definition, code="gpt"
        )
        """
        map1D_TM("wcs","z",linacposition,"mockup2m.gdf","Z","Ez",ffacl,phil,w);
        wakefield("wcs","z",  6.78904 + 4.06667 / 2, 4.06667, 50, "Sz5um10mm.gdf", "z","","","Wz", "FieldFactorWz", 10 * 122 / 4.06667) ;
        """
        if self.crest is None:
            self.crest = 0
        subname = str(relpos[2]).replace(".", "")
        output = ""
        if field_file_name is not None:
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
                        (9.0 / (2.0 * np.pi))
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
                + str(field_file_name)
                + '", "z", "Ez", ffac'
                + subname
                + ", phi"
                + subname
                + ", w"
                + subname
                + ");\n"
            )
            # if wakefield_file_name is not None:
            #     wccs_label, wvalue_text = ccs.ccs_text(self.middle, self.rotation)
            #     self.set_wakefield_column_names(wakefield_file_name)
            #     output += (
            #         "wakefield("
            #         + ccs.name
            #         + ", "
            #         + wccs_label
            #         + ", "
            #         + wvalue_text
            #         + ", "
            #         + str(self.length)
            #         + ', 50, "'
            #         + str(wakefield_file_name)
            #         + '", '
            #         + self.wakefieldcolumstring
            #         + ', "FieldFactorWz", '
            #         + str(self.cells)
            #         + " / "
            #         + str(self.length)
            #         + ', "FieldFactorWx", '
            #         + str(self.cells)
            #         + " / "
            #         + str(self.length)
            #         + ', "FieldFactorWy", '
            #         + str(self.cells)
            #         + " / "
            #         + str(self.length)
            #         + ") ;\n"
            #     )
        else:
            output = ""
        return output
