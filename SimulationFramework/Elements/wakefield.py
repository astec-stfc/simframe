from SimulationFramework.Elements.cavity import cavity
from SimulationFramework.Elements.drift import drift
from SimulationFramework.Framework_objects import elements_Elegant
from SimulationFramework.Modules.merge_two_dicts import merge_two_dicts


class wakefield(cavity):

    def __init__(self, name=None, type="longitudinal_wakefield", **kwargs):
        super().__init__(name, type, **kwargs)
        self.add_default("coupling_cell_length", 0)

    def _write_ASTRA(self, startn):
        field_ref_pos = self.get_field_reference_position()
        field_file_name = self.generate_field_file_name(self.field_definition, code="astra")
        efield_def = ["Wk_filename", {"value": "'" + field_file_name + "'", "default": ""}]
        output = ""
        if self.scale_kick > 0:
            for n in range(startn, startn + self.cells):
                output += self._write_ASTRA_dictionary(
                    dict(
                        [
                            [
                                "Wk_Type",
                                {
                                    "value": self.waketype,
                                    "default": "'Taylor_Method_F'",
                                },
                            ],
                            efield_def,
                            ["Wk_x", {"value": self.x_offset, "default": 0}],
                            ["Wk_y", {"value": self.y_offset, "default": 0}],
                            [
                                "Wk_z",
                                {
                                    "value": field_ref_pos[2]
                                    + self.coupling_cell_length
                                    + (0.5 + n - 1) * self.cell_length
                                },
                            ],
                            ["Wk_ex", {"value": self.scale_field_ex, "default": 0}],
                            ["Wk_ey", {"value": self.scale_field_ey, "default": 0}],
                            ["Wk_ez", {"value": self.scale_field_ez, "default": 1}],
                            ["Wk_hx", {"value": self.scale_field_hx, "default": 1}],
                            ["Wk_hy", {"value": self.scale_field_hy, "default": 0}],
                            ["Wk_hz", {"value": self.scale_field_hz, "default": 0}],
                            [
                                "Wk_equi_grid",
                                {"value": self.equal_grid, "default": 0.66},
                            ],
                            ["Wk_N_bin", {"value": 10, "default": 100}],
                            [
                                "Wk_ip_method",
                                {"value": self.interpolation_method, "default": 2},
                            ],
                            ["Wk_smooth", {"value": self.smooth, "default": 0.25}],
                            ["Wk_sub", {"value": self.subbins, "default": 10}],
                            [
                                "Wk_scaling",
                                {"value": 1 * self.scale_kick, "default": 1},
                            ],
                        ]
                    ),
                    n,
                )
                output += "\n"
            output += "\n"
        return output

    def _write_Elegant(self):
        field_file_name = self.generate_field_file_name(self.field_definition, code="elegant")
        wholestring = ""
        etype = self._convertType_Elegant(self.objecttype)
        string = self.objectname + ": " + etype
        if self.length > 0:
            d = drift(
                self.objectname + "-drift", type="drift", **{"length": self.length}
            )
            wholestring += d._write_Elegant()
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
        return wholestring
