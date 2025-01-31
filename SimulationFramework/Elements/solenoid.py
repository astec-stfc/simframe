from SimulationFramework.Framework_objects import frameworkElement, expand_substitution


class solenoid(frameworkElement):

    def __init__(self, name=None, type="solenoid", **kwargs):
        super().__init__(name, type, **kwargs)
        self.add_default("scale_field", True)
        self.add_default("field_scale", 1)
        self.add_default("field_type", "1D")
        self.add_default("default_array_names", ["Z", "Bz"])

    def update_field_definition(self):
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

    def write_ASTRA(self, n, **kwargs):
        basename = self.generate_field_file_name(self.field_definition)
        efield_def = ["FILE_BFieLD", {"value": "'" + basename + "'", "default": ""}]
        return self._write_ASTRA(
            dict(
                [
                    ["S_pos", {"value": self.middle[2] + self.dz, "default": 0}],
                    efield_def,
                    ["MaxB", {"value": self.get_field_amplitude, "default": 0}],
                    ["S_smooth", {"value": self.smooth, "default": 10}],
                    ["S_xoff", {"value": self.middle[0] + self.dx, "default": 0}],
                    ["S_yoff", {"value": self.middle[1] + self.dy, "default": 0}],
                    ["S_xrot", {"value": self.y_rot + self.dy_rot, "default": 0}],
                    ["S_yrot", {"value": self.x_rot + self.dx_rot, "default": 0}],
                    ["S_noscale", {"value": not self.scale_field, "default": False}],
                ]
            ),
            n,
        )

    def write_GPT(self, Brho, ccs, *args, **kwargs):
        ccs_label, value_text = ccs.ccs_text(self.middle, self.rotation)
        if self.field_type.lower() == "1d":
            self.default_array_names = ["Z", "Bz"]
            """
            map1D_B("wcs",xOffset,0,zOffset+0.,cos(angle),0,-sin(angle),0,1,0,"bas_sol_norm.gdf","Z","Bz",gunSolField);
            """
            output = (
                "map1D_B"
                + "( "
                + ccs.name
                + ", "
                + ccs_label
                + ", "
                + value_text
                + ", "
                + '"'
                + str(self.generate_field_file_name(self.field_definition_gdf))
                + '", '
                + self.array_names_string()
                + ", "
                + str(expand_substitution(self, self.field_amplitude))
                + ");\n"
            )
        elif self.field_type.lower() == "3d":
            self.default_array_names = ["X", "Y", "Z", "Bx", "By", "Bz"]
            """
            map3D_B("wcs", xOffset,0,zOffset+0.,cos(angle),0,-sin(angle),0,1,0, "sol3.gdf", "x", "y", "z", "Bx", "By", "Bz", scale3);
            """
            output = (
                "map3D_B"
                + "( "
                + ccs.name
                + ", "
                + ccs_label
                + ", "
                + value_text
                + ", "
                + '"'
                + str(self.generate_field_file_name(self.field_definition_gdf))
                + '", '
                + self.array_names_string()
                + ", "
                + str(expand_substitution(self, self.field_amplitude))
                + ");\n"
            )
        return output
