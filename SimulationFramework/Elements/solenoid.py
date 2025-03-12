from SimulationFramework.Framework_objects import frameworkElement, expand_substitution


class solenoid(frameworkElement):

    def __init__(self, name=None, type="solenoid", **kwargs):
        super().__init__(name, type, **kwargs)
        self.add_default("scale_field", True)
        self.add_default("field_scale", 1)
        self.add_default("field_type", "1DMagnetoStatic")
        self.add_default("default_array_names", ["Z", "Bz"])

    def _write_ASTRA(self, n, **kwargs):
        field_ref_pos = self.get_field_reference_position()
        field_file_name = self.generate_field_file_name(self.field_definition, code="astra")
        efield_def = ["FILE_BFieLD", {"value": "'" + field_file_name + "'", "default": ""}]
        return self._write_ASTRA_dictionary(
            dict(
                [
                    ["S_pos", {"value": field_ref_pos[2] + self.dz, "default": 0}],
                    efield_def,
                    ["MaxB", {"value": self.get_field_amplitude, "default": 0}],
                    ["S_smooth", {"value": self.smooth, "default": 10}],
                    ["S_xoff", {"value": field_ref_pos[0] + self.dx, "default": 0}],
                    ["S_yoff", {"value": field_ref_pos[1] + self.dy, "default": 0}],
                    ["S_xrot", {"value": self.y_rot + self.dy_rot, "default": 0}],
                    ["S_yrot", {"value": self.x_rot + self.dx_rot, "default": 0}],
                ]
            ),
            n,
        )

    def _write_GPT(self, Brho, ccs, *args, **kwargs):
        field_ref_pos = self.get_field_reference_position()
        field_file_name = self.generate_field_file_name(self.field_definition, code="gpt")
        ccs_label, value_text = ccs.ccs_text(field_ref_pos, self.rotation)
        if self.field_type.lower() == "1dmagnetostatic":
            self.default_array_names = ["z", "Bz"]
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
                + str(field_file_name)
                + '", '
                + self.array_names_string()
                + ", "
                + str(expand_substitution(self, self.get_field_amplitude))
                + ");\n"
            )
        elif self.field_type.lower() == "3dmagnetostatic":
            self.default_array_names = ["x", "y", "z", "Bx", "By", "Bz"]
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
                + str(field_file_name)
                + '", '
                + self.array_names_string()
                + ", "
                + str(expand_substitution(self, self.get_field_amplitude))
                + ");\n"
            )
        return output
