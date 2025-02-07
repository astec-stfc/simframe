from SimulationFramework.Framework_objects import frameworkElement


class quadrupole(frameworkElement):

    def __init__(self, name=None, type="quadrupole", **kwargs):
        super().__init__(name, type, **kwargs)
        self.add_default("k1l", 0)
        self.add_default("n_kicks", 4)
        self.add_default("field_reference_position", "middle")
        self.strength_errors = [0]

    @property
    def k1(self) -> float:
        """Return the quadrupole K1 value in m^-2"""
        return float(self.k1l) / float(self.length) if self.length > 0 else 0

    @k1.setter
    def k1(self, k1):
        self.k1l = self.length * k1

    @property
    def dk1(self):
        return self.strength_errors[0]

    @dk1.setter
    def dk1(self, dk1):
        self.strength_errors[0] = dk1

    def _write_ASTRA(self, n: int) -> str:
        field_ref_pos = self.get_field_reference_position()
        astradict = dict(
            [
                ["Q_pos", {"value": field_ref_pos[2] + self.dz, "default": 0}],
                ["Q_xoff", {"value": field_ref_pos[0], "default": 0, "type": "not_zero"}],
                [
                    "Q_yoff",
                    {
                        "value": field_ref_pos[1] + self.dy,
                        "default": None,
                        "type": "not_zero",
                    },
                ],
                [
                    "Q_xrot",
                    {
                        "value": -1 * self.y_rot + self.dy_rot,
                        "default": None,
                        "type": "not_zero",
                    },
                ],
                [
                    "Q_yrot",
                    {
                        "value": -1 * self.x_rot + self.dx_rot,
                        "default": None,
                        "type": "not_zero",
                    },
                ],
                [
                    "Q_zrot",
                    {
                        "value": -1 * self.z_rot + self.dz_rot,
                        "default": None,
                        "type": "not_zero",
                    },
                ],
                ["Q_smooth", {"value": self.smooth, "default": None}],
                ["Q_bore", {"value": self.bore, "default": None, "type": "not_zero"}],
                ["Q_noscale", {"value": self.scale_field}],
                ["Q_mult_a", {"type": "list", "value": self.multipoles}],
            ]
        )
        if self.field_definition:
            basename = self.generate_field_file_name(self.field_definition)
            astradict.update(
                dict(
                    [
                        ["Q_type", {"value": "'" + basename + "'", "default": None}],
                        ["q_grad", {"value": self.gradient, "default": None}],
                    ]
                )
            )
        elif abs(self.k1 + self.dk1) > 0:
            astradict.update(
                dict(
                    [
                        ["Q_k", {"value": self.k1 + self.dk1, "default": 0}],
                        ["Q_length", {"value": self.length, "default": 0}],
                    ]
                )
            )
        if abs(self.k1 + self.dk1) > 0 or self.field_definition:
            return self._write_ASTRA_dictionary(astradict, n)
        else:
            return None

    def _write_GPT(self, Brho, ccs="wcs", *args, **kwargs):
        field_ref_pos = self.get_field_reference_position()
        ccs_label, value_text = ccs.ccs_text(field_ref_pos, self.rotation)
        output = (
            str(self.objecttype)
            + "( "
            + ccs.name
            + ", "
            + ccs_label
            + ", "
            + value_text
            + ", "
            + str(self.length)
            + ", "
            + str((-Brho * self.k1) if not self.gradient else -1 * self.gradient)
            + (
                ", " + str(self.fringe_field_coefficient)
                if self.fringe_field_coefficient > 0
                else ""
            )
            + ");\n"
        )
        return output
