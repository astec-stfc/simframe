from SimulationFramework.Framework_objects import frameworkElement, expand_substitution


class quadrupole(frameworkElement):

    def __init__(self, name=None, type="quadrupole", **kwargs):
        super().__init__(name, type, **kwargs)
        self.add_default("k1l", 0)
        self.add_default("n_kicks", 4)
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

    def update_field_definition(self) -> None:
        """Updates the field definitions to allow for the relative sub-directory location"""
        if hasattr(self, "field_definition") and self.field_definition is not None:
            self.field_definition = expand_substitution(self, self.field_definition)

    def write_ASTRA(self, n: int, **kwargs) -> str:
        astradict = dict(
            [
                ["Q_pos", {"value": self.middle[2] + self.dz, "default": 0}],
                ["Q_xoff", {"value": self.middle[0], "default": 0, "type": "not_zero"}],
                [
                    "Q_yoff",
                    {
                        "value": self.middle[1] + self.dy,
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
            self.generate_field_file_name(self.field_definition)
            astradict.update(
                dict(
                    [
                        ["Q_type", {"value": self.field_definition, "default": None}],
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
            return self._write_ASTRA(astradict, n)
        else:
            return None

    def write_GPT(self, Brho, ccs="wcs", *args, **kwargs):
        ccs_label, value_text = ccs.ccs_text(self.middle, self.rotation)
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
