from SimulationFramework.Framework_objects import frameworkElement
from typing import List
from pydantic import computed_field

class quadrupole(frameworkElement):
    """
    Class defining a quadrupole magnet
    """

    k1l: float = 0.0
    """Quadrupole strength"""

    n_kicks: int = 4
    """Number of kicks for tracking through the quad"""

    field_reference_position: str = "middle"
    """Reference position for quadrupole field file"""

    fringe_field_coefficient: float = 0.0
    """Quad fringe field coefficient"""

    strength_errors: list = [0]
    """Quadrupole strength errors"""

    gradient: float = 0.0
    """Magnetic field gradient"""

    scale_field: bool = False
    """Flag indicating whether to scale the field from the field file"""

    multipoles: List[float] = [0]
    """Multipole elements in the quad"""

    smooth: int | float = 2
    """Number of points to smooth the field map [ASTRA only]"""

    bore: float | None = None
    """Bore radius of the quadrupole"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(quadrupole, self).__init__(
            *args,
            **kwargs,
        )

    @computed_field
    @property
    def k1(self) -> float:
        """
        Quadrupole K1 value in m^-2

        Returns
        --------
        float
            Quadrupole K1
        """
        return float(self.k1l) / float(self.length) if self.length > 0 else 0

    @k1.setter
    def k1(self, k1: float) -> None:
        """
        Set the quadrupole K1

        Parameters
        ----------
        k1: float
            Quadrupole K1
        """
        self.k1l = float(self.length) * float(k1) if self.length > 0 else k1

    @property
    def dk1(self) -> float:
        """
        Quadrupole strength error

        Returns
        -------
        float
            Quadrupole strength error
        """
        return self.strength_errors[0]

    @dk1.setter
    def dk1(self, dk1: float) -> None:
        """
        Set the quadrupole strength error

        Parameters
        ----------
        dk1: float
            Quadrupole strength error
        """
        self.strength_errors[0] = dk1

    def _write_ASTRA(self, n: int) -> str | None:
        """
        Writes the quadrupole element string for ASTRA.

        Note that in astra `Q_xrot` means a rotation about the y-axis and vice versa.

        Parameters
        ----------
        n: int
            Dipole index

        Returns
        -------
        str or None
            String representation of the element for ASTRA, or None if quadrupole strength is zero
        """
        field_ref_pos = self.get_field_reference_position()
        astradict = dict(
            [
                ["Q_pos", {"value": field_ref_pos[2] + self.dz, "default": 0}],
                [
                    "Q_xoff",
                    {"value": field_ref_pos[0], "default": 0, "type": "not_zero"},
                ],
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
                ["Q_bore", {"value": self.bore, "default": 0.037, "type": "not_zero"}],
                ["Q_noscale", {"value": self.scale_field}],
                ["Q_mult_a", {"type": "list", "value": self.multipoles}],
            ]
        )
        if self.field_definition:
            field_file_name = self.generate_field_file_name(
                self.field_definition, code="astra"
            )
            astradict.update(
                dict(
                    [
                        [
                            "Q_type",
                            {"value": "'" + field_file_name + "'", "default": None},
                        ],
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
            + str((-Brho * self.k1) if abs(self.gradient) == 0 else -1 * self.gradient)
            + (
                ", " + str(self.fringe_field_coefficient)
                if self.fringe_field_coefficient > 0
                else ""
            )
            + ");\n"
        )
        return output
