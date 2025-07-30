from SimulationFramework.Framework_objects import frameworkElement


class octupole(frameworkElement):
    """
    Class defining an octupole element.
    """

    k3l: float = 0.0
    """Octupole strength"""

    n_kicks: int = 20
    """Number of kicks to apply"""

    strength_errors: list = [0]
    """Strength errors"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(octupole, self).__init__(
            *args,
            **kwargs,
        )

    @property
    def k3(self) -> float:
        """
        Octupole strength in m^-4

        Returns
        -------
        float
            Octupole strength
        """
        return self.k3l / self.length

    @k3.setter
    def k3(self, k3: float) -> None:
        """
        Setter for octupole strength

        Parameters
        ----------
        k3: float
            Octupole strength
        """
        self.k3l = self.length * k3

    @property
    def dk3(self) -> float:
        """
        Normalised octupole strength error

        Returns
        -------
        float:
            Octupole strength error
        """
        return self.strength_errors[0]

    @dk3.setter
    def dk3(self, dk3: float) -> None:
        """
        Setter for octupole strength error

        Parameters
        ----------
        dk3: float
            Octupole strength error
        """
        self.strength_errors[0] = dk3

    def _write_GPT(self, Brho, ccs="wcs", *args, **kwargs):
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
            + str(-Brho * self.k3)
            + ");\n"
        )
        return output
