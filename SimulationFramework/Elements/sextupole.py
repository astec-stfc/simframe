from SimulationFramework.Framework_objects import frameworkElement


class sextupole(frameworkElement):
    """
    Class defining a sextupole element
    """

    k2l: float = 0.0
    """Sextupole strength"""

    n_kicks: int = 20
    """Number of kicks for sextupole tracking"""

    strength_errors: list = [0]
    """Sextupole strength errors"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(sextupole, self).__init__(
            *args,
            **kwargs,
        )

    @property
    def k2(self) -> float:
        """
        Sextupole strength in m^-3

        Returns
        -------
        float
            Sextupole strength
        """
        return self.k2l / self.length

    @k2.setter
    def k2(self, k2: float) -> None:
        """
        Setter for sextupole strength

        Parameters
        ----------
        k3: float
            Sextupole strength
        """
        self.k2l = self.length * k2

    @property
    def dk2(self) -> float:
        """
        Normalised sextupole strength error

        Returns
        -------
        float:
            Sextupole strength error
        """
        return self.strength_errors[0]

    @dk2.setter
    def dk2(self, dk2: float) -> None:
        """
        Setter for sextupole strength error

        Parameters
        ----------
        dk3: float
            Sextupole strength error
        """
        self.strength_errors[0] = dk2

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
            + str(-Brho * self.k2)
            + ");\n"
        )
        return output
