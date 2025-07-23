from SimulationFramework.Framework_objects import frameworkElement


class octupole(frameworkElement):
    k3l: float = 0.0
    n_kicks: int = 20
    strength_errors: list = [0]

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
    def k3(self):
        return self.k3l / self.length

    @k3.setter
    def k3(self, k3):
        self.k3l = self.length * k3

    @property
    def dk3(self):
        return self.strength_errors[0]

    @dk3.setter
    def dk3(self, dk3):
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
