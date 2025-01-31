from SimulationFramework.Framework_objects import frameworkElement


class sextupole(frameworkElement):

    def __init__(self, name=None, type="sextupole", **kwargs):
        super().__init__(name, type, **kwargs)
        self.add_default("k2l", 0)
        self.add_default("n_kicks", 20)
        self.strength_errors = [0]

    @property
    def k2(self):
        return self.k2l / self.length

    @k2.setter
    def k2(self, k2):
        self.k2l = self.length * k2

    @property
    def dk2(self):
        return self.strength_errors[0]

    @dk2.setter
    def dk2(self, dk2):
        self.strength_errors[0] = dk2

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
            + str(-Brho * self.k2)
            + ");\n"
        )
        return output
