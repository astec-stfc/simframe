from SimulationFramework.Framework_objects import frameworkElement


class global_error(frameworkElement):

    def __init__(self, name=None, type="global_error", **kwargs):
        super().__init__(name, "global_error", **kwargs)
        # self._errordict = {}

    def add_Error(self, type, sigma):
        if type in global_Error_Types:
            self.add_property(type, sigma)

    def write_ASTRA(self):
        return self._write_ASTRA(
            dict([[key, {"value": value}] for key, value in self._errordict])
        )

    def write_GPT(self, Brho, ccs="wcs", *args, **kwargs):
        relpos, relrot = ccs.relative_position(self.middle, [0, 0, 0])
        coord = self.gpt_coordinates(relpos, relrot)
        output = (
            str(self.objecttype)
            + "( "
            + ccs.name
            + ", "
            + coord
            + ", "
            + str(self.length)
            + ", "
            + str(Brho * self.k1)
            + ");\n"
        )
        return output
