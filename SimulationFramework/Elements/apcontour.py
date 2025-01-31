from SimulationFramework.Framework_objects import frameworkElement


class apcontour(frameworkElement):

    def __init__(self, name=None, type="apcontour", **kwargs):
        super().__init__(name, type, **kwargs)
        self.add_default("resolution", 0.001)
