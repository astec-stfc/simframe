from SimulationFramework.Framework_objects import frameworkElement


class charge(frameworkElement):
    def __init__(self, name=None, type="charge", **kwargs):
        super().__init__(name, type, **kwargs)
