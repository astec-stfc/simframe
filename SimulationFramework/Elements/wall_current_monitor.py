from SimulationFramework.Framework_objects import frameworkElement


class wall_current_monitor(frameworkElement):

    def __init__(self, name=None, type="wall_current_monitor", **kwargs):
        super().__init__(name, type, **kwargs)
