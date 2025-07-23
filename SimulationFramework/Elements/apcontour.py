from SimulationFramework.Framework_objects import frameworkElement


class apcontour(frameworkElement):
    resolution: float = 0.001

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(apcontour, self).__init__(
            *args,
            **kwargs
        )
        self.add_default("resolution", 0.001)
