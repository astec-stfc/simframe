from SimulationFramework.Framework_objects import frameworkElement


class wall_current_monitor(frameworkElement):
    """
    Class defining a wall current monitor element
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(wall_current_monitor, self).__init__(
            *args,
            **kwargs,
        )
