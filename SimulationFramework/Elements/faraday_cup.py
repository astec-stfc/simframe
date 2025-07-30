from SimulationFramework.Elements.wall_current_monitor import wall_current_monitor


class faraday_cup(wall_current_monitor):
    """
    Class defining a faraday cup element.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(faraday_cup, self).__init__(
            *args,
            **kwargs,
        )
