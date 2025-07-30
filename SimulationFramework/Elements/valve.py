from SimulationFramework.Framework_objects import csrdrift


class valve(csrdrift):
    """
    Class defining a vacuum valve element
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(valve, self).__init__(
            *args,
            **kwargs,
        )
