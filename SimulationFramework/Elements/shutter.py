from SimulationFramework.Framework_objects import csrdrift


class shutter(csrdrift):
    """
    Class defining a shutter element.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(shutter, self).__init__(
            *args,
            **kwargs,
        )
