from SimulationFramework.Elements.aperture import aperture


class rcollimator(aperture):
    """
    Class defining a rectangular collimator.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(rcollimator, self).__init__(
            *args,
            **kwargs,
        )
