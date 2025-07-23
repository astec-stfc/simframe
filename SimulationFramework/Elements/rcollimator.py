from SimulationFramework.Elements.aperture import aperture


class rcollimator(aperture):

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super(rcollimator, self).__init__(
            *args,
            **kwargs,
        )
