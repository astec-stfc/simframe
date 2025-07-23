from SimulationFramework.Elements.aperture import aperture


class collimator(aperture):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(collimator, self).__init__(
            *args,
            **kwargs,
        )
