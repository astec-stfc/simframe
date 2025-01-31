from SimulationFramework.Elements.aperture import aperture


class collimator(aperture):

    def __init__(self, name=None, type="collimator", **kwargs):
        super().__init__(name, type, **kwargs)
