from SimulationFramework.Elements.aperture import aperture


class rcollimator(aperture):

    def __init__(self, name=None, type="rcollimator", **kwargs):
        super().__init__(name, type, **kwargs)
