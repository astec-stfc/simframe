from SimulationFramework.Framework_objects import csrdrift


class bellows(csrdrift):

    def __init__(self, name=None, type="bellows", **kwargs):
        super().__init__(name, type, **kwargs)
