from SimulationFramework.Framework_objects import csrdrift


class shutter(csrdrift):

    def __init__(self, name=None, type="shutter", **kwargs):
        super().__init__(name, type, **kwargs)
