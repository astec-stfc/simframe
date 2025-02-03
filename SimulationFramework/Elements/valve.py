from SimulationFramework.Framework_objects import csrdrift


class valve(csrdrift):

    def __init__(self, name=None, type="valve", **kwargs):
        super().__init__(name, type, **kwargs)
