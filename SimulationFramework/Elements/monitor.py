from SimulationFramework.Elements.screen import screen


class monitor(screen):

    def __init__(self, name=None, type="monitor", **kwargs):
        super().__init__(name, type, **kwargs)
