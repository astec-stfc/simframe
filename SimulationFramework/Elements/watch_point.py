from SimulationFramework.Elements.screen import screen


class watch_point(screen):

    def __init__(self, name=None, type="watch_point", **kwargs):
        super().__init__(name, "screen", **kwargs)
