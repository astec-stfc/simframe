from SimulationFramework.Elements.screen import screen


class bunch_length_monitor(screen):

    def __init__(self, name=None, type="beam_arrival_monitor", **kwargs):
        super().__init__(name, type, **kwargs)

    def write_ASTRA(self, n, **kwargs):
        return ""
