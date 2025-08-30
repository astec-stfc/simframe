from SimulationFramework.Elements.screen import screen


class bunch_length_monitor(screen):
    """
    Class defining a bunch length monitor
    """

    def _write_ASTRA(self, n, **kwargs):
        return ""
