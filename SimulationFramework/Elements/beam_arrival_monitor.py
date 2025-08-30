from SimulationFramework.Elements.screen import screen


class beam_arrival_monitor(screen):
    """
    Class defining a beam arrival monitor.
    """

    def _write_ASTRA(self, n, **kwargs):
        return ""
