from SimulationFramework.Elements.screen import screen


class bunch_length_monitor(screen):
    """
    Class defining a bunch length monitor
    """

    def __init__(self, *args, **kwargs):
        super(bunch_length_monitor, self).__init__(
            *args,
            **kwargs,
        )

    def _write_ASTRA(self, n, **kwargs):
        return ""
