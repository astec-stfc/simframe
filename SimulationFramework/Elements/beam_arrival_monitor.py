from SimulationFramework.Elements.screen import screen


class beam_arrival_monitor(screen):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(beam_arrival_monitor, self).__init__(
            *args,
            **kwargs
        )

    def _write_ASTRA(self, n, **kwargs):
        return ""
