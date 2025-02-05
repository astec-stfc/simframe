from SimulationFramework.Elements.screen import screen


class beam_position_monitor(screen):

    def __init__(self, name=None, type="beam_position_monitor", **kwargs):
        super().__init__(name, type, **kwargs)

    def _write_ASTRA(self, n, **kwargs):
        return self._write_ASTRA_dictionary(
            dict(
                [
                    ["Screen", {"value": self.middle[2], "default": 0}],
                    ["Scr_xrot", {"value": self.y_rot + self.dy_rot, "default": 0}],
                    ["Scr_yrot", {"value": self.x_rot + self.dx_rot, "default": 0}],
                ]
            ),
            n,
        )
