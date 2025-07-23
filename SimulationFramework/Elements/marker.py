from SimulationFramework.Elements.screen import screen


class marker(screen):

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super(marker, self).__init__(
            *args,
            **kwargs,
        )

    def _write_CSRTrack(self, n):
        return ""

    def _write_Elegant(self) -> str:
        obj = self.objecttype
        self.objecttype = "screen"
        output = super()._write_Elegant()
        self.objecttype = obj
        return output
