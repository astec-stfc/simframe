from SimulationFramework.Elements.screen import screen


class marker(screen):

    def __init__(self, name=None, type="marker", **kwargs):
        super().__init__(name, "screen", **kwargs)

    def _write_CSRTrack(self, n):
        return ""

    def _write_Elegant(self) -> str:
        obj = self.objecttype
        self.objecttype = "screen"
        output = super()._write_Elegant()
        self.objecttype = obj
        return output
