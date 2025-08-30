from SimulationFramework.Elements.screen import screen


class marker(screen):
    """
    Class defining a marker element.
    """

    def _write_CSRTrack(self, n):
        return ""

    def _write_Elegant(self) -> str:
        """
        Writes the marker element string for ELEGANT (same as :class:`~SimulationFramework.Elements.screen`).

        Returns
        -------
        str or None
            String representation of the element for ELEGANT
        """
        obj = self.objecttype
        self.objecttype = "screen"
        output = super()._write_Elegant()
        self.objecttype = obj
        return output
