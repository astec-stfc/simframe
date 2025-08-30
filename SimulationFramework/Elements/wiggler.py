from SimulationFramework.Framework_objects import frameworkElement, elements_Elegant


class wiggler(frameworkElement):
    """
    Class defining a wiggler element
    """

    k: float = 0.0
    """Wiggler strength parameter"""

    peak_field: float = 0.0
    """Wiggler peak magnetic field"""

    radius: float = 0.0
    """Wiggler radius"""

    def _write_ASTRA(self, n: int, **kwargs) -> str:
        """
        Writes the wiggler element string for ASTRA.

        Parameters
        ----------
        n: int
            Wiggler index

        Returns
        -------
        str
            String representation of the element for ASTRA
        """
        return self._write_ASTRA_dictionary(
            dict(
                [
                    ["Q_pos", {"value": self.middle[2] + self.dz, "default": 0}],
                ]
            ),
            n,
        )

    def _write_Elegant(self) -> str:
        """
        Writes the wiggler element string for ELEGANT.

        Returns
        -------
        str
            String representation of the element for ELEGANT
        """
        wholestring = ""
        if (
            (hasattr(self, "k") and abs(self.k) > 0)
            or (hasattr(self, "peak_field") and abs(self.peak_field) > 0)
            or (hasattr(self, "radius") and abs(self.radius) > 0)
        ):
            etype = self._convertType_Elegant(self.objecttype)
        else:
            etype = "drift"
        string = self.objectname + ": " + etype
        for key, value in self.objectproperties.items():
            if (
                not key == "name"
                and not key == "type"
                and not key == "commandtype"
                and self._convertKeyword_Elegant(key) in elements_Elegant[etype]
            ):
                value = (
                    getattr(self, key)
                    if hasattr(self, key) and getattr(self, key) is not None
                    else value
                )
                key = self._convertKeyword_Elegant(key)
                tmpstring = ", " + key + " = " + str(value)
                if len(string + tmpstring) > 76:
                    wholestring += string + ",&\n"
                    string = ""
                    string += tmpstring[2::]
                else:
                    string += tmpstring
        wholestring += string + ";\n"
        return wholestring
