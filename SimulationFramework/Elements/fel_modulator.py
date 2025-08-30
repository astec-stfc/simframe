from SimulationFramework.Framework_objects import frameworkElement, elements_Elegant


class fel_modulator(frameworkElement):
    """
    Class defining a modulator undulator.
    """

    n_steps: int | None = None
    """Number of steps for tracking"""

    periods: int | None = None
    """Number of periods in the modulator"""

    peak_field: float = 0.0
    """Peak magnetic field"""

    gradient: float | None = None
    """Gradient of magnetic field"""

    method: str | None = None
    """Tracking method"""

    wavelength: float = 0.0
    """Wavelength of laser"""

    peak_power: float = 0.0
    """Laser peak power"""

    phase: float = 0.0
    """Laser phase"""

    horizontal_mode_number: int | None = None
    """Horizontal mode number"""

    vertical_mode_number: int | None = None
    """Vertical mode number"""

    helical: bool = False
    """Flag to indicate whether a helical undulator is used"""

    time_offset: float = 0.0
    """Time offset between laser pulse and particle beam"""

    def __init__(
        self,
        objecttype="fel_modulator",
        *args,
        **kwargs,
    ):
        super(fel_modulator, self).__init__(
            objecttype=objecttype,
            *args,
            **kwargs,
        )

    def _write_ASTRA(self, n, **kwargs) -> str:
        """
        Writes the modulator element string for ASTRA.
        #TODO is this a valid element?

        Parameters
        ----------
        n: int
            Modulator index

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
        Writes the modulator element string for ELEGANT.

        Returns
        -------
        str or None
            String representation of the element for ELEGANT
        """
        wholestring = ""
        etype = self._convertType_Elegant(self.objecttype)
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
