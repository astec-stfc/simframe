from SimulationFramework.Framework_objects import frameworkElement, elements_Elegant
from SimulationFramework.Modules.merge_two_dicts import merge_two_dicts


class fel_modulator(frameworkElement):
    """
    Class defining a modulator undulator.
    """

    n_steps: int = None
    """Number of steps for tracking"""

    periods: int = None
    """Number of periods in the modulator"""

    peak_field: float = None
    """Peak magnetic field"""
    gradient: float = None
    """Gradient of magnetic field"""

    method: str = None
    """Tracking method"""

    wavelength: float = None
    """Wavelength of laser"""

    peak_power: float = None
    """Laser peak power"""

    phase: float = None
    """Laser phase"""

    horizontal_mode_number: int = None
    """Horizontal mode number"""

    vertical_mode_number: int = None
    """Vertical mode number"""

    helical: bool = None
    """Flag to indicate whether a helical undulator is used"""

    time_offset: float = None
    """Time offset between laser pulse and particle beam"""

    def __init__(
        self,
        objecttype="modulator",
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
        for key, value in self.objectproperties:
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
