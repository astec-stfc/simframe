from SimulationFramework.Elements.cavity import cavity, elements_Elegant


class rf_deflecting_cavity(cavity):
    """
    Class defining a transverse deflecting cavity (TDC) element.
    """

    n_kicks: int = 10
    """Number of TDC kicks"""

    n_cells: int | float | None = 1
    """Number of cells"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(rf_deflecting_cavity, self).__init__(
            *args,
            **kwargs,
        )

    def _write_Elegant(self) -> str:
        """
        Writes the TDC element string for ELEGANT.

        Returns
        -------
        str or None
            String representation of the element for ELEGANT
        """
        wholestring = ""
        etype = self._convertType_Elegant(self.objecttype)
        string = self.objectname + ": " + etype
        for key, value in self.objectproperties.items():
            # print('RFDF before', key, value)
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
                key = self._convertKeyword_Elegant(key).lower()
                # In ELEGANT the voltages need to be compensated
                # value = abs((self.cells+4.1) * self.cell_length * (1 / np.sqrt(2)) * value) if key == 'voltage' else value
                # In CAVITY NKICK = n_cells
                value = 0 if key == "n_kicks" else value
                if key == "n_bins" and value > 0:
                    print(
                        "WARNING: Cavity n_bins is not zero - check log file to ensure correct behaviour!"
                    )
                value = 1 if value is True else value
                value = 0 if value is False else value
                # print('RFDF after', key, value)
                tmpstring = ", " + key + " = " + str(value)
                if len(string + tmpstring) > 76:
                    wholestring += string + ",&\n"
                    string = ""
                    string += tmpstring[2::]
                else:
                    string += tmpstring
        wholestring += string + ";\n"
        return wholestring
