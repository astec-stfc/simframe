from SimulationFramework.Framework_objects import frameworkElement, elements_Elegant


class scatter(frameworkElement):
    probability: float = None
    horizontal_scatter: float = None
    vertical_scatter: float = None
    horizontal_momentum_scatter: float = None
    vertical_momentum_scatter: float = None
    relative_momentum_scatter: float = None

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(scatter, self).__init__(
            *args,
            **kwargs,
        )
        # print('Scatter object ', self.objectname,' - DP = ', self.objectproperties)

    def _write_Elegant(self):
        wholestring = ""
        etype = "scatter"
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
