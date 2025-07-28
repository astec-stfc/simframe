from SimulationFramework.Framework_objects import frameworkElement, elements_Elegant
from SimulationFramework.Modules.merge_two_dicts import merge_two_dicts


class cleaner(frameworkElement):

    def __init__(
            self,
            objecttype: str = "scatter",
            *args,
            **kwargs,
    ):
        super(cleaner, self).__init__(
            objecttype=objecttype,
            *args,
            **kwargs,
        )

    def _write_Elegant(self):
        wholestring = ""
        etype = "clean"
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
