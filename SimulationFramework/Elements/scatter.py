from SimulationFramework.Framework_objects import frameworkElement, elements_Elegant
from SimulationFramework.Modules.merge_two_dicts import merge_two_dicts


class scatter(frameworkElement):

    def __init__(self, name=None, type="scatter", **kwargs):
        super().__init__(name, type, **kwargs)
        # print('Scatter object ', self.objectname,' - DP = ', self.objectproperties)

    def _write_Elegant(self):
        wholestring = ""
        etype = "scatter"
        string = self.objectname + ": " + etype
        k1 = self.k1 if self.k1 is not None else 0
        for key, value in list(
            merge_two_dicts(
                {"k1": k1}, merge_two_dicts(self.objectproperties, self.objectdefaults)
            ).items()
        ):
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
