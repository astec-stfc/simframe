from SimulationFramework.Framework_objects import frameworkElement, elements_Elegant
from SimulationFramework.Modules.merge_two_dicts import merge_two_dicts


class wiggler(frameworkElement):

    def __init__(self, name=None, type="wiggler", **kwargs):
        super().__init__(name, type, **kwargs)
        # self.add_default('k1l', 0)
        # self.add_default('n_steps', 1*self.periods)

    def write_ASTRA(self, n, **kwargs):
        return self._write_ASTRA(
            dict(
                [
                    ["Q_pos", {"value": self.middle[2] + self.dz, "default": 0}],
                ]
            ),
            n,
        )

    def _write_Elegant(self):
        wholestring = ""
        if (
            ("k" in self and abs(self.k) > 0)
            or ("peak_field" in self and abs(self.peak_field) > 0)
            or ("radius" in self and abs(self.radius) > 0)
        ):
            etype = self._convertType_Elegant(self.objecttype)
        else:
            etype = "drift"
        string = self.objectname + ": " + etype
        for key, value in list(
            merge_two_dicts(self.objectproperties, self.objectdefaults).items()
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
