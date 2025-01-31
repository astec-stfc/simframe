from SimulationFramework.Elements.cavity import cavity, elements_Elegant
from SimulationFramework.Modules.merge_two_dicts import merge_two_dicts


class rf_deflecting_cavity(cavity):

    def __init__(self, name=None, type="rf_deflecting_cavity", **kwargs):
        super().__init__(name, type, **kwargs)
        self.add_default("n_kicks", 10)

    def _write_Elegant(self):
        wholestring = ""
        etype = self._convertType_Elegant(self.objecttype)
        string = self.objectname + ": " + etype
        for key, value in list(
            merge_two_dicts(self.objectproperties, self.objectdefaults).items()
        ):
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
