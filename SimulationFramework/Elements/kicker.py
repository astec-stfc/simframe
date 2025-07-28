from SimulationFramework.Framework_objects import elements_Elegant
from SimulationFramework.Elements.dipole import dipole
import numpy as np
from SimulationFramework.Modules.merge_two_dicts import merge_two_dicts


class kicker(dipole):
    horizontal_kick: float = 0.0
    vertical_kick: float = 0.0

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(kicker, self).__init__(
            *args,
            **kwargs
        )


    def __setattr__(self, name, value):
        # Let Pydantic set known fields normally
        if name in self.model_fields:
            super().__setattr__(name, value)
        else:
            # Store extras in __dict__ (allowed by Config.extra = 'allow')
            self.__dict__[name] = value

    def get_angle(self):
        hkick = self.horizontal_kick if self.horizontal_kick is not None else 0
        vkick = self.vertical_kick if self.vertical_kick is not None else 0
        return np.sqrt(hkick**2 + vkick**2)

    @property
    def z_rot(self):
        hkick = self.horizontal_kick if self.horizontal_kick is not None else 0
        vkick = self.vertical_kick if self.vertical_kick is not None else 0
        return self.global_rotation[0] + np.arctan2(vkick, hkick)

    def _write_ASTRA(self, n, **kwargs):
        output = ""
        output = super()._write_ASTRA(n)
        return output

    def _write_GPT(self, Brho, ccs="wcs", *args, **kwargs):
        return ""

    def gpt_ccs(self, ccs):
        return ccs

    def _write_Elegant(self):
        wholestring = ""
        etype = self._convertType_Elegant(self.objecttype)
        string = self.objectname + ": " + etype
        setattr(self, "k1", self.k1 if self.k1 is not None else 0)
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
                value = 1 if value is True else value
                value = 0 if value is False else value
                tmpstring = ", " + key + " = " + str(value)
                if len(string + tmpstring) > 76:
                    wholestring += string + ",&\n"
                    string = ""
                    string += tmpstring[2::]
                else:
                    string += tmpstring
        wholestring += string + ";\n"
        return wholestring
