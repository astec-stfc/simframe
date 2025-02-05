from SimulationFramework.Framework_objects import frameworkElement


class aperture(frameworkElement):

    def __init__(self, name=None, type="aperture", **kwargs):
        super().__init__(name, type, **kwargs)
        self.number_of_elements = 1

    def _write_GPT(self, Brho, ccs="wcs", *args, **kwargs):
        return ""
        # if self.shape == 'elliptical':
        #     output = 'rmax'
        # else:
        #     output = 'xymax'
        # output += '( "wcs", '+self.gpt_coordinates()+', '+str(self.horizontal_size)+', '+str(self.length)+');\n'
        # return output

    def _write_ASTRA_Common(self, dic):
        if hasattr(self, "negative_extent") and self.negative_extent is not None:
            dic["Ap_Z1"] = {"value": self.negative_extent, "default": 0}
            dic["a_pos"] = {"value": self.start[2]}
        else:
            dic["Ap_Z1"] = {"value": self.start[2] + self.dz, "default": 0}
        if hasattr(self, "positive_extent") and self.positive_extent is not None:
            dic["Ap_Z2"] = {"value": self.positive_extent, "default": 0}
            dic["a_pos"] = {"value": self.start[2]}
        else:
            end = (
                self.end[2] + self.dz
                if self.end[2] >= (self.start[2] + 1e-3)
                else self.start[2] + self.dz + 1e-3
            )
            dic["Ap_Z2"] = {"value": end, "default": 0}
        dic["A_xrot"] = {
            "value": self.y_rot + self.dy_rot,
            "default": 0,
            "type": "not_zero",
        }
        dic["A_yrot"] = {
            "value": self.x_rot + self.dx_rot,
            "default": 0,
            "type": "not_zero",
        }
        dic["A_zrot"] = {
            "value": self.z_rot + self.dz_rot,
            "default": 0,
            "type": "not_zero",
        }
        return dic

    def _write_ASTRA_Circular(self, n):
        dic = dict()
        dic["File_Aperture"] = {"value": "RAD"}
        if self.radius is not None:
            radius = self.radius
        elif self.horizontal_size > 0 and self.vertical_size > 0:
            radius = min([self.horizontal_size, self.vertical_size])
        elif self.horizontal_size > 0:
            radius = self.horizontal_size
        elif self.vertical_size > 0:
            radius = self.vertical_size
        else:
            radius = 1
        dic["Ap_R"] = {"value": 1e3 * radius}
        return self._write_ASTRA_Common(dic)

    def _write_ASTRA_Planar(self, n, plane, width):
        dic = dict()
        dic["File_Aperture"] = {"value": plane}
        dic["Ap_R"] = {"value": width}
        return self._write_ASTRA_Common(dic)

    def _write_ASTRA(self, n: int, **kwargs) -> str:
        self.number_of_elements = 0
        if self.shape == "elliptical" or self.shape == "circular":
            self.number_of_elements += 1
            dic = self._write_ASTRA_Circular(n)
            return self._write_ASTRA_dictionary(dic, n)
        elif self.shape == "planar" or self.shape == "rectangular":
            text = ""
            if self.horizontal_size is not None and self.horizontal_size > 0:
                dic = self._write_ASTRA_Planar(n, "Col_X", 1e3 * self.horizontal_size)
                text += self._write_ASTRA_dictionary(dic, n)
                self.number_of_elements += 1
            if self.vertical_size is not None and self.vertical_size > 0:
                dic = self._write_ASTRA_Planar(n, "Col_Y", 1e3 * self.vertical_size)
                if self.number_of_elements > 0:
                    self.number_of_elements += 1
                    n = n + 1
                    text += "\n"
                text += self._write_ASTRA_dictionary(dic, n)
            return text
        elif self.shape == "scraper":
            text = ""
            if self.horizontal_size is not None and self.horizontal_size > 0:
                dic = self._write_ASTRA_Planar(n, "Scr_X", 1e3 * self.horizontal_size)
                text += self._write_ASTRA_dictionary(dic, n)
                self.number_of_elements += 1
            if self.vertical_size is not None and self.vertical_size > 0:
                dic = self._write_ASTRA_Planar(n, "Scr_Y", 1e3 * self.vertical_size)
                if self.number_of_elements > 0:
                    self.number_of_elements += 1
                    n = n + 1
                    text += "\n"
                text += self._write_ASTRA_dictionary(dic, n)
            return text
