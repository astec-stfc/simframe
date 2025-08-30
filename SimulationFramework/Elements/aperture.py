from SimulationFramework.Framework_objects import frameworkElement
from typing import Literal


class aperture(frameworkElement):
    """
    Class defining an aperture or collimator.
    """

    number_of_elements: int | None = None
    """Number of aperture elements"""

    shape: Literal["elliptical", "planar", "circular", "rectangular", "scraper"] = None
    """Aperture shape"""

    horizontal_size: float | None = None
    """Horizontal size of aperture"""

    vertical_size: float | None = None
    """Vertical size of aperture"""

    radius: float | None = None
    """Radius of aperture"""

    negative_extent: float | None = None
    """Longitudinal start position of an aperture"""

    positive_extent: float | None = None
    """Longitudinal end position of an aperture"""

    def __init__(self, *args, **kwargs):
        super(aperture, self).__init__(*args, **kwargs)

    def _write_GPT(self, Brho: float, ccs: str = "wcs", *args, **kwargs) -> str:
        """
        Writes the element string for GPT [currently not in use]

        Parameters
        ----------
        Brho: float
            ?
        ccs: str
            GPT coordinate system

        Returns
        -------
        str
            String representation of the element [currently empty]
        """
        return ""
        # if self.shape == 'elliptical':
        #     output = 'rmax'
        # else:
        #     output = 'xymax'
        # output += '( "wcs", '+self.gpt_coordinates()+', '+str(self.horizontal_size)+', '+str(self.length)+');\n'
        # return output

    def _write_ASTRA_Common(self, dic: dict) -> dict:
        """
        Creates the part of the ASTRA element dictionary common to all apertures in ASTRA

        Parameters
        ----------
        dic: dict
            Dictionary containing the parameters for the aperture

        Returns
        -------
        dict
            ASTRA dictionary with parameters and values
        """
        if self.negative_extent is not None:
            dic["Ap_Z1"] = {"value": self.negative_extent, "default": 0}
            dic["a_pos"] = {"value": self.start[2]}
        else:
            dic["Ap_Z1"] = {"value": self.start[2] + self.dz, "default": 0}
        if self.positive_extent is not None:
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
            "value": self.x_rot + self.dx_rot,
            "default": 0,
            "type": "not_zero",
        }
        dic["A_yrot"] = {
            "value": self.y_rot + self.dy_rot,
            "default": 0,
            "type": "not_zero",
        }
        dic["A_zrot"] = {
            "value": self.z_rot + self.dz_rot,
            "default": 0,
            "type": "not_zero",
        }
        return dic

    def _write_ASTRA_Circular(self) -> dict:
        """
        Creates the part of the ASTRA element dictionary relevant to circular apertures in ASTRA

        Parameters
        ----------
        dic: dict
            Dictionary containing the parameters for the aperture

        Returns
        -------
        dict
            ASTRA dictionary with parameters and values
        """
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

    def _write_ASTRA_Planar(self, plane, width) -> dict:
        """
        Creates the part of the ASTRA element dictionary common to all apertures in ASTRA

        Parameters
        ----------
        dic: dict
            Dictionary containing the parameters for the aperture

        Returns
        -------
        dict
            ASTRA dictionary with parameters and values
        """
        dic = dict()
        dic["File_Aperture"] = {"value": plane}
        dic["Ap_R"] = {"value": width}
        return self._write_ASTRA_Common(dic)

    def _write_ASTRA(self, n: int, **kwargs) -> str:
        """
        Writes the aperture element string for ASTRA

        Parameters
        ----------
        n: int
            Element index number
        **kwargs: dict
            Keyword args

        Returns
        -------
        str
            String representation of the element for ASTRA

        Raises:
        -------
        ValueError
            If `shape` is not in the list of allowed values.
        """
        self.number_of_elements = 0
        if self.shape in ["elliptical", "circular"]:
            self.number_of_elements += 1
            dic = self._write_ASTRA_Circular()
            return self._write_ASTRA_dictionary(dic, n)
        elif self.shape in ["planar", "rectangular"]:
            text = ""
            if self.horizontal_size is not None and self.horizontal_size > 0:
                dic = self._write_ASTRA_Planar("Col_X", 1e3 * self.horizontal_size)
                text += self._write_ASTRA_dictionary(dic, n)
                self.number_of_elements += 1
            if self.vertical_size is not None and self.vertical_size > 0:
                dic = self._write_ASTRA_Planar("Col_Y", 1e3 * self.vertical_size)
                if self.number_of_elements > 0:
                    self.number_of_elements += 1
                    n = n + 1
                    text += "\n"
                text += self._write_ASTRA_dictionary(dic, n)
            return text
        elif self.shape == "scraper":
            text = ""
            if self.horizontal_size is not None and self.horizontal_size > 0:
                dic = self._write_ASTRA_Planar("Scr_X", 1e3 * self.horizontal_size)
                text += self._write_ASTRA_dictionary(dic, n)
                self.number_of_elements += 1
            if self.vertical_size is not None and self.vertical_size > 0:
                dic = self._write_ASTRA_Planar("Scr_Y", 1e3 * self.vertical_size)
                if self.number_of_elements > 0:
                    self.number_of_elements += 1
                    n = n + 1
                    text += "\n"
                text += self._write_ASTRA_dictionary(dic, n)
            return text
        else:
            raise ValueError(
                "shape must be in ['elliptical', 'planar', 'circular', 'rectangular', 'scraper']"
            )
