from SimulationFramework.Elements.cavity import cavity
from SimulationFramework.Elements.drift import drift
from SimulationFramework.Framework_objects import elements_Elegant
from SimulationFramework.Modules.Fields import field


class wakefield(cavity):
    """
    Class defining a wakefield element
    """

    coupling_cell_length: float = 0.0
    """Cavity coupling cell length"""

    scale_kick: float = 1.0
    """Wake kick scaling factor"""

    fringe_field_coefficient: float | None = None
    """Fringe field coefficient for cavity"""

    x_offset: float = 0.0
    """Horizontal offset"""

    y_offset: float = 0.0
    """Vertical offset"""

    zcolumn: str = '"z"'
    """String representing the z position in the wake file"""

    wxcolumn: str = '"Wx"'
    """String representing the horizontal wake in the wake file"""

    wycolumn: str = '"Wy"'
    """String representing the vertical wake in the wake file"""

    wzcolumn: str = '"Wz"'
    """String representing the longitudinal wake in the wake file"""

    interpolation_method: int = 2
    """Interpolation method for ASTRA: 0 = rectangular, 1 = triangular, 2 = Gaussian."""

    equal_grid: float = 0.66
    """If 1.0 an equidistant grid is set up, if 0.0 a grid with equal charge per grid cell is
    employed. Values between 1.0 and 0.0 result in intermediate binning based on
    a linear combination of the two methods."""

    smooth: float = 0.25
    """Smoothing parameter for Gaussian interpolation."""

    subbins: int = 10
    """Sub binning parameter."""

    field_definition: str | field | None = None
    """Wakefield definition"""

    waketype: str = "Taylor_Method_F"
    """Type of wakefield, see `ASTRA manual`_
    
    .. _ASTRA manual: https://www.desy.de/~mpyflo/Astra_manual/Astra-Manual_V3.2.pdf"""

    inputfile: str = None
    """Wake file name for setting in ELEGANT."""

    tcolumn: str = '"t"'
    """Time column name"""

    wcolumn: str = '"Wz"'
    """Longitudinal wake column name"""

    scale_field_ex: float = 0.0
    """x-component of the longitudinal direction vector."""

    scale_field_ey: float = 0.0
    """y-component of the longitudinal direction vector."""

    scale_field_ez: float = 1.0
    """z-component of the longitudinal direction vector."""

    scale_field_hx: float = 1.0
    """x-component of the horizontal direction vector."""

    scale_field_hy: float = 0.0
    """y-component of the horizontal direction vector."""

    scale_field_hz: float = 0.0
    """z-component of the horizontal direction vector."""

    cells: int | None = 0
    """Number of cells (if wake originated from a cavity)"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(wakefield, self).__init__(
            *args,
            **kwargs,
        )

    def _write_ASTRA(self, startn: int) -> str:
        """
        Writes the wakefield element string for ASTRA. Each cell in a cavity gets its own &WAKE element.

        Parameters
        ----------
        n: int
            Wake index

        Returns
        -------
        str
            String representation of the element for ASTRA
        """
        field_ref_pos = self.get_field_reference_position()
        field_file_name = self.generate_field_file_name(
            self.field_definition, code="astra"
        )
        efield_def = [
            "Wk_filename",
            {"value": "'" + field_file_name + "'", "default": ""},
        ]
        output = ""
        if self.scale_kick > 0:
            for n in range(startn, startn + self.cells):
                output += self._write_ASTRA_dictionary(
                    dict(
                        [
                            [
                                "Wk_Type",
                                {
                                    "value": '"' + self.waketype + '"',
                                    "default": "'Taylor_Method_F'",
                                },
                            ],
                            efield_def,
                            ["Wk_x", {"value": self.x_offset, "default": 0}],
                            ["Wk_y", {"value": self.y_offset, "default": 0}],
                            [
                                "Wk_z",
                                {
                                    "value": field_ref_pos[2]
                                    + self.coupling_cell_length
                                    + (0.5 + n - 1) * self.cell_length
                                },
                            ],
                            ["Wk_ex", {"value": self.scale_field_ex, "default": 0}],
                            ["Wk_ey", {"value": self.scale_field_ey, "default": 0}],
                            ["Wk_ez", {"value": self.scale_field_ez, "default": 1}],
                            ["Wk_hx", {"value": self.scale_field_hx, "default": 1}],
                            ["Wk_hy", {"value": self.scale_field_hy, "default": 0}],
                            ["Wk_hz", {"value": self.scale_field_hz, "default": 0}],
                            [
                                "Wk_equi_grid",
                                {"value": self.equal_grid, "default": 0.66},
                            ],
                            ["Wk_N_bin", {"value": 10, "default": 100}],
                            [
                                "Wk_ip_method",
                                {"value": self.interpolation_method, "default": 2},
                            ],
                            ["Wk_smooth", {"value": self.smooth, "default": 0.25}],
                            ["Wk_sub", {"value": self.subbins, "default": 10}],
                            [
                                "Wk_scaling",
                                {"value": 1 * self.scale_kick, "default": 1},
                            ],
                        ]
                    ),
                    n,
                )
                output += "\n"
            output += "\n"
        return output

    def set_column_names(self, file_name: str) -> str:
        """
        Set wakefield column names based on the `field_definition`.

        Parameters
        ----------
        file_name: str
            Name of wakefield dile

        Returns
        -------
        str
            Wake type
        """
        self.tcolumn = '"t"'
        if not isinstance(self.field_definition, field):
            self.update_field_definition()
        if self.field_definition.field_type.lower() == "longitudinalwake":
            etype = "wake"
            self.wcolumn = '"Wz"'
            self.inputfile = '"' + file_name + '"'
        elif self.field_definition.field_type.lower() == "transversewake":
            etype = "trwake"
            self.wxcolumn = '"Wx"'
            self.wycolumn = '"Wy"'
            self.inputfile = '"' + file_name + '"'
        elif self.field_definition.field_type.lower() == "3dwake":
            etype = "wake3d"
            self.wxcolumn = '"Wx"'
            self.wycolumn = '"Wy"'
            self.wzcolumn = '"Wz"'
            self.inputfile = '"' + file_name + '"'
        return etype

    def _write_Elegant(self) -> str:
        """
        Writes the wakefield element string for ELEGANT.

        Returns
        -------
        str or None
            String representation of the element for ELEGANT
        """
        wakefield_file_name = self.generate_field_file_name(
            self.field_definition, code="elegant"
        )
        etype = self.set_column_names(wakefield_file_name)
        wholestring = ""
        string = self.objectname + ": " + etype
        if self.length > 0:
            d = drift(
                self.objectname + "-drift", type="drift", **{"length": self.length}
            )
            wholestring += d._write_Elegant()
        for key, value in self.objectproperties.items():
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

    def _write_GPT(self, Brho, ccs="wcs", output_ccs=None, *args, **kwargs):
        field_ref_pos = self.get_field_reference_position()
        field_file_name = self.generate_field_file_name(
            self.field_definition, code="gpt"
        )
        self.set_column_names(field_file_name)
        self.fringe_field_coefficient = (
            self.fringe_field_coefficient
            if self.fringe_field_coefficient is not None
            else 3.0 / self.cell_length
        )
        output = ""
        if self.scale_kick > 0:
            for n in range(self.cells):
                ccs_label, value_text = ccs.ccs_text(
                    [
                        field_ref_pos[0],
                        field_ref_pos[1],
                        field_ref_pos[2]
                        + self.coupling_cell_length
                        + n * self.cell_length,
                    ],
                    self.rotation,
                )
                output += (
                    "wakefield"
                    + "( "
                    + ccs.name
                    + ", "
                    + ccs_label
                    + ", "
                    + value_text
                    + ", "
                    + str(self.cell_length)
                    + ", "
                    + str(self.fringe_field_coefficient)
                    + ', "'
                    + str(field_file_name)
                    + '", '
                    + self.zcolumn
                    + ", "
                    + self.wxcolumn
                    + ", "
                    + self.wycolumn
                    + ", "
                    + self.wzcolumn
                    + ");\n"
                )
        return output
