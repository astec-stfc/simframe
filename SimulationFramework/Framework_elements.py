import os
from operator import add
from copy import copy
import numpy as np
from munch import Munch
from .Framework_objects import frameworkElement, elements_Elegant, csrdrift
from .FrameworkHelperFunctions import (
    checkValue,
    chop,
    _rotation_matrix,
    expand_substitution,
)
from .Modules.merge_two_dicts import merge_two_dicts
from .Modules import Beams as rbf


class dipole(frameworkElement):

    def __init__(self, name=None, type="dipole", **kwargs):
        super().__init__(name, type, **kwargs)
        self.add_default("csr_bins", 100)
        self.add_default("deltaL", 0)
        self.add_default("csr_enable", 1)
        self.add_default("isr_enable", True)
        self.add_default("n_kicks", 8)
        self.add_default("sr_enable", True)
        self.add_default("integration_order", 4)
        self.add_default("nonlinear", 1)
        self.add_default("smoothing_half_width", 1)
        self.add_default("edge_order", 2)
        self.add_default("edge1_effects", 1)
        self.add_default("edge2_effects", 1)

    # @property
    # def middle(self):
    #     start = self.position_start
    #     length_vector = self.rotated_position([0,0, self.length / 2.0], offset=[0,0,0], theta=self.theta)
    #     starting_middle = length_vector
    #     # print(self.objectname, self.theta, self.starting_rotation, self.rotated_position(starting_middle, offset=self.starting_offset, theta=self.starting_rotation)[0])
    #     return np.array(start) + self.rotated_position(starting_middle, offset=self.starting_offset, theta=self.starting_rotation)

    # @property
    # def middle(self):
    #     sx, sy, sz = self.position_start
    #     angle = -self.angle
    #     l = self.length
    #     if abs(angle) > 0:
    #         cx = 0
    #         cy = 0
    #         cz = (l * np.tan(angle/2.0)) / angle
    #         vec = [cx, cy, cz]
    #     else:
    #         vec = [0,0,l/2.0]
    #     # print (vec)
    #     return np.array(self.position_start) + self.rotated_position(np.array(vec), offset=self.starting_offset, theta=self.y_rot)

    @property
    def arc_middle(self):
        sx, sy, sz = self.position_start
        angle = -self.angle
        len = self.length
        r = len / angle
        if abs(angle) > 0:
            cx = r * (np.cos(angle / 2.0) - 1)
            cy = 0
            cz = r * np.sin(angle / 2.0)
            vec = [cx, cy, cz]
        else:
            vec = [0, 0, len / 2.0]
        # print (vec)
        return np.array(self.position_start) + self.rotated_position(
            np.array(vec), offset=self.starting_offset, theta=self.y_rot
        )

    @property
    def line_middle(self):
        sx, sy, sz = self.position_start
        angle = -self.angle
        len = self.length
        r = len / angle
        if abs(angle) > 0:
            cx = 0.5 * r * (np.cos(angle) - 1)
            cy = 0
            cz = 0.5 * r * np.sin(angle)
            vec = [cx, cy, cz]
        else:
            vec = [0, 0, len / 2.0]
        # print (vec)
        return np.array(self.position_start) + self.rotated_position(
            np.array(vec), offset=self.starting_offset, theta=self.y_rot
        )

    @property
    def TD_middle(self):
        sx, sy, sz = self.position_start
        angle = -self.angle
        len = self.length
        r = len / angle
        if abs(angle) > 0:
            cx = 0.25 * r * (2.0 * np.cos(angle / 2.0) + np.cos(angle) - 3)
            cy = 0
            cz = 0.25 * r * (2 * np.sin(angle / 2.0) + np.sin(angle))
            vec = [cx, cy, cz]
        else:
            vec = [0, 0, len / 2.0]
        # print (vec)
        return np.array(self.position_start) + self.rotated_position(
            np.array(vec), offset=self.starting_offset, theta=self.y_rot
        )

    @property
    def intersection(self):
        sx, sy, sz = self.position_start
        angle = -self.angle
        len = self.length
        if abs(angle) > 0:
            cx = 0
            cy = 0
            cz = len * np.tan(0.5 * angle) / angle
            vec = [cx, cy, cz]
        else:
            vec = [0, 0, len / 2.0]
        return np.array(self.position_start) + self.rotated_position(
            np.array(vec), offset=self.starting_offset, theta=self.y_rot
        )

    @property
    def position_start(self):
        middle = self.centre
        angle = -self.angle
        len = self.length
        if abs(angle) > 0:
            cx = 0
            cy = 0
            cz = -len * np.tan(0.5 * angle) / angle
            vec = [cx, cy, cz]
        else:
            vec = [0, 0, -len / 2.0]
        # print(self.objectname, 'start', np.array(middle) + self.rotated_position(np.array(vec), offset=self.starting_offset, theta=self.y_rot))
        return np.array(middle) + self.rotated_position(
            np.array(vec), offset=self.starting_offset, theta=self.y_rot
        )

    @property
    def position_end(self):
        start = self.position_start
        angle = -self.angle
        if abs(angle) > 1e-9:
            ex = (self.length * (1 - np.cos(angle))) / angle
            ey = 0
            ez = (self.length * (np.sin(angle))) / angle
            vec = [ex, ey, ez]
        else:
            vec = [0, 0, self.length]
        # print(self.objectname, 'start', start, 'end', np.array(start) + self.rotated_position(np.array(vec), offset=self.starting_offset, theta=self.y_rot))
        return np.array(start) + self.rotated_position(
            np.array(vec), offset=self.starting_offset, theta=self.y_rot
        )

    @property
    def astra_end(self):
        angle = -self.angle
        if abs(self.angle) > 1e-9:
            ex = -1 * (self.length * (np.cos(angle) - 1)) / angle
            ey = 0
            ez = (self.length * (np.sin(angle))) / angle
            return np.array(self.position_start) + self.rotated_position(
                np.array([ex, ey, ez]), offset=self.starting_offset, theta=0
            )
        else:
            return np.array(self.position_start) + self.rotated_position(
                np.array([0, 0, self.length]), offset=self.starting_offset, theta=0
            )

    @property
    def width(self):
        if "width" in self.objectproperties:
            return self.objectproperties["width"]
        else:
            return 0.2

    @width.setter
    def width(self, w):
        self.objectproperties["width"] = w

    def __neg__(self):
        newself = copy.deepcopy(self)
        if (
            "exit_edge_angle" in newself.objectproperties
            and "entrance_edge_angle" in newself.objectproperties
        ):
            e1 = newself["entrance_edge_angle"]
            e2 = newself["exit_edge_angle"]
            newself.objectproperties["entrance_edge_angle"] = e2
            newself.objectproperties["exit_edge_angle"] = e1
        elif "entrance_edge_angle" in newself.objectproperties:
            newself.objectproperties["exit_edge_angle"] = newself.objectproperties[
                "entrance_edge_angle"
            ]
            del newself.objectproperties["entrance_edge_angle"]
        elif "exit_edge_angle" in newself.objectproperties:
            newself.objectproperties["entrance_edge_angle"] = newself.objectproperties[
                "exit_edge_angle"
            ]
            del newself.objectproperties["exit_edge_angle"]
        newself.objectname = "-" + newself.objectname
        return newself

    def check_value(self, estr, default=0):
        if estr in self.objectproperties:
            if isinstance(self.objectproperties[estr], str):
                return checkValue(self, self.objectproperties[estr], default)
            else:
                return self.objectproperties[estr]
        else:
            return default

    @property
    def intersect(self):
        return self.length * np.tan(0.5 * self.angle) / self.angle

    @property
    def rho(self):
        return (
            self.length / self.angle
            if self.length is not None and abs(self.angle) > 1e-9
            else 0
        )

    @property
    def e1(self):
        return self.check_value("entrance_edge_angle")

    @property
    def e2(self):
        return self.check_value("exit_edge_angle")

    def _write_Elegant(self):
        wholestring = ""
        # etype = self._convertType_Elegant(self.objecttype)
        etype = "csrcsbend" if self.csr_enable or self.csr_enable > 0 else "csbend"
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
                # if 'bins' in key or 'bins' in self._convertKeyword_Elegant(key):
                # print('BINS KEY ', key, '  ', self._convertKeyword_Elegant(key))
                if "edge_angle" in key:
                    key = self._convertKeyword_Elegant(key)
                    value = (
                        getattr(self, key)
                        if hasattr(self, key) and getattr(self, key) is not None
                        else value
                    )
                else:
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

    @property
    def corners(self):
        corners = [0, 0, 0, 0]
        if hasattr(self, "global_rotation") and self.global_rotation is not None:
            rotation = (
                self.global_rotation[2]
                if len(self.global_rotation) == 3
                else self.global_rotation
            )
        else:
            rotation = 0
        theta = self.e1 + rotation
        corners[0] = np.array(
            list(
                map(
                    add,
                    np.transpose(self.position_start),
                    np.dot([-self.width * self.length, 0, 0], _rotation_matrix(theta)),
                )
            )
        )
        corners[3] = np.array(
            list(
                map(
                    add,
                    np.transpose(self.position_start),
                    np.dot([self.width * self.length, 0, 0], _rotation_matrix(theta)),
                )
            )
        )
        theta = self.angle - self.e2 + rotation
        corners[1] = np.array(
            list(
                map(
                    add,
                    np.transpose(self.end),
                    np.dot([-self.width * self.length, 0, 0], _rotation_matrix(theta)),
                )
            )
        )
        corners[2] = np.array(
            list(
                map(
                    add,
                    np.transpose(self.end),
                    np.dot([self.width * self.length, 0, 0], _rotation_matrix(theta)),
                )
            )
        )
        # print(self.objectname, self.position_start, self.end)
        # print('rotation = ', rotation)
        # corners = [self.rotated_position(x, offset=self.starting_offset, theta=rotation) for x in corners]
        return corners

    def write_CSRTrack(self, n):
        z1 = self.position_start[2]
        z2 = self.position_end[2]
        return (
            """dipole{\nposition{rho="""
            + str(z1)
            + """, psi="""
            + str(chop(self.theta + self.e1))
            + """, marker=d"""
            + str(n)
            + """a}\nproperties{r="""
            + str(self.rho)
            + """}\nposition{rho="""
            + str(z2)
            + """, psi="""
            + str(chop(self.theta + self.angle - self.e2))
            + """, marker=d"""
            + str(n)
            + """b}\n}\n"""
        )

    def write_ASTRA(self, n, **kwargs):
        # print('self.start = ', self.position_start)
        # print('self.end = ', self.position_end)
        # print('self.rotation = ', self.global_rotation[2])
        # print('self.astra_end = ', self.astra_end)
        if abs(checkValue(self, "strength", default=0)) > 0 or abs(self.rho) > 0:
            corners = self.corners
            if self.plane is None:
                self.plane = "horizontal"
            params = dict(
                [
                    [
                        "D_Type",
                        {"value": "'" + self.plane + "'", "default": "'horizontal'"},
                    ],
                    [
                        "D_Gap",
                        {
                            "type": "list",
                            "value": [self.gap, self.gap],
                            "default": [0.0001, 0.0001],
                        },
                    ],
                    ["D1", {"type": "array", "value": [corners[3][0], corners[3][2]]}],
                    ["D3", {"type": "array", "value": [corners[2][0], corners[2][2]]}],
                    ["D4", {"type": "array", "value": [corners[1][0], corners[1][2]]}],
                    ["D2", {"type": "array", "value": [corners[0][0], corners[0][2]]}],
                    # ['D_xoff', {'value': self.start[0] + self.dx, 'default': 0}],
                    # ['D_yoff', {'value': self.start[1] + self.dy, 'default': 0}],
                    # ['D_zoff', {'value': self.dz, 'default': 0}],
                    # ['D_xrot', {'value': self.y_rot + self.dy_rot, 'default': 0}],
                    # ['D_yrot', {'value': self.x_rot + self.dx_rot, 'default': 0}],
                    ["D_zrot", {"value": self.z_rot + self.dz_rot, "default": 0}],
                ]
            )
            if (
                abs(checkValue(self, "strength", default=0)) > 0
                or not abs(self.rho) > 0
            ):
                params["D_strength"] = {
                    "value": checkValue(self, "strength", 0),
                    "default": 1e6,
                }
            else:
                params["D_radius"] = {"value": 1 * self.rho, "default": 1e6}
            return self._write_ASTRA(params, n)
        else:
            return None

    def gpt_coordinates(self, position, rotation):
        angle = -1 * self.angle
        x, y, z = chop(position, 1e-6)
        psi, phi, theta = rotation
        output = ""
        for c in [0, 0, z]:
            output += str(c) + ", "
        output += "cos(" + str(angle) + "), 0, -sin(" + str(angle) + "), 0, 1 ,0"
        return output

    def write_GPT(self, Brho, ccs, *args, **kwargs):
        field = 1.0 * self.angle * Brho / self.length
        if abs(field) > 0 and abs(self.rho) < 100:
            relpos, relrot = ccs.relative_position(self.middle, self.global_rotation)
            coord = self.gpt_coordinates(relpos, relrot)
            new_ccs = self.gpt_ccs(ccs).name
            b1 = np.round(
                (
                    1.0
                    / (
                        2
                        * self.check_value("half_gap", default=0.016)
                        * self.check_value("edge_field_integral", default=0.5)
                    )
                    if abs(self.check_value("half_gap", default=0.016)) > 0
                    else 10000
                ),
                2,
            )
            dl = 0 if self.deltaL is None else self.deltaL
            e1 = self.e1 if self.angle >= 0 else -1 * self.e1
            e2 = self.e2 if self.angle >= 0 else -1 * self.e2
            # print(self.objectname, ' - deltaL = ', dl)
            # b1 = 0.
            """
            ccs( "wcs", 0, 0, startofdipole +  intersect1, Cos(theta), 0, -Sin(theta), 0, 1, 0, "bend1" ) ;
            sectormagnet( "wcs", "bend1", rho, field, e1, e2, 0., 100., 0 ) ;
            """
            output = "ccs( " + ccs.name + ", " + coord + ", " + new_ccs + ");\n"
            output += (
                "sectormagnet( "
                + ccs.name
                + ", "
                + new_ccs
                + ", "
                + str(abs(self.rho))
                + ", "
                + str(abs(field))
                + ", "
                + str(e1)
                + ", "
                + str(e2)
                + ", "
                + str(dl)
                + ", "
                + str(b1)
                + ", 0);\n"
            )
        else:
            output = ""
        return output

    def gpt_ccs(self, ccs):
        if abs(self.angle) > 0 and abs(self.rho) < 100:
            # print('Creating new CCS')
            number = (
                str(int(ccs._name.split("_")[1]) + 1) if ccs._name != "wcs" else "1"
            )
            name = "ccs_" + number if ccs._name != "wcs" else "ccs_1"
            # print('middle position = ', self.start, self.middle)
            return gpt_ccs(
                name,
                self.middle,
                self.global_rotation + np.array([0, 0, -self.angle]),
                0 * abs(self.intersect),
            )
        else:
            return ccs


class kicker(dipole):

    def __init__(self, name=None, type="kicker", **kwargs):
        super().__init__(name, type, **kwargs)

    @property
    def angle(self):
        hkick = self.horizontal_kick if self.horizontal_kick is not None else 0
        vkick = self.vertical_kick if self.vertical_kick is not None else 0
        return np.sqrt(hkick**2 + vkick**2)

    @property
    def z_rot(self):
        hkick = self.horizontal_kick if self.horizontal_kick is not None else 0
        vkick = self.vertical_kick if self.vertical_kick is not None else 0
        return self.global_rotation[0] + np.arctan2(vkick, hkick)

    def write_ASTRA(self, n, **kwargs):
        output = ""
        output = super().write_ASTRA(n)
        return output

    def write_GPT(self, Brho, ccs="wcs", *args, **kwargs):
        return ""

    def gpt_ccs(self, ccs):
        return ccs

    def _write_Elegant(self):
        wholestring = ""
        etype = self._convertType_Elegant(self.objecttype)
        string = self.objectname + ": " + etype
        k1 = self.k1 if self.k1 is not None else 0
        k2 = self.k2 if self.k2 is not None else 0
        keydict = merge_two_dicts(
            {"k1": k1, "k2": k2},
            merge_two_dicts(self.objectproperties, self.objectdefaults),
        )
        for key, value in keydict.items():
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


class quadrupole(frameworkElement):

    def __init__(self, name=None, type="quadrupole", **kwargs):
        super().__init__(name, type, **kwargs)
        self.add_default("k1l", 0)
        self.add_default("n_kicks", 4)
        self.strength_errors = [0]

    @property
    def k1(self) -> float:
        """Return the quadrupole K1 value in m^-2"""
        return float(self.k1l) / float(self.length) if self.length > 0 else 0

    @k1.setter
    def k1(self, k1):
        self.k1l = self.length * k1

    @property
    def dk1(self):
        return self.strength_errors[0]

    @dk1.setter
    def dk1(self, dk1):
        self.strength_errors[0] = dk1

    def update_field_definition(self) -> None:
        """Updates the field definitions to allow for the relative sub-directory location"""
        if hasattr(self, "field_definition") and self.field_definition is not None:
            self.field_definition = expand_substitution(self, self.field_definition)

    def write_ASTRA(self, n: int, **kwargs) -> str:
        astradict = dict(
            [
                ["Q_pos", {"value": self.middle[2] + self.dz, "default": 0}],
                ["Q_xoff", {"value": self.middle[0], "default": 0, "type": "not_zero"}],
                [
                    "Q_yoff",
                    {
                        "value": self.middle[1] + self.dy,
                        "default": None,
                        "type": "not_zero",
                    },
                ],
                [
                    "Q_xrot",
                    {
                        "value": -1 * self.y_rot + self.dy_rot,
                        "default": None,
                        "type": "not_zero",
                    },
                ],
                [
                    "Q_yrot",
                    {
                        "value": -1 * self.x_rot + self.dx_rot,
                        "default": None,
                        "type": "not_zero",
                    },
                ],
                [
                    "Q_zrot",
                    {
                        "value": -1 * self.z_rot + self.dz_rot,
                        "default": None,
                        "type": "not_zero",
                    },
                ],
                ["Q_smooth", {"value": self.smooth, "default": None}],
                ["Q_bore", {"value": self.bore, "default": None, "type": "not_zero"}],
                ["Q_noscale", {"value": self.scale_field}],
                ["Q_mult_a", {"type": "list", "value": self.multipoles}],
            ]
        )
        if self.field_definition:
            self.generate_field_file_name(self.field_definition)
            astradict.update(
                dict(
                    [
                        ["Q_type", {"value": self.field_definition, "default": None}],
                        ["q_grad", {"value": self.gradient, "default": None}],
                    ]
                )
            )
        elif abs(self.k1 + self.dk1) > 0:
            astradict.update(
                dict(
                    [
                        ["Q_k", {"value": self.k1 + self.dk1, "default": 0}],
                        ["Q_length", {"value": self.length, "default": 0}],
                    ]
                )
            )
        if abs(self.k1 + self.dk1) > 0 or self.field_definition:
            return self._write_ASTRA(astradict, n)
        else:
            return None

    def write_GPT(self, Brho, ccs="wcs", *args, **kwargs):
        ccs_label, value_text = ccs.ccs_text(self.middle, self.rotation)
        output = (
            str(self.objecttype)
            + "( "
            + ccs.name
            + ", "
            + ccs_label
            + ", "
            + value_text
            + ", "
            + str(self.length)
            + ", "
            + str((-Brho * self.k1) if not self.gradient else -1 * self.gradient)
            + (
                ", " + str(self.fringe_field_coefficient)
                if self.fringe_field_coefficient > 0
                else ""
            )
            + ");\n"
        )
        return output


class sextupole(frameworkElement):

    def __init__(self, name=None, type="sextupole", **kwargs):
        super().__init__(name, type, **kwargs)
        self.add_default("k2l", 0)
        self.add_default("n_kicks", 20)
        self.strength_errors = [0]

    @property
    def k2(self):
        return self.k2l / self.length

    @k2.setter
    def k2(self, k2):
        self.k2l = self.length * k2

    @property
    def dk2(self):
        return self.strength_errors[0]

    @dk2.setter
    def dk2(self, dk2):
        self.strength_errors[0] = dk2

    def write_GPT(self, Brho, ccs="wcs", *args, **kwargs):
        ccs_label, value_text = ccs.ccs_text(self.middle, self.rotation)
        output = (
            str(self.objecttype)
            + "( "
            + ccs.name
            + ", "
            + ccs_label
            + ", "
            + value_text
            + ", "
            + str(self.length)
            + ", "
            + str(-Brho * self.k2)
            + ");\n"
        )
        return output


class cavity(frameworkElement):

    def __init__(self, name=None, type="cavity", **kwargs):
        super().__init__(name, type, **kwargs)
        self.add_default("tcolumn", '"t"')
        self.add_default("zcolumn", '"Z"')
        self.add_default("ezcolumn", '"Ez"')
        self.add_default("wzcolumn", '"W"')
        self.add_default("wxcolumn", '"W"')
        self.add_default("wycolumn", '"W"')
        self.add_default("wcolumn", '"Ez"')
        self.add_default("change_p0", 1)
        self.add_default("n_kicks", self.n_cells)
        # self.add_default('method', '"non-adaptive runge-kutta"')
        self.add_default("end1_focus", 1)
        self.add_default("end2_focus", 1)
        self.add_default("body_focus_model", "SRS")
        self.add_default("lsc_bins", 100)
        self.add_default("current_bins", 0)
        self.add_default("interpolate_current_bins", 1)
        self.add_default("smooth_current_bins", 1)
        self.add_default("coupling_cell_length", 0)
        self.add_default("field_amplitude", 0)
        self.add_default("crest", 0)

    def update_field_definition(self) -> None:
        """Updates the field definitions to allow for the relative sub-directory location"""
        if hasattr(self, "field_definition") and self.field_definition is not None:
            self.field_definition = expand_substitution(self, self.field_definition)
        if (
            hasattr(self, "field_definition_sdds")
            and self.field_definition_sdds is not None
        ):
            self.field_definition_sdds = expand_substitution(
                self, self.field_definition_sdds
            )
        if (
            hasattr(self, "field_definition_gdf")
            and self.field_definition_gdf is not None
        ):
            self.field_definition_gdf = expand_substitution(
                self, self.field_definition_gdf
            )
        if (
            hasattr(self, "longitudinal_wakefield_sdds")
            and self.longitudinal_wakefield_sdds is not None
        ):
            self.longitudinal_wakefield_sdds = expand_substitution(
                self, self.longitudinal_wakefield_sdds
            )
        if (
            hasattr(self, "transverse_wakefield_sdds")
            and self.transverse_wakefield_sdds is not None
        ):
            self.transverse_wakefield_sdds = expand_substitution(
                self, self.transverse_wakefield_sdds
            )

    @property
    def cells(self):
        if (self.n_cells == 0 or self.n_cells is None) and self.cell_length > 0:
            cells = round((self.length - self.cell_length) / self.cell_length)
            cells = int(cells - (cells % 3))
        elif (
            self.n_cells > 0 and (self.cell_length is not None and self.cell_length) > 0
        ):
            if self.cell_length == self.length:
                cells = 1
            else:
                cells = int(self.n_cells - (self.n_cells % 3))
        else:
            cells = None
        return cells

    def write_ASTRA(self, n, **kwargs):
        auto_phase = kwargs["auto_phase"] if "auto_phase" in kwargs else True
        crest = self.crest if not auto_phase else 0
        basename = self.generate_field_file_name(self.field_definition)
        efield_def = ["FILE_EFieLD", {"value": "'" + basename + "'", "default": ""}]
        return self._write_ASTRA(
            dict(
                [
                    ["C_pos", {"value": self.start[2] + self.dz, "default": 0}],
                    efield_def,
                    ["C_numb", {"value": self.cells}],
                    ["Nue", {"value": float(self.frequency) / 1e9, "default": 2998.5}],
                    [
                        "MaxE",
                        {"value": float(self.field_amplitude) / 1e6, "default": 0},
                    ],
                    ["Phi", {"value": crest - self.phase, "default": 0.0}],
                    ["C_smooth", {"value": self.smooth, "default": None}],
                    [
                        "C_xoff",
                        {
                            "value": self.start[0] + self.dx,
                            "default": None,
                            "type": "not_zero",
                        },
                    ],
                    [
                        "C_yoff",
                        {
                            "value": self.start[1] + self.dy,
                            "default": None,
                            "type": "not_zero",
                        },
                    ],
                    [
                        "C_xrot",
                        {
                            "value": self.y_rot + self.dy_rot,
                            "default": None,
                            "type": "not_zero",
                        },
                    ],
                    [
                        "C_yrot",
                        {
                            "value": self.x_rot + self.dx_rot,
                            "default": None,
                            "type": "not_zero",
                        },
                    ],
                    [
                        "C_zrot",
                        {
                            "value": self.z_rot + self.dz_rot,
                            "default": None,
                            "type": "not_zero",
                        },
                    ],
                ]
            ),
            n,
        )

    def _write_Elegant(self):
        self.update_field_definition()
        original_field_definition_sdds = self.field_definition_sdds
        if self.field_definition_sdds is not None:
            self.field_definition_sdds = (
                '"' + self.generate_field_file_name(self.field_definition_sdds) + '"'
            )
        original_longitudinal_wakefield_sdds = self.longitudinal_wakefield_sdds
        if self.longitudinal_wakefield_sdds is not None:
            self.longitudinal_wakefield_sdds = (
                '"'
                + self.generate_field_file_name(self.longitudinal_wakefield_sdds)
                + '"'
            )
        original_transverse_wakefield_sdds = self.transverse_wakefield_sdds
        if self.transverse_wakefield_sdds is not None:
            self.transverse_wakefield_sdds = (
                '"'
                + self.generate_field_file_name(self.transverse_wakefield_sdds)
                + '"'
            )
        wholestring = ""
        etype = self._convertType_Elegant(self.objecttype)
        if (
            not hasattr(self, "longitudinal_wakefield_sdds")
            or self.longitudinal_wakefield_sdds is None
        ) and (
            not hasattr(self, "transverse_wakefield_sdds")
            or self.transverse_wakefield_sdds is None
        ):
            # print('cavity ', self.objectname, ' is an RFCA!')
            etype = "rfca"
        if self.field_definition_sdds is not None:
            etype = "rftmez0"
            self.ez_peak = self.field_amplitude
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
                key = self._convertKeyword_Elegant(key).lower()
                if etype == "rftmez0" and key == "freq":
                    key = "frequency"
                if (
                    self.objecttype == "cavity"
                    or self.objecttype == "rf_deflecting_cavity"
                ):
                    if etype == "rftmez0":
                        # If using rftmez0 or similar
                        value = (
                            ((value) / 360.0) * (2 * 3.14159)
                            if key == "phase"
                            else value
                        )
                    else:
                        # In ELEGANT all phases are +90degrees!!
                        value = 90 - value if key == "phase" else value
                    # In ELEGANT the voltages need to be compensated
                    value = (
                        abs(
                            (self.cells + 4.1)
                            * self.cell_length
                            * (1 / np.sqrt(2))
                            * value
                        )
                        if key == "volt"
                        else value
                    )
                    # If using rftmez0 or similar
                    value = (
                        abs(1e-3 / (np.sqrt(2)) * value) if key == "ez_peak" else value
                    )
                    # In CAVITY NKICK = n_cells
                    value = 3 * self.cells if key == "n_kicks" else value
                    if key == "n_bins" and value > 0:
                        print(
                            "WARNING: Cavity n_bins is not zero - check log file to ensure correct behaviour!"
                        )
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
        self.field_definition_sdds = original_field_definition_sdds
        self.longitudinal_wakefield_sdds = original_longitudinal_wakefield_sdds
        self.transverse_wakefield_sdds = original_transverse_wakefield_sdds
        return wholestring

    def write_GPT(self, Brho, ccs="wcs", *args, **kwargs):
        self.update_field_definition()
        ccs_label, value_text = ccs.ccs_text(self.middle, self.rotation)
        relpos, relrot = ccs.relative_position(self.middle, self.global_rotation)
        """
        map1D_TM("wcs","z",linacposition,"mockup2m.gdf","Z","Ez",ffacl,phil,w);
        wakefield("wcs","z",  6.78904 + 4.06667 / 2, 4.06667, 50, "Sz5um10mm.gdf", "z","","","Wz", "FieldFactorWz", 10 * 122 / 4.06667) ;
        """
        if self.crest is None:
            self.crest = 0
        subname = str(relpos[2]).replace(".", "")
        if expand_substitution(self, self.field_definition_gdf) is not None:
            output = (
                "f"
                + subname
                + " = "
                + str(self.frequency)
                + ";\n"
                + "w"
                + subname
                + " = 2*pi*f"
                + subname
                + ";\n"
                + "phi"
                + subname
                + " = "
                + str((self.crest + 90 - self.phase) % 360.0)
                + "/deg;\n"
            )
            if self.Structure_Type == "TravellingWave":
                output += (
                    "ffac"
                    + subname
                    + " = "
                    + str(
                        (1 + (0.005 * self.length**1.5))
                        * (9.0 / (2.0 * np.pi))
                        * self.field_amplitude
                    )
                    + ";\n"
                )
            else:
                output += "ffac" + subname + " = " + str(self.field_amplitude) + ";\n"

            # if False and self.Structure_Type == 'TravellingWave' and hasattr(self, 'attenuation_constant') and hasattr(self, 'shunt_impedance') and hasattr(self, 'design_power') and hasattr(self, 'design_gamma'):
            #     '''
            #     trwlinac(ECS,ao,Rs,Po,P,Go,thetao,phi,w,L)
            #     '''
            #     relpos, relrot = ccs.relative_position(self.middle, self.global_rotation)
            #     power = float(self.field_amplitude) / 25e6 * float(self.design_power)
            #     output += 'trwlinac' + '( ' + ccs.name + ', "z", '+ str(relpos[2]+self.coupling_cell_length) + ', ' + str(self.attenuation_constant / self.length) + ', ' + str(float(self.shunt_impedance) / self.length)\
            #             + ', ' + str(float(self.design_power) / self.length) + ', ' + str(power / self.length) + ', ' + str(1000/0.511) + ', ' + str(self.crest)\
            #             + ', '+str(self.phase)+', w'+subname+', ' + str(self.length) + ');\n'
            # else:
            output += (
                "map1D_TM"
                + "( "
                + ccs.name
                + ", "
                + ccs_label
                + ", "
                + value_text
                + ', "'
                + str(self.generate_field_file_name(self.field_definition_gdf))
                + '", "Z","Ez", ffac'
                + subname
                + ", phi"
                + subname
                + ", w"
                + subname
                + ");\n"
            )
            if expand_substitution(self, self.wakefield_gdf) is not None:
                output += (
                    "wakefield("
                    + ccs.name
                    + ", "
                    + ccs_label
                    + ", "
                    + value_text
                    + ", "
                    + str(self.length)
                    + ', 50, "'
                    + str(self.generate_field_file_name(self.wakefield_gdf))
                    + '", "z","Wx","Wy","Wz", "FieldFactorWz", '
                    + str(self.cells)
                    + " / "
                    + str(self.length)
                    + ', "FieldFactorWx", '
                    + str(self.cells)
                    + " / "
                    + str(self.length)
                    + ', "FieldFactorWy", '
                    + str(self.cells)
                    + " / "
                    + str(self.length)
                    + ") ;\n"
                )
            else:
                if (
                    expand_substitution(self, self.longitudinal_wakefield_gdf)
                    is not None
                ):
                    output += (
                        "wakefield("
                        + ccs.name
                        + ", "
                        + ccs_label
                        + ", "
                        + value_text
                        + ", "
                        + str(self.length)
                        + ', 50, "'
                        + str(
                            self.generate_field_file_name(
                                self.longitudinal_wakefield_gdf
                            )
                        )
                        + '", "z","","","Wz", "FieldFactorWz", '
                        + str(self.cells)
                        + " / "
                        + str(self.length)
                        + ") ;\n"
                    )
                if expand_substitution(self, self.transverse_wakefield_gdf) is not None:
                    output += (
                        "wakefield("
                        + ccs.name
                        + ", "
                        + ccs_label
                        + ", "
                        + value_text
                        + ", "
                        + str(self.length)
                        + ', 50, "'
                        + str(
                            self.generate_field_file_name(self.transverse_wakefield_gdf)
                        )
                        + '", "z","Wx","Wy","", "FieldFactorWx", '
                        + str(self.cells)
                        + " / "
                        + str(self.length)
                        + ', "FieldFactorWy", '
                        + str(self.cells)
                        + " / "
                        + str(self.length)
                        + ") ;\n"
                    )
        else:
            output = ""
        return output


class longitudinal_wakefield(cavity):

    def __init__(self, name=None, type="longitudinal_wakefield", **kwargs):
        super().__init__(name, type, **kwargs)
        self.add_default("coupling_cell_length", 0)

    def write_ASTRA(self, startn):
        self.update_field_definition()
        basename = self.generate_field_file_name(self.field_definition)
        efield_def = ["Wk_filename", {"value": "'" + basename + "'", "default": ""}]
        output = ""
        if self.scale_kick > 0:
            for n in range(startn, startn + self.cells):
                output += self._write_ASTRA(
                    dict(
                        [
                            [
                                "Wk_Type",
                                {
                                    "value": self.waketype,
                                    "default": "'Taylor_Method_F'",
                                },
                            ],
                            efield_def,
                            ["Wk_x", {"value": self.x_offset, "default": 0}],
                            ["Wk_y", {"value": self.y_offset, "default": 0}],
                            [
                                "Wk_z",
                                {
                                    "value": self.start[2]
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

    def _write_Elegant(self):
        self.update_field_definition()
        original_field_definition_sdds = self.field_definition_sdds
        if self.field_definition_sdds is not None:
            self.field_definition_sdds = (
                '"' + self.generate_field_file_name(self.field_definition_sdds) + '"'
            )
        wholestring = ""
        etype = self._convertType_Elegant(self.objecttype)
        string = self.objectname + ": " + etype
        if self.length > 0:
            d = drift(
                self.objectname + "-drift", type="drift", **{"length": self.length}
            )
            wholestring += d._write_Elegant()
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
        self.field_definition_sdds = original_field_definition_sdds
        return wholestring


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


class solenoid(frameworkElement):

    def __init__(self, name=None, type="solenoid", **kwargs):
        super().__init__(name, type, **kwargs)
        self.add_default("scale_field", True)
        self.add_default("field_scale", 1)
        self.add_default("field_type", "1D")
        self.add_default("default_array_names", ["Z", "Bz"])

    def update_field_definition(self):
        if hasattr(self, "field_definition") and self.field_definition is not None:
            self.field_definition = expand_substitution(self, self.field_definition)
        if (
            hasattr(self, "field_definition_sdds")
            and self.field_definition_sdds is not None
        ):
            self.field_definition_sdds = expand_substitution(
                self, self.field_definition_sdds
            )
        if (
            hasattr(self, "field_definition_gdf")
            and self.field_definition_gdf is not None
        ):
            self.field_definition_gdf = expand_substitution(
                self, self.field_definition_gdf
            )

    def write_ASTRA(self, n, **kwargs):
        basename = self.generate_field_file_name(self.field_definition)
        efield_def = ["FILE_BFieLD", {"value": "'" + basename + "'", "default": ""}]
        return self._write_ASTRA(
            dict(
                [
                    ["S_pos", {"value": self.middle[2] + self.dz, "default": 0}],
                    efield_def,
                    ["MaxB", {"value": self.get_field_amplitude, "default": 0}],
                    ["S_smooth", {"value": self.smooth, "default": 10}],
                    ["S_xoff", {"value": self.middle[0] + self.dx, "default": 0}],
                    ["S_yoff", {"value": self.middle[1] + self.dy, "default": 0}],
                    ["S_xrot", {"value": self.y_rot + self.dy_rot, "default": 0}],
                    ["S_yrot", {"value": self.x_rot + self.dx_rot, "default": 0}],
                    ["S_noscale", {"value": not self.scale_field, "default": False}],
                ]
            ),
            n,
        )

    def write_GPT(self, Brho, ccs, *args, **kwargs):
        ccs_label, value_text = ccs.ccs_text(self.middle, self.rotation)
        if self.field_type.lower() == "1d":
            self.default_array_names = ["Z", "Bz"]
            """
            map1D_B("wcs",xOffset,0,zOffset+0.,cos(angle),0,-sin(angle),0,1,0,"bas_sol_norm.gdf","Z","Bz",gunSolField);
            """
            output = (
                "map1D_B"
                + "( "
                + ccs.name
                + ", "
                + ccs_label
                + ", "
                + value_text
                + ", "
                + '"'
                + str(self.generate_field_file_name(self.field_definition_gdf))
                + '", '
                + self.array_names_string()
                + ", "
                + str(expand_substitution(self, self.field_amplitude))
                + ");\n"
            )
        elif self.field_type.lower() == "3d":
            self.default_array_names = ["X", "Y", "Z", "Bx", "By", "Bz"]
            """
            map3D_B("wcs", xOffset,0,zOffset+0.,cos(angle),0,-sin(angle),0,1,0, "sol3.gdf", "x", "y", "z", "Bx", "By", "Bz", scale3);
            """
            output = (
                "map3D_B"
                + "( "
                + ccs.name
                + ", "
                + ccs_label
                + ", "
                + value_text
                + ", "
                + '"'
                + str(self.generate_field_file_name(self.field_definition_gdf))
                + '", '
                + self.array_names_string()
                + ", "
                + str(expand_substitution(self, self.field_amplitude))
                + ");\n"
            )
        return output


class aperture(frameworkElement):

    def __init__(self, name=None, type="aperture", **kwargs):
        super().__init__(name, type, **kwargs)
        self.number_of_elements = 1

    def write_GPT(self, Brho, ccs="wcs", *args, **kwargs):
        return ""
        # if self.shape == 'elliptical':
        #     output = 'rmax'
        # else:
        #     output = 'xymax'
        # output += '( "wcs", '+self.gpt_coordinates()+', '+str(self.horizontal_size)+', '+str(self.length)+');\n'
        # return output

    def write_ASTRA_Common(self, dic):
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

    def write_ASTRA_Circular(self, n):
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
        return self.write_ASTRA_Common(dic)

    def write_ASTRA_Planar(self, n, plane, width):
        dic = dict()
        dic["File_Aperture"] = {"value": plane}
        dic["Ap_R"] = {"value": width}
        return self.write_ASTRA_Common(dic)

    def write_ASTRA(self, n: int, **kwargs) -> str:
        self.number_of_elements = 0
        if self.shape == "elliptical" or self.shape == "circular":
            self.number_of_elements += 1
            dic = self.write_ASTRA_Circular(n)
            return self._write_ASTRA(dic, n)
        elif self.shape == "planar" or self.shape == "rectangular":
            text = ""
            if self.horizontal_size is not None and self.horizontal_size > 0:
                dic = self.write_ASTRA_Planar(n, "Col_X", 1e3 * self.horizontal_size)
                text += self._write_ASTRA(dic, n)
                self.number_of_elements += 1
            if self.vertical_size is not None and self.vertical_size > 0:
                dic = self.write_ASTRA_Planar(n, "Col_Y", 1e3 * self.vertical_size)
                if self.number_of_elements > 0:
                    self.number_of_elements += 1
                    n = n + 1
                    text += "\n"
                text += self._write_ASTRA(dic, n)
            return text
        elif self.shape == "scraper":
            text = ""
            if self.horizontal_size is not None and self.horizontal_size > 0:
                dic = self.write_ASTRA_Planar(n, "Scr_X", 1e3 * self.horizontal_size)
                text += self._write_ASTRA(dic, n)
                self.number_of_elements += 1
            if self.vertical_size is not None and self.vertical_size > 0:
                dic = self.write_ASTRA_Planar(n, "Scr_Y", 1e3 * self.vertical_size)
                if self.number_of_elements > 0:
                    self.number_of_elements += 1
                    n = n + 1
                    text += "\n"
                text += self._write_ASTRA(dic, n)
            return text


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


class cleaner(frameworkElement):

    def __init__(self, name=None, type="scatter", **kwargs):
        super().__init__(name, type, **kwargs)
        # print('Scatter object ', self.objectname,' - DP = ', self.objectproperties)

    def _write_Elegant(self):
        wholestring = ""
        etype = "clean"
        string = self.objectname + ": " + etype
        for key, value in merge_two_dicts(
            self.objectproperties, self.objectdefaults
        ).items():
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


class wall_current_monitor(frameworkElement):

    def __init__(self, name=None, type="wall_current_monitor", **kwargs):
        super().__init__(name, type, **kwargs)


class integrated_current_transformer(wall_current_monitor):

    def __init__(self, name=None, type="integrated_current_transformer", **kwargs):
        super().__init__(name, type, **kwargs)


class faraday_cup(wall_current_monitor):

    def __init__(self, name=None, type="faraday_cup", **kwargs):
        super().__init__(name, type, **kwargs)


class screen(frameworkElement):

    def __init__(self, name=None, type="screen", **kwargs):
        super().__init__(name, type, **kwargs)
        self.beam = rbf.beam()
        if "output_filename" not in kwargs:
            self.output_filename = str(self.objectname) + ".sdds"

    def write_ASTRA(self, n, **kwargs):
        return self._write_ASTRA(
            dict(
                [
                    ["Screen", {"value": self.middle[2], "default": 0}],
                    ["Scr_xrot", {"value": self.y_rot + self.dy_rot, "default": 0}],
                    ["Scr_yrot", {"value": self.x_rot + self.dx_rot, "default": 0}],
                ]
            ),
            n,
        )

    def _write_Elegant(self):
        wholestring = ""
        etype = self._convertType_Elegant(self.objecttype)
        string = self.objectname + ": " + etype
        # if self.length > 0:
        #     d = drift(self.objectname+'-drift-01', type='drift', **{'length': self.length/2})
        #     wholestring+=d._write_Elegant()
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
        # if self.length > 0:
        #     d = drift(self.objectname+'-drift-02', type='drift', **{'length': self.length/2})
        #     wholestring+=d._write_Elegant()
        return wholestring

    def write_CSRTrack(self, n):
        z = self.middle[2]
        return (
            """quadrupole{\nposition{rho="""
            + str(z)
            + """, psi=0.0, marker=screen"""
            + str(n)
            + """a}\nproperties{strength=0.0, alpha=0, horizontal_offset=0,vertical_offset=0}\nposition{rho="""
            + str(z + 1e-6)
            + """, psi=0.0, marker=screen"""
            + str(n)
            + """b}\n}\n"""
        )

    def write_GPT(self, Brho, ccs="wcs", output_ccs=None, *args, **kwargs):
        relpos, _ = ccs.relative_position(self.middle, self.global_rotation)
        ccs_label, value_text = ccs.ccs_text(self.middle, self.rotation)
        self.gpt_screen_position = relpos[2]
        output = "screen( " + ccs.name + ', "I", ' + str(relpos[2]) + ","
        if output_ccs is not None:
            output += '"' + str(output_ccs) + '"'
        else:
            output += ccs.name
        output += ");\n"
        return output

    def find_ASTRA_filename(self, lattice, master_run_no, mult):
        for i in [0, -0.001, 0.001]:
            tempfilename = (
                lattice
                + "."
                + str(int(round((self.middle[2] + i - self.zstart[2]) * mult))).zfill(4)
                + "."
                + str(master_run_no).zfill(3)
            )
            # print(self.middle[2]+i-self.zstart[2], tempfilename, os.path.isfile(self.global_parameters['master_subdir'] + '/' + tempfilename))
            if os.path.isfile(
                self.global_parameters["master_subdir"] + "/" + tempfilename
            ):
                return tempfilename
        return None

    def astra_to_hdf5(self, lattice, cathode=False, mult=100):
        master_run_no = (
            self.global_parameters["run_no"]
            if "run_no" in self.global_parameters
            else 1
        )
        astrabeamfilename = self.find_ASTRA_filename(lattice, master_run_no, mult)
        if astrabeamfilename is None:
            print(("Screen Error: ", lattice, self.middle[2], self.zstart[2]))
        else:
            rbf.astra.read_astra_beam_file(
                self.beam,
                (
                    self.global_parameters["master_subdir"] + "/" + astrabeamfilename
                ).strip('"'),
                normaliseZ=False,
            )
            rbf.hdf5.rotate_beamXZ(
                self.beam,
                -1 * self.starting_rotation,
                preOffset=[0, 0, 0],
                postOffset=-1 * np.array(self.starting_offset),
            )
            HDF5filename = (self.objectname + ".hdf5").strip('"')
            toffset = self.beam["toffset"]
            rbf.hdf5.write_HDF5_beam_file(
                self.beam,
                self.global_parameters["master_subdir"] + "/" + HDF5filename,
                centered=False,
                sourcefilename=astrabeamfilename,
                pos=self.middle,
                cathode=cathode,
                toffset=toffset,
            )
            if self.global_parameters["delete_tracking_files"]:
                os.remove(
                    (
                        self.global_parameters["master_subdir"]
                        + "/"
                        + astrabeamfilename
                    ).strip('"')
                )

    def sdds_to_hdf5(self, sddsindex=1):
        self.beam.sddsindex = sddsindex
        elegantbeamfilename = self.output_filename.replace(".sdds", ".SDDS").strip('"')
        # print('sdds_to_hdf5')
        rbf.sdds.read_SDDS_beam_file(
            self.beam, self.global_parameters["master_subdir"] + "/" + elegantbeamfilename
        )
        # print('sdds_to_hdf5', 'read_SDDS_beam_file')
        HDF5filename = (
            self.output_filename.replace(".sdds", ".hdf5")
            .replace(".SDDS", ".hdf5")
            .strip('"')
        )
        rbf.hdf5.write_HDF5_beam_file(
            self.beam,
            self.global_parameters["master_subdir"] + "/" + HDF5filename,
            centered=False,
            sourcefilename=elegantbeamfilename,
            pos=self.middle,
            zoffset=self.end,
            toffset=(-1 * np.mean(self.global_parameters["beam"].t)),
        )
        # print('sdds_to_hdf5', 'write_HDF5_beam_file')
        if self.global_parameters["delete_tracking_files"]:
            os.remove(
                (
                    self.global_parameters["master_subdir"] + "/" + elegantbeamfilename
                ).strip('"')
            )

    def gdf_to_hdf5(self, gptbeamfilename, cathode=False, gdfbeam=None):
        # gptbeamfilename = self.objectname + '.' + str(int(round((self.allElementObjects[self.end].position_end[2])*100))).zfill(4) + '.' + str(master_run_no).zfill(3)
        # try:
        # print('Converting screen', self.objectname,'at', self.gpt_screen_position)
        rbf.gdf.read_gdf_beam_file(
            self.beam,
            self.global_parameters["master_subdir"] + "/" + gptbeamfilename,
            position=self.gpt_screen_position,
            gdfbeam=gdfbeam,
        )
        HDF5filename = self.objectname + ".hdf5"
        rbf.hdf5.write_HDF5_beam_file(
            self.beam,
            self.global_parameters["master_subdir"] + "/" + HDF5filename,
            centered=False,
            sourcefilename=gptbeamfilename,
            pos=self.middle,
            xoffset=self.end[0],
            cathode=cathode,
            toffset=(-1 * np.mean(self.beam.t)),
        )
        # except:
        #     print('Error with screen', self.objectname,'at', self.gpt_screen_position)
        if self.global_parameters["delete_tracking_files"]:
            os.remove(
                (self.global_parameters["master_subdir"] + "/" + gptbeamfilename).strip(
                    '"'
                )
            )


class monitor(screen):

    def __init__(self, name=None, type="monitor", **kwargs):
        super().__init__(name, type, **kwargs)


class watch_point(screen):

    def __init__(self, name=None, type="watch_point", **kwargs):
        super().__init__(name, "screen", **kwargs)


class beam_position_monitor(screen):

    def __init__(self, name=None, type="beam_position_monitor", **kwargs):
        super().__init__(name, type, **kwargs)

    def write_ASTRA(self, n, **kwargs):
        return self._write_ASTRA(
            dict(
                [
                    ["Screen", {"value": self.middle[2], "default": 0}],
                    ["Scr_xrot", {"value": self.y_rot + self.dy_rot, "default": 0}],
                    ["Scr_yrot", {"value": self.x_rot + self.dx_rot, "default": 0}],
                ]
            ),
            n,
        )


class beam_arrival_monitor(screen):

    def __init__(self, name=None, type="beam_arrival_monitor", **kwargs):
        super().__init__(name, type, **kwargs)

    def write_ASTRA(self, n, **kwargs):
        return ""


class bunch_length_monitor(screen):

    def __init__(self, name=None, type="beam_arrival_monitor", **kwargs):
        super().__init__(name, type, **kwargs)

    def write_ASTRA(self, n, **kwargs):
        return ""


class collimator(aperture):

    def __init__(self, name=None, type="collimator", **kwargs):
        super().__init__(name, type, **kwargs)


class rcollimator(aperture):

    def __init__(self, name=None, type="rcollimator", **kwargs):
        super().__init__(name, type, **kwargs)


class apcontour(frameworkElement):

    def __init__(self, name=None, type="apcontour", **kwargs):
        super().__init__(name, type, **kwargs)
        self.add_default("resolution", 0.001)


class center(frameworkElement):

    def __init__(self, name=None, type="center", **kwargs):
        super().__init__(name, type, **kwargs)


class marker(screen):

    def __init__(self, name=None, type="marker", **kwargs):
        super().__init__(name, "screen", **kwargs)

    def write_CSRTrack(self, n):
        return ""

    def _write_Elegant(self) -> str:
        obj = self.objecttype
        self.objecttype = "screen"
        output = super()._write_Elegant()
        self.objecttype = obj
        return output


class drift(frameworkElement):

    def __init__(self, name=None, type="drift", **kwargs):
        super().__init__(name, type, **kwargs)

    # def _write_Elegant(self):
    #     wholestring=''
    #     etype = self._convertType_Elegant(self.objecttype)
    #     string = self.objectname+': '+ etype
    #     for key, value in list(merge_two_dicts(self.objectproperties, self.objectdefaults).items()):
    #         if not key is 'name' and not key is 'type' and not key is 'commandtype' and self._convertKeyword_Elegant(key) in elements_Elegant[etype]:
    #             value = getattr(self, key) if hasattr(self, key) and getattr(self, key) is not None else value
    #             key = self._convertKeyword_Elegant(key)
    #             value = 1 if value is True else value
    #             value = 0 if value is False else value
    #             tmpstring = ', '+key+' = '+str(value)
    #             if len(string+tmpstring) > 76:
    #                 wholestring+=string+',&\n'
    #                 string=''
    #                 string+=tmpstring[2::]
    #             else:
    #                 string+= tmpstring
    #     wholestring+=string+';\n'
    #     return wholestring


class shutter(csrdrift):

    def __init__(self, name=None, type="shutter", **kwargs):
        super().__init__(name, type, **kwargs)


class valve(csrdrift):

    def __init__(self, name=None, type="valve", **kwargs):
        super().__init__(name, type, **kwargs)


class bellows(csrdrift):

    def __init__(self, name=None, type="bellows", **kwargs):
        super().__init__(name, type, **kwargs)


class fel_modulator(frameworkElement):

    def __init__(self, name=None, type="modulator", **kwargs):
        super().__init__(name, type, **kwargs)
        self.add_default("k1l", 0)
        self.add_default("n_steps", 1 * self.periods)

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
        etype = self._convertType_Elegant(self.objecttype)
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


class charge(frameworkElement):
    def __init__(self, name=None, type="charge", **kwargs):
        super().__init__(name, "charge", **kwargs)


class gpt_ccs(Munch):

    def __init__(self, name, position, rotation, intersect=0):
        super().__init__()
        self._name = name
        self.intersect = intersect
        # print(self._name, self.intersect)
        self.x, self.y, self.z = position
        self.psi, self.phi, self.theta = rotation

    def relative_position(self, position, rotation):
        x, y, z = position
        pitch, yaw, roll = rotation
        # psi, phi, theta = rotation

        # print(self.name, [x - self.x, y - self.y, z - self.z])
        # print(self.name, [psi - self.psi, phi - self.phi, theta - self.theta])
        # newpos = [x - self.x, y - self.y, z - self.z]
        length = np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2 + (z - self.z) ** 2)
        # print('newpos = ', self.name,  x, self.x, y, self.y, z, self.z)
        finalrot = np.array([pitch - self.psi, yaw - self.phi, roll - self.theta])
        finalpos = np.array(
            [0, 0, abs(self.intersect) + length]
        )  # + np.dot(np.array(newpos), _rotation_matrix(-self.theta))
        # print(self._name, finalpos, finalrot)
        return finalpos, finalrot

    @property
    def name(self):
        return '"' + self._name + '"'

    @property
    def position(self):
        return self.x, self.y, self.z

    @property
    def rotation(self):
        return self.psi, self.phi, self.theta

    def ccs_text(self, position, rotation):
        finalpos, finalrot = self.relative_position(position, rotation)
        x, y, z = finalpos
        psi, phi, theta = finalrot
        ccs_label = ""
        value_text = ""
        if abs(x) > 0:
            ccs_label += "x"
            value_text += "," + str(x)
        if abs(y) > 0:
            ccs_label += "y"
            value_text += "," + str(y)
        if abs(z) > 0:
            ccs_label += "z"
            value_text += "," + str(z)
        if abs(psi) > 0:
            ccs_label += "X"
            value_text += "," + str(psi)
        if abs(phi) > 0:
            ccs_label += "Y"
            value_text += "," + str(phi)
        if abs(theta) > 0:
            ccs_label += "Z"
            value_text += "," + str(theta)
        return '"' + ccs_label + '"', value_text.strip(",")


class global_error(frameworkElement):

    def __init__(self, name=None, type="global_error", **kwargs):
        super().__init__(name, "global_error", **kwargs)
        # self._errordict = {}

    def add_Error(self, type, sigma):
        if type in global_Error_Types:
            self.add_property(type, sigma)

    def write_ASTRA(self):
        return self._write_ASTRA(
            dict([[key, {"value": value}] for key, value in self._errordict])
        )

    def write_GPT(self, Brho, ccs="wcs", *args, **kwargs):
        relpos, relrot = ccs.relative_position(self.middle, [0, 0, 0])
        coord = self.gpt_coordinates(relpos, relrot)
        output = (
            str(self.objecttype)
            + "( "
            + ccs.name
            + ", "
            + coord
            + ", "
            + str(self.length)
            + ", "
            + str(Brho * self.k1)
            + ");\n"
        )
        return output
