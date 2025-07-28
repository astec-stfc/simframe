from copy import copy
import numpy as np
from SimulationFramework.Elements.gpt_ccs import gpt_ccs
from SimulationFramework.Framework_objects import (
    frameworkElement,
    elements_Elegant,
    _rotation_matrix,
    chop,
    type_conversion_rules_Ocelot,
)
from SimulationFramework.FrameworkHelperFunctions import checkValue
from SimulationFramework.Modules.merge_two_dicts import merge_two_dicts
from ocelot.cpbd.elements import Aperture, Marker
import inspect


def add(x, y):
    return x + y


class dipole(frameworkElement):
    csr_bins: int = 100
    deltaL: float = 0
    csr_enable: int = 1
    isr_enable: bool = True
    n_kicks: int = 8
    sr_enable: bool = True
    integration_order: int = 4
    nonlinear: int = 1
    smoothing_half_width: int = 1
    edge_order: int = 2
    edge1_effects: int = 1
    edge2_effects: int = 1
    angle: float = 0.0
    width: float = 0.2
    entrance_edge_angle: float | str = "angle"
    exit_edge_angle: float | str = "angle"

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(dipole, self).__init__(
            *args,
            **kwargs,
        )


    def __setattr__(self, name, value):
        # Let Pydantic set known fields normally
        if name in self.model_fields:
            super().__setattr__(name, value)
        else:
            # Store extras in __dict__ (allowed by Config.extra = 'allow')
            self.__dict__[name] = value

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

    def get_angle(self):
        return self.angle

    @property
    def arc_middle(self):
        sx, sy, sz = self.position_start
        angle = -self.get_angle()
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
        angle = -self.get_angle()
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
        angle = -self.get_angle()
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
        angle = -self.get_angle()
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
        angle = -self.get_angle()
        len = self.length
        if abs(angle) > 0:
            cx = 0
            cy = 0
            cz = -len * np.tan(0.5 * angle) / angle
            vec = [cx, cy, cz]
        else:
            vec = [0, 0, -len / 2.0]
        return np.array(middle) + self.rotated_position(
            np.array(vec), offset=self.starting_offset, theta=self.y_rot
        )

    @property
    def position_end(self):
        start = self.position_start
        angle = -self.get_angle()
        if abs(angle) > 1e-9:
            ex = (self.length * (1 - np.cos(angle))) / angle
            ey = 0
            ez = (self.length * (np.sin(angle))) / angle
            vec = [ex, ey, ez]
        else:
            vec = [0, 0, self.length]
        return np.array(start) + self.rotated_position(
            np.array(vec), offset=self.starting_offset, theta=self.y_rot
        )

    @property
    def astra_end(self):
        angle = -self.get_angle()
        if abs(angle) > 1e-9:
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
        return self.length * np.tan(0.5 * self.get_angle()) / self.get_angle()

    @property
    def rho(self):
        return (
            self.length / self.get_angle()
            if self.length is not None and abs(self.get_angle()) > 1e-9
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
        setattr(self, "k1", self.k1 if self.k1 is not None else 0)
        for key, value in self.objectproperties:
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

    def _write_Ocelot(self):
        k1 = self.k1 if self.k1 is not None else 0
        k2 = self.k2 if self.k2 is not None else 0
        keydict = merge_two_dicts(
            {"k1": k1, "k2": k2},
            merge_two_dicts(self.objectproperties, self.objectdefaults),
        )
        valdict = {"eid": self.objectname}
        for key, value in keydict.items():
            if (not key in ["name", "type", "commandtype"]) and (
                not type(type_conversion_rules_Ocelot[self.objecttype])
                in [Aperture, Marker]
            ):
                if "edge_angle" in key:
                    key = self._convertKeword_Ocelot(key)
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
                    key = self._convertKeword_Ocelot(key)
                value = 1 if value is True else value
                value = 0 if value is False else value
                if (
                    key
                    in inspect.getfullargspec(
                        type_conversion_rules_Ocelot[self.objecttype]
                    ).args
                ):
                    valdict.update({key: value})
        obj = type_conversion_rules_Ocelot[self.objecttype](**valdict)
        return obj

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
        theta = self.get_angle() - self.e2 + rotation
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

    def _write_CSRTrack(self, n):
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
            + str(chop(self.theta + self.get_angle() - self.e2))
            + """, marker=d"""
            + str(n)
            + """b}\n}\n"""
        )

    def _write_ASTRA(self, n, **kwargs):
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
            return self._write_ASTRA_dictionary(params, n)
        else:
            return None

    def gpt_coordinates(self, position, rotation):
        angle = -1 * self.get_angle()
        x, y, z = chop(position, 1e-6)
        psi, phi, theta = rotation
        output = ""
        for c in [0, 0, z]:
            output += str(c) + ", "
        output += "cos(" + str(angle) + "), 0, -sin(" + str(angle) + "), 0, 1 ,0"
        return output

    def _write_GPT(self, Brho, ccs, *args, **kwargs):
        field = 1.0 * self.get_angle() * Brho / self.length
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
            e1 = self.e1 if self.get_angle() >= 0 else -1 * self.e1
            e2 = self.e2 if self.get_angle() >= 0 else -1 * self.e2
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
        if abs(self.get_angle()) > 0 and abs(self.rho) < 100:
            # print('Creating new CCS')
            number = (
                str(int(ccs._name.split("_")[1]) + 1) if ccs._name != "wcs" else "1"
            )
            name = "ccs_" + number if ccs._name != "wcs" else "ccs_1"
            # print('middle position = ', self.start, self.middle)
            return gpt_ccs(
                name,
                self.middle,
                self.global_rotation + np.array([0, 0, -self.get_angle()]),
                0 * abs(self.intersect),
            )
        else:
            return ccs
