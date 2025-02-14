from munch import Munch
import numpy as np


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
        if ccs_label == "" and value_text == "":
            ccs_label = "z"
            value_text = "," + str(0)
        return '"' + ccs_label + '"', value_text.strip(",")
