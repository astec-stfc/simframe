import os
from ..SDDSFile import SDDSFile
import numpy as np
from .. import constants


def read_elegant_floor_file(
    self, filename, offset=[0, 0, 0], rotation=[0, 0, 0], reset=True
):
    if reset:
        self.reset_dicts()
    elegantObject = read_sdds_file(filename)
    elegantData = elegantObject.data
    self["x"] = twissData(
        [np.round(x + offset[0], decimals=6) for x in elegantData["X"]], units="m"
    )
    self["y"] = twissData(
        [np.round(y + offset[1], decimals=6) for y in elegantData["Y"]], units="m"
    )
    self["z"] = twissData(
        [np.round(z + offset[2], decimals=6) for z in elegantData["Z"]], units="m"
    )
    self["theta"] = twissData(
        [np.round(theta + rotation[0], decimals=6) for theta in elegantData["theta"]],
        units="radians",
    )
    self["phi"] = twissData(
        [np.round(phi + rotation[1], decimals=6) for phi in elegantData["phi"]],
        units="radians",
    )
    self["psi"] = twissData(
        [np.round(psi + rotation[2], decimals=6) for psi in elegantData["psi"]],
        units="radians",
    )
    xyz = list(zip(self["x"], self["y"], self["z"]))
    thetaphipsi = list(zip(self["phi"], self["psi"], self["theta"]))
    return list(zip(elegantData["ElementName"], xyz[-1:] + xyz[:-1], xyz, thetaphipsi))[
        1:
    ]


def read_elegant_twiss_files(self, filename, startS=0, reset=True):
    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            print('reading new file', f)
            read_elegant_twiss_files(self, f, reset=False)
    elif os.path.isfile(filename):
        pre, ext = os.path.splitext(filename)
        lattice_name = os.path.basename(pre)
        self.sddsindex += 1
        elegantObject = SDDSFile(index=(self.sddsindex))
        elegantObject.read_file(pre + ".flr")
        elegantObject.read_file(pre + ".sig")
        elegantObject.read_file(pre + ".twi")
        elegantObject.read_file(pre + ".cen")
        elegantData = elegantObject.data
        for k in elegantData:
            # handling for multiple elegant runs per file (e.g. error simulations)
            # by default extract only the first run (in ELEGANT this is the fiducial)
            if isinstance(elegantData[k], np.ndarray) and (elegantData[k].ndim > 1):
                elegantData[k] = elegantData[k][0]
        z = elegantData["Z"]
        self.append("z", z)
        cp = elegantData["pCentral0"] * self.E0
        # self.append('cp', cp)
        self.append("cp", cp / constants.elementary_charge)
        self.append("mean_cp", cp / constants.elementary_charge)
        ke = np.array(
            (np.sqrt(self.E0**2 + cp**2) - self.E0**2) / constants.elementary_charge
        )
        self.append("kinetic_energy", ke)
        gamma = 1 + ke / self.E0_eV
        self.append("gamma", gamma)
        self.append("mean_gamma", gamma)
        self.append("p", cp * self.q_over_c)
        self.append("enx", elegantData["enx"])
        self.append("ex", elegantData["ex"])
        self.append("eny", elegantData["eny"])
        self.append("ey", elegantData["ey"])
        self.append("enz", np.zeros(len(elegantData["Z"])))
        self.append("ez", np.zeros(len(elegantData["Z"])))
        self.append("beta_x", elegantData["betax"])
        self.append("alpha_x", elegantData["alphax"])
        self.append("gamma_x", (1 + elegantData["alphax"] ** 2) / elegantData["betax"])
        self.append("beta_y", elegantData["betay"])
        self.append("alpha_y", elegantData["alphay"])
        self.append("gamma_y", (1 + elegantData["alphay"] ** 2) / elegantData["betay"])
        self.append("beta_z", np.zeros(len(elegantData["Z"])))
        self.append("gamma_z", np.zeros(len(elegantData["Z"])))
        self.append("alpha_z", np.zeros(len(elegantData["Z"])))
        self.append("sigma_x", elegantData["Sx"])
        self.append("sigma_y", elegantData["Sy"])
        self.append("sigma_xp", elegantData["Sxp"])
        self.append("sigma_yp", elegantData["Syp"])
        self.append("sigma_t", elegantData["St"])
        self.append("mean_x", elegantData["Cx"])
        self.append("mean_y", elegantData["Cy"])
        beta = np.sqrt(1 - (gamma**-2))
        # print 'len(z) = ', len(z), '  len(beta) = ', len(beta)
        self.append("t", z / (beta * constants.speed_of_light))
        self.append("sigma_z", elegantData["St"] * (beta * constants.speed_of_light))
        # self.append('sigma_cp', elegantData['Sdelta'] * cp )
        self.append(
            "sigma_cp", elegantData["Sdelta"] * cp / constants.elementary_charge
        )
        # print('elegant = ', (elegantData['Sdelta'] * cp / constants.elementary_charge)[-1)
        self.append("sigma_p", elegantData["Sdelta"])
        self.append("mux", elegantData["psix"] / (2 * constants.pi))
        self.append("muy", elegantData["psiy"] / (2 * constants.pi))
        self.append("eta_x", elegantData["etax"])
        self.append("eta_xp", elegantData["etaxp"])
        self.append('eta_y', elegantData['etay'])
        self.append('eta_yp', elegantData['etayp'])
        self.append("element_name", elegantData["ElementName"])
        self.append("lattice_name", np.full(len(elegantData["ElementName"]), lattice_name))
        # ## BEAM parameters
        self.append("ecnx", elegantData["ecnx"])
        self.append("ecny", elegantData["ecny"])
        self.append("eta_x_beam", elegantData["s16"] / (elegantData["s6"] ** 2))
        self.append("eta_xp_beam", elegantData["s26"] / (elegantData["s6"] ** 2))
        self.append("eta_y_beam", elegantData["s36"] / (elegantData["s6"] ** 2))
        self.append("eta_yp_beam", elegantData["s46"] / (elegantData["s6"] ** 2))
        self.append("beta_x_beam", elegantData["betaxBeam"])
        self.append("beta_y_beam", elegantData["betayBeam"])
        self.append("alpha_x_beam", elegantData["alphaxBeam"])
        self.append("alpha_y_beam", elegantData["alphayBeam"])
        self["cp_eV"] = self["cp"]
        self["sigma_cp_eV"] = self["sigma_cp"]
        self.elegantData = elegantData
