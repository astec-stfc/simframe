import os
import numpy as np
from .. import constants


def cumtrapz(x=[], y=[]):
    return [np.trapz(x=x[:n], y=y[:n]) for n in range(len(x))]


def read_ocelot_twiss_files(self, filename, reset=True):
    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            self.read_ocelot_twiss_files(f, reset=False)
    elif os.path.isfile(filename):
        lattice_name = os.path.basename(filename).split(".")[0]
        if "xemit" not in filename.lower():
            filename = filename.replace("Yemit", "Xemit").replace("Zemit", "Xemit")
        fdat = {}
        with np.load(filename) as data:
            for key, value in data.items():
                fdat.update({key: value})
        interpret_ocelot_data(self, lattice_name, fdat)


def interpret_ocelot_data(self, fdat):
    self.append("z", fdat["s"])
    self.append("s", fdat["s"])
    E = fdat["E"] * 1e9
    ke = E - self.E0_eV
    gamma = E / self.E0_eV
    cp = np.sqrt(E**2 - self.E0_eV**2)
    # self.append('cp', cp)
    self.append("cp", cp)
    self.append("mean_cp", cp)
    self.append("kinetic_energy", ke)
    self.append("gamma", gamma)
    self.append("mean_gamma", gamma)
    self.append("p", cp * self.q_over_c)
    self.append("enx", fdat["emit_xn"])
    self.append("ex", fdat["emit_x"])
    self.append("eny", fdat["emit_yn"])
    self.append("ey", fdat["emit_y"])
    self.append("enz", np.zeros(len(fdat["s"])))
    self.append("ez", np.zeros(len(fdat["s"])))
    self.append("beta_x", fdat["beta_x"])
    self.append("alpha_x", fdat["alpha_x"])
    self.append("gamma_x", fdat["gamma_x"])
    self.append("beta_y", fdat["beta_y"])
    self.append("alpha_y", fdat["alpha_y"])
    self.append("gamma_y", fdat["gamma_y"])
    self.append("beta_z", np.zeros(len(fdat["s"])))
    self.append("gamma_z", np.zeros(len(fdat["s"])))
    self.append("alpha_z", np.zeros(len(fdat["s"])))
    self.append("sigma_x", np.sqrt(fdat["xx"]))
    self.append("sigma_y", np.sqrt(fdat["yy"]))
    self.append("sigma_xp", np.sqrt(fdat["pxpx"]))
    self.append("sigma_yp", np.sqrt(fdat["pypy"]))
    self.append("sigma_t", np.sqrt(fdat["tautau"]) / constants.speed_of_light)
    self.append("mean_x", fdat["x"])
    self.append("mean_y", fdat["y"])
    beta = np.sqrt(1 - (gamma**-2))
    self.append("t", fdat["s"] / (beta * constants.speed_of_light))
    self.append("sigma_z", np.sqrt(fdat["tautau"]) * beta)
    self.append("sigma_cp", np.sqrt(fdat["pp"]) * cp)
    self.append("sigma_p", np.sqrt(fdat["pp"]))
    self.append("mux", fdat["mux"])
    self.append("muy", fdat["muy"])
    self.append("eta_x", fdat["Dx"])
    self.append("eta_xp", fdat["Dxp"])
    self.append("eta_y", fdat["Dy"])
    self.append("eta_yp", fdat["Dyp"])
    self.append("element_name", np.zeros(len(fdat["s"])))
    self.append("lattice_name", np.zeros(len(fdat["s"])), lattice_name)
    # ## BEAM parameters
    self.append("ecnx", fdat["emit_xn"])
    self.append("ecny", fdat["emit_yn"])
    self.append("eta_x_beam", fdat["Dx"])
    self.append("eta_xp_beam", fdat["Dxp"])
    self.append("eta_y_beam", fdat["Dy"])
    self.append("eta_yp_beam", fdat["Dyp"])
    self.append("beta_x_beam", np.sqrt(fdat["emit_x"] / np.sqrt(fdat["xx"])))
    self.append("beta_y_beam", np.sqrt(fdat["emit_y"] / np.sqrt(fdat["yy"])))
    self.append("alpha_x_beam", np.sqrt(fdat["emit_x"] / np.sqrt(fdat["pxpx"])))
    self.append("alpha_y_beam", np.sqrt(fdat["emit_y"] / np.sqrt(fdat["pypy"])))
    self["cp_eV"] = self["cp"]
    self["sigma_cp_eV"] = self["sigma_cp"]
    self["cp_eV"] = self["cp"]
    self["sigma_cp_eV"] = self["sigma_cp"]
    for k in self.__dict__.keys():
        try:
            if len(getattr(self, k)) < len(getattr(self, "z")):
                self.append(k, np.zeros(len(fdat["s"])))
        except Exception:
            pass
