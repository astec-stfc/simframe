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


def interpret_ocelot_data(self, lattice_name, fdat):
    fdat["s"] += self.z[-1]
    self.append("z", fdat["s"])
    cp = fdat["E"] * 1e-3
    # self.append('cp', cp)
    self.cp.val = np.append(self.cp.val, cp / constants.elementary_charge)
    ke = np.array(
        (np.sqrt(self.E0**2 + cp**2) - self.E0**2) / constants.elementary_charge
    )
    self.kinetic_energy.val = np.append(self.kinetic_energy.val, ke)
    gamma = 1 + ke / self.E0_eV
    self.gamma.val = np.append(self.gamma.val, gamma)
    self.p.val = np.append(self.p.val, cp * self.q_over_c)
    self.enx.val = np.append(self.enx.val, fdat["emit_xn"])
    self.ex.val = np.append(self.ex.val, fdat["emit_x"])
    self.eny.val = np.append(self.eny.val, fdat["emit_yn"])
    self.ey.val = np.append(self.ey.val, fdat["emit_y"])
    self.enz.val = np.append(self.enz.val, np.zeros(len(fdat["s"])))
    self.ez.val = np.append(self.ez.val, np.zeros(len(fdat["s"])))
    self.beta_x.val = np.append(self.beta_x.val, fdat["beta_x"])
    self.alpha_x.val = np.append(self.alpha_x.val, fdat["alpha_x"])
    self.gamma_x.val = np.append(self.gamma_x.val, fdat["gamma_x"])
    self.beta_y.val = np.append(self.beta_y.val, fdat["beta_y"])
    self.alpha_y.val = np.append(self.alpha_y.val, fdat["alpha_y"])
    self.gamma_y.val = np.append(self.gamma_y.val, fdat["gamma_y"])
    self.beta_z.val = np.append(self.beta_z.val, np.zeros(len(fdat["s"])))
    self.gamma_z.val = np.append(self.gamma_z.val, np.zeros(len(fdat["s"])))
    self.alpha_z.val = np.append(self.alpha_z.val, np.zeros(len(fdat["s"])))
    self.sigma_x.val = np.append(self.sigma_x.val, fdat["xx"])
    self.sigma_y.val = np.append(self.sigma_y.val, fdat["yy"])
    self.sigma_xp.val = np.append(self.sigma_xp.val, fdat["pxpx"])
    self.sigma_yp.val = np.append(self.sigma_yp.val, fdat["pypy"])
    self.sigma_t.val = np.append(self.sigma_t.val, fdat["tautau"])
    self.mean_x.val = np.append(self.mean_x.val, fdat["x"])
    self.mean_y.val = np.append(self.mean_y.val, fdat["y"])
    beta = np.sqrt(1 - (gamma**-2))
    self.t.val = np.append(self.t.val, fdat["s"] / (beta * constants.speed_of_light))
    self.sigma_z.val = np.append(
        self.sigma_z.val, fdat["tautau"] * (beta * constants.speed_of_light)
    )
    # self.append('sigma_cp', elegantData['Sdelta'] * cp )
    self.sigma_cp.val = np.append(
        self.sigma_cp.val, fdat["pp"] * cp / constants.elementary_charge
    )
    # print('elegant = ', (elegantData['Sdelta'] * cp / constants.elementary_charge)[-1)
    self.sigma_p.val = np.append(self.sigma_p.val, fdat["pp"])
    self.mux.val = np.append(self.mux.val, fdat["mux"])
    self.muy.val = np.append(self.muy.val, fdat["muy"])
    self.eta_x.val = np.append(self.eta_x.val, fdat["Dx"])
    self.eta_xp.val = np.append(self.eta_xp.val, fdat["Dxp"])
    self.eta_y.val = np.append(self.eta_y.val, fdat["Dy"])
    self.eta_yp.val = np.append(self.eta_yp.val, fdat["Dyp"])
    self.element_name.val = np.append(self.element_name.val, np.zeros(len(fdat["s"])))
    self.lattice_name.val = np.append(
        self.lattice_name.val, np.zeros(len(fdat["s"])), lattice_name
    )
    # ## BEAM parameters
    self.ecnx.val = np.append(self.ecnx.val, fdat["emit_xn"])
    self.ecny.val = np.append(self.ecny.val, fdat["emit_yn"])
    self.eta_x_beam.val = np.append(self.eta_x_beam.val, fdat["Dx"])
    self.eta_xp_beam.val = np.append(self.eta_xp_beam.val, fdat["Dxp"])
    self.eta_y_beam.val = np.append(self.eta_y_beam.val, fdat["Dy"])
    self.eta_yp_beam.val = np.append(self.eta_yp_beam.val, fdat["Dyp"])
    self.beta_x_beam.val = np.append(
        self.beta_x_beam.val, np.sqrt(fdat["emit_x"] / fdat["xx"])
    )
    self.beta_y_beam.val = np.append(
        self.beta_y_beam.val, np.sqrt(fdat["emit_y"] / fdat["yy"])
    )
    self.alpha_x_beam.val = np.append(
        self.alpha_x_beam.val, np.sqrt(fdat["emit_x"] / fdat["pxpx"])
    )
    self.alpha_y_beam.val = np.append(
        self.alpha_y_beam.val, np.sqrt(fdat["emit_y"] / fdat["pypy"])
    )
    self.cp_eV = self.cp
    self.cp_eV = self.cp
    self.sigma_cp_eV = self.sigma_cp
    for k in self.__dict__.keys():
        try:
            if len(getattr(self, k)) < len(getattr(self, "z")):
                self.append(k, np.zeros(len(fdat["s"])))
        except Exception as e:
            pass
