import os
import numpy as np
from ..gdf_emit import gdf_emit
from .. import constants


def read_gdf_emit_file_object(self, file):
    if isinstance(file, (str)):
        return gdf_emit(file)
    elif isinstance(file, (gdf_emit)):
        return file
    else:
        raise Exception("file is not str or gdf object!")


def read_gdf_twiss_files(self, filename=None, gdfbeam=None, reset=True):
    if reset:
        self.reset_dicts()
    if isinstance(filename, (list, tuple)):
        for f in filename:
            self.read_gdf_twiss_files(filename=f, reset=False)
    elif os.path.isfile(filename):
        lattice_name = os.path.basename(filename).split(".")[0]
        if gdfbeam is None and filename is not None:
            self.gdfbeam = gdfbeamdata = read_gdf_emit_file_object(self, filename)

        if hasattr(gdfbeamdata, "avgz"):
            # mjohnson code added 2022-08-11
            nsteps = len(gdfbeamdata.avgz)
            z_sort = np.array(
                [x for _, x in sorted(zip(gdfbeamdata.avgt, gdfbeamdata.avgz))],
                dtype=float,
            )
            order = np.array(
                [x for _, x in sorted(zip(gdfbeamdata.avgt, np.arange(nsteps)))],
                dtype=int,
            )

            offset = 0.0
            for i in range(1, nsteps):
                pos = 0.0 if i == 0 else z_sort[i - 1]
                if z_sort[i] < pos:
                    offset += pos

                gdfbeamdata.avgz[order[i]] = z_sort[i] + offset

            # original code begins
            self.append("z", gdfbeamdata.avgz)

        elif hasattr(gdfbeamdata, "position"):
            self.append("z", gdfbeamdata.position)
        cp = self.E0 * np.sqrt(gdfbeamdata.avgG**2 - 1)
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
        self.append("enx", gdfbeamdata.nemixrms)
        self.append("ex", gdfbeamdata.nemixrms / gdfbeamdata.avgG)
        self.append("eny", gdfbeamdata.nemiyrms)
        self.append("ey", gdfbeamdata.nemiyrms / gdfbeamdata.avgG)
        self.append("enz", gdfbeamdata.nemizrms)
        self.append("ez", gdfbeamdata.nemizrms / gdfbeamdata.avgG)
        self.append("beta_x", gdfbeamdata.CSbetax)
        self.append("alpha_x", gdfbeamdata.CSalphax)
        self.append("gamma_x", (1 + gdfbeamdata.CSalphax**2) / gdfbeamdata.CSbetax)
        self.append("beta_y", gdfbeamdata.CSbetay)
        self.append("alpha_y", gdfbeamdata.CSalphay)
        self.append("gamma_y", (1 + gdfbeamdata.CSalphax**2) / gdfbeamdata.CSbetay)
        self.append("beta_z", np.zeros(len(gdfbeamdata.stdx)))
        self.append("gamma_z", np.zeros(len(gdfbeamdata.stdx)))
        self.append("alpha_z", np.zeros(len(gdfbeamdata.stdx)))
        self.append("sigma_x", gdfbeamdata.stdx)
        self.append("sigma_y", gdfbeamdata.stdy)
        self.append("sigma_xp", gdfbeamdata.stdBx/gdfbeamdata.avgBz)
        self.append("sigma_yp", gdfbeamdata.stdBy/gdfbeamdata.avgBz)
        self.append("mean_x", gdfbeamdata.avgx)
        self.append("mean_y", gdfbeamdata.avgy)
        beta = np.sqrt(1 - (gamma**-2))
        if hasattr(gdfbeamdata, "stdt"):
            self.append("sigma_t", gdfbeamdata.stdt)
        else:
            self.append("sigma_t", gdfbeamdata.stdz / (beta * constants.speed_of_light))
        if hasattr(gdfbeamdata, "avgt"):
            self.append("t", gdfbeamdata.avgt)
        else:
            self.append("t", gdfbeamdata.time)
        self.append("sigma_z", gdfbeamdata.stdz)
        # self.append('sigma_cp', (gdfbeamdata.stdG / gdfbeamdata.avgG) * cp)
        self.append(
            "sigma_cp",
            (gdfbeamdata.stdG / gdfbeamdata.avgG) * cp / constants.elementary_charge,
        )
        self.append("sigma_p", (gdfbeamdata.stdG / gdfbeamdata.avgG))
        self.append("mux", np.zeros(len(gdfbeamdata.stdx)))
        self.append("muy", np.zeros(len(gdfbeamdata.stdx)))
        self.append("eta_x", np.zeros(len(gdfbeamdata.stdx)))
        self.append("eta_xp", np.zeros(len(gdfbeamdata.stdx)))
        self.append('eta_y', np.zeros(len(gdfbeamdata.stdy)))
        self.append('eta_yp', np.zeros(len(gdfbeamdata.stdy)))
        self.append("element_name", np.zeros(len(gdfbeamdata.stdx)), "")
        self.append("lattice_name", np.zeros(len(gdfbeamdata.stdx)), lattice_name)
        # ## BEAM parameters
        self.append("ecnx", np.zeros(len(gdfbeamdata.stdx)))
        self.append("ecny", np.zeros(len(gdfbeamdata.stdx)))
        self.append("eta_x_beam", np.zeros(len(gdfbeamdata.stdx)))
        self.append("eta_xp_beam", np.zeros(len(gdfbeamdata.stdx)))
        self.append("eta_y_beam", np.zeros(len(gdfbeamdata.stdx)))
        self.append("eta_yp_beam", np.zeros(len(gdfbeamdata.stdx)))
        self.append("beta_x_beam", np.zeros(len(gdfbeamdata.stdx)))
        self.append("beta_y_beam", np.zeros(len(gdfbeamdata.stdx)))
        self.append("alpha_x_beam", np.zeros(len(gdfbeamdata.stdx)))
        self.append("alpha_y_beam", np.zeros(len(gdfbeamdata.stdx)))
        self["cp_eV"] = self["cp"]
        self["sigma_cp_eV"] = self["sigma_cp"]
