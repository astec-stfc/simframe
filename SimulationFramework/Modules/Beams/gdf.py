import easygdf
import numpy as np
from ..gdf_beam import gdf_beam
from .. import constants


def write_gdf_beam_file(
    self,
    filename,
    normaliseX=False,
    normaliseZ=False,
    cathode=False,
    charge=None,
    mass=None,
):
    q = self._beam.particle_charge
    m = self._beam.particle_mass

    if self._beam.nmacro and len(self._beam.nmacro) == len(self.x):
        nmacro = abs(self._beam.nmacro)
    elif len(self._beam.charge) == len(self.x):
        nmacro = abs(self._beam.charge / q)
    else:
        nmacro = np.full(
            len(self.x),
            abs(self._beam.total_charge / constants.elementary_charge / len(self.x)),
        )
    x = (
        self.x
        if not normaliseX
        else (
            (self.x - normaliseX)
            if isinstance(normaliseX, (int, float))
            else (self.x - np.mean(self.x))
        )
    )
    z = (
        self.z
        if not normaliseZ
        else (
            (self.z - normaliseZ)
            if isinstance(normaliseZ, (int, float))
            else (self.z - np.mean(self.z))
        )
    )
    dataarray = {
        "x": x,
        "y": self.y,
        "z": z,
        "q": q,
        "m": m,
        "nmacro": nmacro,
        "GBx": self.gamma * self.Bx,
        "GBy": self.gamma * self.By,
        "GBz": self.gamma * self.Bz,
    }
    if cathode:
        dataarray["t"] = self.t
    easygdf.save_initial_distribution(filename, **dataarray)


def read_gdf_beam_file_object(self, file):
    if isinstance(file, str):
        gdfbeam = gdf_beam(file)
    elif isinstance(file, gdf_beam):
        gdfbeam = file
    else:
        raise Exception("file is not str or gdf_beam object!")
    return gdfbeam


def read_gdf_beam_file_info(self, file):
    self.reset_dicts()
    gdfbeam = gdf_beam(self, file)
    return gdfbeam


def read_gdf_beam_file(
    self,
    file=None,
    position=None,
    time=None,
    charge=None,
    longitudinal_reference="t",
    gdfbeam=None,
):
    self.reset_dicts()
    if gdfbeam is None and file is not None:
        gdfbeam = read_gdf_beam_file_object(self, file)
    elif gdfbeam is None and file is None:
        return None

    if position is not None:  # and (time is not None or block is not None):
        self.longitudinal_reference = "t"
        gdfbeamdata = gdfbeam.get_position(position)
        if gdfbeamdata is not None:
            time = None
        else:
            raise ValueError(f"GDF DID NOT find position {position}")
    elif position is None and time is not None:
        self.longitudinal_reference = "z"
        gdfbeamdata = gdfbeam.get_time(time)
        time = None
    elif position is None and time is None:
        gdfbeamdata = gdfbeam.positions[0]
    else:
        raise ValueError("Could not load gdfbeamdata; position or time not known!")
    self.filename = file
    self.code = "GPT"
    if hasattr(gdfbeamdata, "m"):
        self._beam.particle_mass = gdfbeamdata.m
    else:
        self._beam.particle_mass = np.full(
            len(gdfbeamdata.x), constants.electron_mass
        )
    self._beam.particle_rest_energy = [
        m * constants.speed_of_light**2 for m in self._beam.particle_mass
    ]
    self._beam.particle_rest_energy_eV = [
        E0 / constants.elementary_charge for E0 in self._beam.particle_rest_energy
    ]
    self._beam.particle_charge = [self._beam.sign(q) for q in gdfbeamdata.q]

    self._beam["x"] = gdfbeamdata.x
    self._beam["y"] = gdfbeamdata.y

    if hasattr(gdfbeamdata, "G"):
        self._beam.gamma = gdfbeamdata.G
    if hasattr(gdfbeamdata, "Bx"):
        if not hasattr(gdfbeamdata, "G"):
            beta = np.sqrt(gdfbeamdata.Bx**2 + gdfbeamdata.By**2 + gdfbeamdata.Bz**2)
            self._beam.gamma = 1.0 / np.sqrt(1 - beta**2)
        self._beam.px = (
            self._beam.gamma
            * gdfbeamdata.Bx
            * self._beam.particle_rest_energy
            / constants.speed_of_light
        )
        self._beam.py = (
            self._beam.gamma
            * gdfbeamdata.By
            * self._beam.particle_rest_energy
            / constants.speed_of_light
        )
        self._beam.pz = (
            self._beam.gamma
            * gdfbeamdata.Bz
            * self._beam.particle_rest_energy
            / constants.speed_of_light
        )
    elif hasattr(gdfbeamdata, "GBx"):
        self._beam.px = (
            gdfbeamdata.GBx
            * self._beam.particle_rest_energy
            / constants.speed_of_light
        )
        self._beam.py = (
            gdfbeamdata.GBy
            * self._beam.particle_rest_energy
            / constants.speed_of_light
        )
        self._beam.pz = (
            gdfbeamdata.GBz
            * self._beam.particle_rest_energy
            / constants.speed_of_light
        )
    else:
        raise Exception("GDF File does not have Bx or GBx!")

    if hasattr(gdfbeamdata, "z") and longitudinal_reference == "z":
        self._beam["z"] = gdfbeamdata.z
        self._beam["t"] = np.full(len(self.z), 0)
    elif hasattr(gdfbeamdata, "t") and longitudinal_reference == "t":
        self._beam["t"] = gdfbeamdata.t
        self._beam["z"] = (-1 * gdfbeamdata.Bz * constants.speed_of_light) * (
            gdfbeamdata.t - np.mean(gdfbeamdata.t)
        ) + gdfbeamdata.z
    else:
        if hasattr(gdfbeamdata, "z") and self["longitudinal_reference"] == "z":
            self._beam["z"] = gdfbeamdata.z
            self._beam["t"] = gdfbeamdata.z / (
                -1 * gdfbeamdata.Bz * constants.speed_of_light
            )
            self._beam["t"][self._beam["t"] == np.inf] = 0
        elif hasattr(gdfbeamdata, "t"):
            self._beam["t"] = gdfbeamdata.t
            self._beam["z"] = (-1 * gdfbeamdata.Bz * constants.speed_of_light) * (
                gdfbeamdata.t - np.mean(gdfbeamdata.t)
            ) + gdfbeamdata.z

    if hasattr(gdfbeamdata, "q") and hasattr(gdfbeamdata, "nmacro"):
        self._beam["charge"] = gdfbeamdata.q * gdfbeamdata.nmacro
        self._beam["total_charge"] = np.sum(self._beam["charge"])
        self._beam["nmacro"] = gdfbeamdata.nmacro
    elif hasattr(gdfbeamdata, "q"):
        self._beam["charge"] = gdfbeamdata.q
        self._beam["total_charge"] = np.sum(self._beam["charge"])
        self._beam["nmacro"] = self._beam["charge"] / constants.elementary_charge
    else:
        if charge is None:
            self._beam["total_charge"] = 0
            self._beam["charge"] = np.full(len(self.z), 0)
        else:
            self._beam["total_charge"] = charge

    if hasattr(gdfbeamdata, "nmacro"):
        self._beam["nmacro"] = gdfbeamdata.nmacro
    else:
        self._beam["nmacro"] = np.full(len(self.z), 1)
    return gdfbeam
