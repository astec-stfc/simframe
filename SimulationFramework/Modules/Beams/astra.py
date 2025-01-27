import sys
import numpy as np
import csv
from .. import constants


def read_csv_file(self, file, delimiter=" "):
    with open(file, "r") as f:
        data = np.array(
            [
                line
                for line in csv.reader(
                    f,
                    delimiter=delimiter,
                    quoting=csv.QUOTE_NONNUMERIC,
                    skipinitialspace=True,
                )
            ]
        )
    return data


def write_csv_file(self, file, data):
    if sys.version_info[0] > 2:
        with open(file, "w", newline="") as f:
            writer = csv.writer(
                f, delimiter=" ", quoting=csv.QUOTE_NONNUMERIC, skipinitialspace=True
            )
            [writer.writerow(line) for line in data]
    else:
        with open(file, "wb") as f:
            writer = csv.writer(
                f, delimiter=" ", quoting=csv.QUOTE_NONNUMERIC, skipinitialspace=True
            )
            [writer.writerow(line) for line in data]


def read_astra_beam_file(self, file, normaliseZ=False, keepLost=False):
    self.reset_dicts()
    data = read_csv_file(self, file)
    self.filename = file
    interpret_astra_data(self, data, normaliseZ=normaliseZ, keepLost=keepLost)


def interpret_astra_data(self, data, normaliseZ=False, keepLost=False):
    if not keepLost:
        data = [d for d in data if d[-1] >= 0]
    x, y, z, cpx, cpy, cpz, clock, charge, index, status = np.transpose(data)
    zref = z[0]
    self["code"] = "ASTRA"
    self._beam["reference_particle"] = data[0]
    self._beam["toffset"] = 1e-9 * data[0][6]
    # if normaliseZ:
    #     self._beam['reference_particle'][2] = 0
    self["longitudinal_reference"] = "z"
    # znorm = self.normalise_to_ref_particle(z, subtractmean=True)
    z = self.normalise_to_ref_particle(z, subtractmean=False)
    cpz = self.normalise_to_ref_particle(cpz, subtractmean=False)
    clock = self.normalise_to_ref_particle(clock, subtractmean=True)
    self._beam["px"] = cpx * self.q_over_c
    self._beam["py"] = cpy * self.q_over_c
    self._beam["pz"] = cpz * self.q_over_c
    self._beam["clock"] = 1.0e-9 * clock
    self._beam["charge"] = 1.0e-9 * charge
    self._beam["status"] = status
    self._beam["z"] = z
    self._beam["particle_mass"] = [self.mass_index[i] for i in index]
    self._beam["particle_rest_energy"] = [
        m * constants.speed_of_light**2 for m in self._beam["particle_mass"]
    ]
    self._beam["particle_rest_energy_eV"] = [
        E0 / constants.elementary_charge for E0 in self._beam["particle_rest_energy"]
    ]
    self._beam["particle_charge"] = [
        constants.elementary_charge * self.charge_sign_index[i] for i in index
    ]
    # print self.Bz
    self._beam["t"] = [
        clock if status == -1 else ((z - zref) / (-1 * Bz * constants.speed_of_light))
        for status, z, Bz, clock in zip(
            self._beam["status"], z, self.Bz, self._beam["clock"]
        )
    ]
    # self._beam['t'] = self.z / (1 * self.Bz * constants.speed_of_light)#[time if status is -1 else 0 for time, status in zip(clock, status)]#
    self._beam["x"] = x  # - self.xp * (self.t - np.mean(self.t))
    self._beam["y"] = y  # - self.yp * (self.t - np.mean(self.t))
    self._beam["total_charge"] = np.sum(1.0e-9 * charge)
    self._beam["nmacro"] = np.array(
        np.array(self._beam["charge"]) / self._beam["particle_charge"]
    )


def read_csrtrack_beam_file(self, file):
    self.reset_dicts()
    data = self.read_csv_file(file)
    self["code"] = "CSRTrack"
    self._beam["reference_particle"] = data[0]
    self["longitudinal_reference"] = "z"
    z, x, y, cpz, cpx, cpy, charge = np.transpose(data[1:])
    z = self.normalise_to_ref_particle(z, subtractmean=False)
    cpz = self.normalise_to_ref_particle(cpz, subtractmean=False)
    self._beam["x"] = x
    self._beam["y"] = y
    self._beam["z"] = z
    self._beam["px"] = cpx * self.q_over_c
    self._beam["py"] = cpy * self.q_over_c
    self._beam["pz"] = cpz * self.q_over_c
    self._beam["clock"] = np.full(len(self.x), 0)
    self._beam["clock"][0] = data[0, 0] * 1e-9
    self._beam["status"] = np.full(len(self.x), 1)
    self._beam["t"] = self.z / (
        -1 * self.Bz * constants.speed_of_light
    )  # [time if status is -1 else 0 for time, status in zip(clock, self._beam['status'])]
    self._beam["charge"] = charge
    self._beam["total_charge"] = np.sum(self._beam["charge"])


def read_pacey_beam_file(self, fileName, charge=250e-12):
    self.reset_dicts()
    data = self.read_csv_file(fileName, delimiter="\t")
    self.filename = fileName
    self["code"] = "TPaceyASTRA"
    self["longitudinal_reference"] = "z"
    x, y, z, cpx, cpy, cpz = np.transpose(data)
    # cp = np.sqrt(cpx**2 + cpy**2 + cpz**2)
    self._beam["x"] = x
    self._beam["y"] = y
    self._beam["z"] = z
    self._beam["px"] = cpx * self.q_over_c
    self._beam["py"] = cpy * self.q_over_c
    self._beam["pz"] = cpz * self.q_over_c
    self._beam["t"] = [
        (z / (-1 * Bz * constants.speed_of_light)) for z, Bz in zip(self.z, self.Bz)
    ]
    # self._beam['t'] = self.z / (1 * self.Bz * constants.speed_of_light)#[time if status is -1 else 0 for time, status in zip(clock, status)]#
    self._beam["total_charge"] = charge
    self._beam["charge"] = []


def convert_csrtrackfile_to_astrafile(self, csrtrackfile, astrafile):
    data = read_csv_file(self, csrtrackfile)
    z, x, y, cpz, cpx, cpy, charge = np.transpose(data[1:])
    charge = -charge * 1e9
    clock0 = (data[0, 0] / constants.speed_of_light) * 1e9
    clock = np.full(len(x), 0)
    clock[0] = clock0
    index = np.full(len(x), 1)
    status = np.full(len(x), 5)
    array = np.array([x, y, z, cpx, cpy, cpz, clock, charge, index, status]).transpose()
    write_csv_file(self, astrafile, array)


def cdist(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


def find_nearest_vector(self, nodes, node):
    return cdist([node], nodes).argmin()


def rms(self, x, axis=None):
    return np.sqrt(np.mean(x**2, axis=axis))


def create_ref_particle(self, array, index=0, subtractmean=False):
    array[1:] = array[0] + array[1:]
    if subtractmean:
        array = array - np.mean(array)
    return array


def write_astra_beam_file(
    self,
    file: str,
    index: int = None,
    status: int = 5,
    charge: float | None = None,
    normaliseZ: bool = False,
):
    if not isinstance(index, (list, tuple, np.ndarray)):
        if len(self._beam["charge"]) == len(self._beam.x):
            chargevector = 1e9 * self._beam["charge"]
        else:
            chargevector = np.full(
                len(self._beam.x), 1e9 * self._beam["total_charge"] / len(self._beam.x)
            )
    if index is not None:
        indexvector = np.full(len(self._beam.x), index)
    else:
        # print('write_astra_beam_file: index not found')
        indexvector = self._beam.particle_index
    # print('write_astra_beam_file: index =', indexvector)
    # exit()
    statusvector = (
        self._beam["status"]
        if "status" in self._beam
        else (
            status
            if isinstance(status, (list, tuple, np.ndarray))
            else np.full(len(self._beam.x), status)
        )
    )
    """ if a particle is emitting from the cathode it's z value is 0 and it's clock value is finite, otherwise z is finite and clock is irrelevant (thus zero) """
    if self["longitudinal_reference"] == "t":
        zvector = [
            0 if status == -1 and t == 0 else z
            for status, z, t in zip(statusvector, self._beam.z, self._beam.t)
        ]
    else:
        zvector = self._beam.z
    """ if the clock value is finite, we calculate it from the z value, using Betaz """
    # clockvector = [1e9*z / (1 * Bz * constants.speed_of_light) if status == -1 and t == 0 else 1.0e9*t for status, z, t, Bz in zip(statusvector, self.z, self.t, self.Bz)]
    clockvector = [
        1.0e9 * t
        for status, z, t, Bz in zip(
            statusvector, self._beam.z, self._beam.t, self._beam.Bz
        )
    ]
    """ this is the ASTRA array in all it's glory """
    array = np.array(
        [
            self._beam.x,
            self._beam.y,
            zvector,
            self._beam.cpx,
            self._beam.cpy,
            self._beam.cpz,
            clockvector,
            chargevector,
            indexvector,
            statusvector,
        ]
    ).transpose()
    if hasattr(self._beam, "reference_particle"):
        ref_particle = self._beam["reference_particle"]
        # print 'we have a reference particle! ', ref_particle
        # np.insert(array, 0, ref_particle, axis=0)
    else:
        """take the rms - if the rms is 0 set it to 1, so we don't get a divide by error"""
        rms_vector = [a if abs(a) > 0 else 1 for a in rms(self, array, axis=0)]
        """ normalise the array """
        norm_array = array / rms_vector
        """ take the meen of the normalised array """
        mean_vector = np.mean(norm_array, axis=0)
        """ find the index of the vector that is closest to the mean - if you read in an ASTRA file, this should actually return the reference particle! """
        nearest_idx = find_nearest_vector(self, norm_array, mean_vector)
        ref_particle = array[nearest_idx]
        """ set the closest mean vector to be in position 0 in the array """
        array = np.roll(array, -1 * nearest_idx, axis=0)

    """ normalise Z to the reference particle """
    array[1:, 2] = array[1:, 2] - ref_particle[2]
    """ should we leave Z as the reference value, set it to 0, or set it to be some offset? """
    if normaliseZ is not False:
        array[0, 2] = 0
    if isinstance(normaliseZ, (int, float)):
        # print('Setting z offset', normaliseZ)
        array[0, 2] += normaliseZ
    """ normalise pz and the clock """
    # print('Mean pz = ', np.mean(array[:,5]))
    array[1:, 5] = array[1:, 5] - ref_particle[5]
    array[0, 6] = array[0, 6] + ref_particle[6]
    np.savetxt(
        file,
        array,
        fmt=(
            "%.12e",
            "%.12e",
            "%.12e",
            "%.12e",
            "%.12e",
            "%.12e",
            "%.12e",
            "%.12e",
            "%d",
            "%d",
        ),
    )
