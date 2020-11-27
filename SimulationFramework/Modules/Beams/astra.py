import sys
import time
import numpy as np
import csv
import scipy.constants as constants

def read_csv_file(self, file, delimiter=' '):
    with open(file, 'r') as f:
        data = np.array([l for l in csv.reader(f, delimiter=delimiter,  quoting=csv.QUOTE_NONNUMERIC, skipinitialspace=True)])
    return data

def write_csv_file(self, file, data):
    if sys.version_info[0] > 2:
        with open(file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=' ',  quoting=csv.QUOTE_NONNUMERIC, skipinitialspace=True)
            [writer.writerow(l) for l in data]
    else:
        with open(file, 'wb') as f:
            writer = csv.writer(f, delimiter=' ',  quoting=csv.QUOTE_NONNUMERIC, skipinitialspace=True)
            [writer.writerow(l) for l in data]

def read_astra_beam_file(self, file, normaliseZ=False):
    starttime = time.time()
    self.reset_dicts()
    data = read_csv_file(self, file)
    self.filename = file
    interpret_astra_data(self, data, normaliseZ=normaliseZ)

def interpret_astra_data(self, data, normaliseZ=False):
    x, y, z, cpx, cpy, cpz, clock, charge, index, status = np.transpose(data)
    zref = z[0]
    self['code'] = "ASTRA"
    self._beam['reference_particle'] = data[0]
    # if normaliseZ:
    #     self._beam['reference_particle'][2] = 0
    self['longitudinal_reference'] = 'z'
    znorm = self.normalise_to_ref_particle(z, subtractmean=True)
    z = self.normalise_to_ref_particle(z, subtractmean=False)
    cpz = self.normalise_to_ref_particle(cpz, subtractmean=False)
    clock = self.normalise_to_ref_particle(clock, subtractmean=True)
    cp = np.sqrt(cpx**2 + cpy**2 + cpz**2)
    self._beam['x'] = x
    self._beam['y'] = y
    self._beam['z'] = z
    self._beam['px'] = cpx * self.q_over_c
    self._beam['py'] = cpy * self.q_over_c
    self._beam['pz'] = cpz * self.q_over_c
    self._beam['clock'] = 1.0e-9*clock
    self._beam['charge'] = 1.0e-9*charge
    self._beam['index'] = index
    self._beam['status'] = status
    # print self.Bz
    self._beam['t'] = [clock if status == -1 else ((z-zref) / (-1 * Bz * constants.speed_of_light)) for status, z, Bz, clock in zip(self._beam['status'], z, self.Bz, self._beam['clock'])]
    # self._beam['t'] = self.z / (1 * self.Bz * constants.speed_of_light)#[time if status is -1 else 0 for time, status in zip(clock, status)]#
    self._beam['total_charge'] = np.sum(self._beam['charge'])

def read_csrtrack_beam_file(self, file):
    self.reset_dicts()
    data = self.read_csv_file(file)
    self['code'] = "CSRTrack"
    self._beam['reference_particle'] = data[0]
    self['longitudinal_reference'] = 'z'
    z, x, y, cpz, cpx, cpy, charge = np.transpose(data[1:])
    z = self.normalise_to_ref_particle(z, subtractmean=False)
    cpz = self.normalise_to_ref_particle(cpz, subtractmean=False)
    cp = np.sqrt(cpx**2 + cpy**2 + cpz**2)
    self._beam['x'] = x
    self._beam['y'] = y
    self._beam['z'] = z
    self._beam['px'] = cpx * self.q_over_c
    self._beam['py'] = cpy * self.q_over_c
    self._beam['pz'] = cpz * self.q_over_c
    self._beam['clock'] = np.full(len(self.x), 0)
    self._beam['clock'][0] = data[0, 0] * 1e-9
    self._beam['index'] = np.full(len(self.x), 5)
    self._beam['status'] = np.full(len(self.x), 1)
    self._beam['t'] = self.z / (-1 * self.Bz * constants.speed_of_light)# [time if status is -1 else 0 for time, status in zip(clock, self._beam['status'])]
    self._beam['charge'] = charge
    self._beam['total_charge'] = np.sum(self._beam['charge'])

def read_pacey_beam_file(self, fileName, charge=250e-12):
    self.reset_dicts()
    data = self.read_csv_file(fileName, delimiter='\t')
    self.filename = fileName
    self['code'] = "TPaceyASTRA"
    self['longitudinal_reference'] = 'z'
    x, y, z, cpx, cpy, cpz = np.transpose(data)
    cp = np.sqrt(cpx**2 + cpy**2 + cpz**2)
    self._beam['x'] = x
    self._beam['y'] = y
    self._beam['z'] = z
    self._beam['px'] = cpx * self.q_over_c
    self._beam['py'] = cpy * self.q_over_c
    self._beam['pz'] = cpz * self.q_over_c
    self._beam['t'] = [(z / (-1 * Bz * constants.speed_of_light)) for z, Bz in zip(self.z, self.Bz)]
    # self._beam['t'] = self.z / (1 * self.Bz * constants.speed_of_light)#[time if status is -1 else 0 for time, status in zip(clock, status)]#
    self._beam['total_charge'] = charge
    self._beam['charge'] = []

def convert_csrtrackfile_to_astrafile(self, csrtrackfile, astrafile):
    data = read_csv_file(self, csrtrackfile)
    z, x, y, cpz, cpx, cpy, charge = np.transpose(data[1:])
    charge = -charge*1e9
    clock0 = (data[0, 0] / constants.speed_of_light) * 1e9
    clock = np.full(len(x), 0)
    clock[0] = clock0
    index = np.full(len(x), 1)
    status = np.full(len(x), 5)
    array = np.array([x, y, z, cpx, cpy, cpz, clock, charge, index, status]).transpose()
    write_csv_file(self, astrafile, array)

def find_nearest_vector(self, nodes, node):
    return cdist([node], nodes).argmin()

def rms(self, x, axis=None):
    return np.sqrt(np.mean(x**2, axis=axis))

def create_ref_particle(self, array, index=0, subtractmean=False):
    array[1:] = array[0] + array[1:]
    if subtractmean:
        array = array - np.mean(array)
    return array

def write_astra_beam_file(self, file, index=1, status=5, charge=None, normaliseZ=False):
    if not isinstance(index,(list, tuple, np.ndarray)):
        if len(self._beam['charge']) == len(self.x):
            chargevector = 1e9*self._beam['charge']
        else:
            chargevector = np.full(len(self.x), 1e9*self.charge/len(self.x))
    if not isinstance(index,(list, tuple, np.ndarray)):
        indexvector = np.full(len(self.x), index)
    statusvector = self._beam['status'] if 'status' in self._beam else status if isinstance(status,(list, tuple, np.ndarray)) else np.full(len(self.x), status)
    ''' if a particle is emitting from the cathode it's z value is 0 and it's clock value is finite, otherwise z is finite and clock is irrelevant (thus zero) '''
    if self['longitudinal_reference'] == 't':
        zvector = [0 if status == -1 and t == 0 else z for status, z, t in zip(statusvector, self.z, self.t)]
    else:
        zvector = self.z
    ''' if the clock value is finite, we calculate it from the z value, using Betaz '''
    # clockvector = [1e9*z / (1 * Bz * constants.speed_of_light) if status == -1 and t == 0 else 1.0e9*t for status, z, t, Bz in zip(statusvector, self.z, self.t, self.Bz)]
    clockvector = [1.0e9*t for status, z, t, Bz in zip(statusvector, self.z, self.t, self.Bz)]
    ''' this is the ASTRA array in all it's glory '''
    array = np.array([self.x, self.y, zvector, self.cpx, self.cpy, self.cpz, clockvector, chargevector, indexvector, statusvector]).transpose()
    if 'reference_particle' in self._beam:
        ref_particle = self._beam['reference_particle']
        # print 'we have a reference particle! ', ref_particle
        # np.insert(array, 0, ref_particle, axis=0)
    else:
        ''' take the rms - if the rms is 0 set it to 1, so we don't get a divide by error '''
        rms_vector = [a if abs(a) > 0 else 1 for a in self.rms(array, axis=0)]
        ''' normalise the array '''
        norm_array = array / rms_vector
        ''' take the meen of the normalised array '''
        mean_vector = np.mean(norm_array, axis=0)
        ''' find the index of the vector that is closest to the mean - if you read in an ASTRA file, this should actually return the reference particle! '''
        nearest_idx = self.find_nearest_vector(norm_array, mean_vector);
        ref_particle = array[nearest_idx]
        ''' set the closest mean vector to be in position 0 in the array '''
        array = np.roll(array, -1*nearest_idx, axis=0)

    ''' normalise Z to the reference particle '''
    array[1:,2] = array[1:,2] - ref_particle[2]
    ''' should we leave Z as the reference value, set it to 0, or set it to be some offset? '''
    if not normaliseZ is False:
        array[0,2] = 0
    if not isinstance(normaliseZ,(bool)):
        array[0,2] += normaliseZ
    ''' normalise pz and the clock '''
    # print('Mean pz = ', np.mean(array[:,5]))
    array[1:,5] = array[1:,5] - ref_particle[5]
    array[0,6] = array[0,6] + ref_particle[6]
    np.savetxt(file, array, fmt=('%.12e','%.12e','%.12e','%.12e','%.12e','%.12e','%.12e','%.12e','%d','%d'))
