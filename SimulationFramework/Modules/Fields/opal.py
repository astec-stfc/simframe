import numpy as np
from warnings import warn

def write_opal_field_file(self, frequency: float=None, radius: float=None, fourier: int=100):
    length = self.length
    opal_file = self.filename.replace(".hdf5", ".opal")
    data = None
    header = None
    fourier_ratio = fourier / length
    if fourier_ratio > 1.0:
        warn("Too many fourier components provided; defaulting to 1/100")
        fourier = int(length / 100)
    elif fourier_ratio < 0.01:
        warn("Not enough fourier components provided; defaulting to 1/100")
        fourier = int(length / 100)
    if self.field_type == "1DMagnetoStatic":
        zmin = min(self.z.value.val) * 1e-2
        zmax = max(self.z.value.val) * 1e-2
        data = self.Bz.value.val
        head = ["1DMagnetoStatic", str(fourier)]
        if not radius:
            warn("Magnet radius not provided; defaulting to 10cm")
            radius = 0.1
        rvals = [str(0), str(radius * 100), str(length), fourier]
        zvals = [str(zmin), str(zmax), str(length)]
        header = [head, rvals, zvals]
    elif self.field_type == "1DElectroStatic":
        if not frequency:
            warn("RF Frequency not provided to field class")
            return
        head = ["ASTRADynamic", str(fourier)]
        freq = [str(frequency * 1e-6)]
        zdata = self.z.value.val
        ezdata = self.Ez.value.val
        data = np.transpose([zdata, ezdata])
        header = [head, freq]
    else:
        warn(f"Field type {self.field_type} not supported for OPAL")
    if data:
        with open(f"{opal_file}", "w") as f:
            for h in header:
                f.write(" ".join([str(x) for x in h]) + "\n")
            for d in data:
                f.write(" ".join([str(x) for x in d]) + "\n")
